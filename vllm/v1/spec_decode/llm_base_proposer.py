# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import ast
import dataclasses
import os
import time
from importlib.util import find_spec
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import torch.nn as nn

from vllm.compilation.cuda_graph import CUDAGraphOptions, CUDAGraphWrapper
from vllm.config import (
    CUDAGraphMode,
    VllmConfig,
    get_layers_from_vllm_config,
    replace,
)
from vllm.distributed.parallel_state import get_pp_group
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.model_loader import get_model
from vllm.model_executor.models import supports_multimodal
from vllm.model_executor.models.deepseek_eagle3 import Eagle3DeepseekV2ForCausalLM
from vllm.model_executor.models.interfaces import SupportsMultiModal
from vllm.model_executor.models.llama_eagle3 import Eagle3LlamaForCausalLM
from vllm.model_executor.models.qwen3_dflash import DFlashQwen3ForCausalLM
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.platforms import current_platform
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.attention.backends.tree_attn import (
    TreeAttentionMetadata,
    TreeAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.triton_attn import TritonAttentionMetadata
from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher
from vllm.v1.kv_cache_interface import KVCacheConfig, UniformTypeKVCacheSpecs
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import _SAMPLING_EPS
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.spec_decode.utils import (
    PADDING_SLOT_ID,
    compute_new_slot_mapping,
    copy_and_expand_eagle_inputs_kernel,
    eagle_prepare_inputs_padded_kernel,
    eagle_prepare_next_token_padded_kernel,
    eagle_step_update_slot_mapping_and_metadata,
    extend_all_queries_by_N,
    next_power_of_2,
)
from vllm.v1.utils import CpuGpuBuffer
from vllm.v1.worker.dp_utils import coordinate_batch_across_dp
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch
from vllm.v1.worker.utils import AttentionGroup

logger = init_logger(__name__)


class SpecDecodeBaseProposer:
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        pass_hidden_states_to_model: bool,
        runner=None,
    ):
        self.vllm_config = vllm_config
        assert vllm_config.speculative_config is not None
        self.speculative_config = vllm_config.speculative_config
        self.draft_model_config = self.speculative_config.draft_model_config
        self.method = self.speculative_config.method
        self.pass_hidden_states_to_model = pass_hidden_states_to_model

        self.device = device
        self.dtype = vllm_config.model_config.dtype
        self.max_model_len = vllm_config.model_config.max_model_len
        self.dp_rank = vllm_config.parallel_config.data_parallel_rank
        self.num_speculative_tokens = self.speculative_config.num_speculative_tokens
        draft_hf_config = getattr(self.draft_model_config, "hf_config", None)
        self._share_mtp_indices = getattr(
            draft_hf_config, "index_share_for_mtp_iteration", False
        )

        # We need to get the hidden size from the draft model config because
        # the draft model's hidden size can be different from the target model's
        # hidden size (e.g., Llama 3.3 70B).
        self.hidden_size = self.draft_model_config.get_hidden_size()
        self.inputs_embeds_size = self.draft_model_config.get_inputs_embeds_size()

        # Unifying eagle, draft model, and parallel drafting support.
        # DFlash always uses parallel drafting (all tokens in one pass),
        # but has an additional slot for the next_token_id (does not shift like EAGLE)
        self.parallel_drafting: bool = self.speculative_config.parallel_drafting
        self.extra_slots_per_request = (
            1 if not self.parallel_drafting else self.num_speculative_tokens
        )
        self.net_num_new_slots_per_request = self.extra_slots_per_request - (
            1 if (self.pass_hidden_states_to_model and self.method != "dflash") else 0
        )
        self.needs_extra_input_slots = self.net_num_new_slots_per_request > 0

        self.parallel_drafting_token_id: int = 0
        self.parallel_drafting_hidden_state_tensor: torch.Tensor | None = None
        if self.parallel_drafting:
            self._init_parallel_drafting_params()
        self.use_local_argmax_reduction: bool = (
            self.speculative_config.use_local_argmax_reduction
        )

        self.max_batch_size = vllm_config.scheduler_config.max_num_seqs
        self.max_num_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        self.token_arange_np = np.arange(self.max_num_tokens)

        # Can be specialized by methods like DFlash to reduce the limit
        self.max_query_tokens = self.max_num_tokens
        self.max_positions = self.max_num_tokens

        # Multi-modal data support
        self.mm_registry = MULTIMODAL_REGISTRY
        self.supports_mm_inputs = self.mm_registry.supports_multimodal_inputs(
            vllm_config.model_config
        )

        self.draft_attn_groups: list[AttentionGroup] = []
        self.kv_cache_gid: int = -1
        self.eagle3_use_aux_hidden_state: bool = (
            self._get_eagle3_use_aux_hidden_state_from_config()
        )

        self.compilation_config = self.vllm_config.compilation_config

        # Cudagraph dispatcher for PIECEWISE-only dispatching in eagle.
        # Keys are initialized later via initialize_cudagraph_keys() called from
        # gpu_model_runner._check_and_update_cudagraph_mode after
        # adjust_cudagraph_sizes_for_spec_decode is called.
        self.cudagraph_dispatcher = CudagraphDispatcher(self.vllm_config)
        self._oscar_mtp_draft_graph_capture_metadata: dict[int, Any] = {}
        self._oscar_mtp_draft_decode_capture_sizes: tuple[int, ...] = ()
        draft_margin_threshold = os.environ.get(
            "VLLM_OSCAR_MTP_DRAFT_MARGIN_EARLY_STOP"
        )
        self._oscar_mtp_draft_margin_threshold = (
            float(draft_margin_threshold) if draft_margin_threshold else None
        )
        draft_margin_sentinel = os.environ.get(
            "VLLM_OSCAR_MTP_DRAFT_MARGIN_EARLY_STOP_SENTINEL"
        )
        self._oscar_mtp_draft_margin_sentinel = (
            Path(draft_margin_sentinel) if draft_margin_sentinel else None
        )
        self._oscar_mtp_dynamic_draft_enabled = (
            self._oscar_mtp_draft_margin_threshold is not None
            and self._oscar_mtp_draft_margin_sentinel is None
        )
        self._oscar_mtp_stop_after_current_draft = False

        # persistent buffers for cuda graph
        self.input_ids = torch.zeros(
            self.max_num_tokens, dtype=torch.int32, device=device
        )
        # Use draft model's M-RoPE setting, not target model's
        # Draft models may be text-only even if target is multimodal
        self.uses_mrope = self.draft_model_config.uses_mrope
        self.uses_xdrope_dim = self.vllm_config.model_config.uses_xdrope_dim
        self.draft_uses_xdrope_dim = self.draft_model_config.uses_xdrope_dim
        if self.uses_mrope:
            # NOTE: `mrope_positions` is implemented with one additional dummy
            # position on purpose to make it non-contiguous so that it can work
            # with torch compile.
            # See detailed explanation in https://github.com/vllm-project/vllm/pull/12128#discussion_r1926431923

            # NOTE: When M-RoPE is enabled, position ids are 3D regardless of
            # the modality of inputs. For text-only inputs, each dimension has
            # identical position IDs, making M-RoPE functionally equivalent to
            # 1D-RoPE.
            # See page 5 of https://arxiv.org/abs/2409.12191
            self.mrope_positions = torch.zeros(
                (3, self.max_positions + 1), dtype=torch.int64, device=device
            )
        elif self.uses_xdrope_dim > 0 and self.draft_uses_xdrope_dim > 0:
            self.xdrope_positions = torch.zeros(
                (self.uses_xdrope_dim, self.max_positions + 1),
                dtype=torch.int64,
                device=device,
            )
        else:
            # RoPE need (max_num_tokens,)
            self.positions = torch.zeros(
                self.max_positions,
                dtype=torch.int64,
                device=device,
            )
        self.hidden_states = torch.zeros(
            (self.max_num_tokens, self.hidden_size), dtype=self.dtype, device=device
        )

        # Will be set when we initialize the attention backend
        self.block_size: int = -1

        # We need +1 here because the arange is used to set query_start_loc,
        # which has one more element than batch_size.
        max_num_slots_for_arange = max(self.max_batch_size + 1, self.max_num_tokens)
        self.arange = torch.arange(
            max_num_slots_for_arange, device=device, dtype=torch.int32
        )

        if self.needs_extra_input_slots:
            self._raise_if_padded_drafter_batch_disabled()
            self._raise_if_multimodal()
            self._raise_if_mrope()

        self.is_rejected_token_mask: torch.Tensor | None = None
        self.is_masked_token_mask: torch.Tensor | None = None
        if self.needs_extra_input_slots:
            # For draft models and parallel drafting, we need to keep track of
            # which tokens are rejected to update the slot mapping with padding slots.
            self.is_rejected_token_mask = torch.zeros(
                (self.max_num_tokens,), dtype=torch.bool, device=device
            )
            # For parallel drafting, we also need to keep track of which tokens
            # are parallel-padding tokens used to sample at later positions.
            # We populate this tensor even when using draft models for simplicity.
            self.is_masked_token_mask = torch.zeros(
                (self.max_num_tokens,), dtype=torch.bool, device=device
            )

        self.inputs_embeds = torch.zeros(
            (self.max_num_tokens, self.inputs_embeds_size),
            dtype=self.dtype,
            device=device,
        )

        self.backup_next_token_ids = CpuGpuBuffer(
            self.max_batch_size,
            dtype=torch.int32,
            pin_memory=is_pin_memory_available(),
            device=device,
            with_numpy=True,
        )

        # The local SpeculativeConfig exposes probabilistic drafting through
        # rejection_sample_method.  In that mode retain q(x) for each draft
        # token so the verifier can use the lossless min(1, p(x) / q(x)) test.
        self._enable_probabilistic_draft_probs = (
            self.speculative_config.rejection_sample_method == "probabilistic"
        )
        self._last_draft_probs: torch.Tensor | None = None

        self._slot_mapping_buffer = torch.zeros(
            self.max_positions,
            dtype=torch.int64,
            device=device,
        )

        # Determine allowed attention backends once during initialization.
        self.allowed_attn_types: tuple | None = None
        if current_platform.is_rocm():
            from vllm.v1.attention.backends.mla.indexer import (
                DeepseekV32IndexerMetadata,
            )
            from vllm.v1.attention.backends.mla.rocm_aiter_mla_sparse import (
                ROCMAiterMLASparseMetadata,
            )
            from vllm.v1.attention.backends.rocm_attn import RocmAttentionMetadata

            rocm_types = [
                TritonAttentionMetadata,
                RocmAttentionMetadata,
                ROCMAiterMLASparseMetadata,
                DeepseekV32IndexerMetadata,
            ]
            # ROCM_AITER_FA is an optional backend
            # We check is_enabled() here to avoid importing the backend module during
            # auto-discovery when VLLM_ROCM_USE_AITER=0, which would trigger aiter
            # import and JIT compilation warnings. Explicit backend selection via
            # attention_config still works because the backend module is loaded
            # directly when selected, not through this auto-discovery path.
            # Check if backend module exists to allow explicit selection
            if find_spec(
                AttentionBackendEnum.ROCM_AITER_FA.get_path(include_classname=False)
            ):
                from vllm.v1.attention.backends.rocm_aiter_fa import (
                    AiterFlashAttentionMetadata,
                )

                rocm_types.append(AiterFlashAttentionMetadata)

            # TRITON_MLA backend support for MLA models (e.g., DeepSeek)
            from vllm.model_executor.layers.attention.mla_attention import (
                MLACommonMetadata,
            )

            rocm_types.append(MLACommonMetadata)

            # FlexAttention backend support
            from vllm.v1.attention.backends.flex_attention import FlexAttentionMetadata

            rocm_types.append(FlexAttentionMetadata)

            self.allowed_attn_types = tuple(rocm_types)

        # Parse the speculative token tree.
        spec_token_tree = self.speculative_config.speculative_token_tree
        assert spec_token_tree is not None
        self.tree_choices: list[tuple[int, ...]] = ast.literal_eval(spec_token_tree)
        tree_depth = len(self.tree_choices[-1])
        # Precompute per-level properties of the tree.
        num_drafts_per_level = [0] * tree_depth
        for node in self.tree_choices:
            num_drafts_per_level[len(node) - 1] += 1
        self.cu_drafts_per_level = [num_drafts_per_level[0]]
        self.child_drafts_per_level = [num_drafts_per_level[0]]
        for level in range(1, tree_depth):
            self.cu_drafts_per_level.append(
                self.cu_drafts_per_level[-1] + num_drafts_per_level[level]
            )
            self.child_drafts_per_level.append(
                num_drafts_per_level[level] // num_drafts_per_level[level - 1]
            )
        # Precompute draft position offsets in flattened tree.
        self.tree_draft_pos_offsets = torch.arange(
            1, len(self.tree_choices) + 1, device=device, dtype=torch.int32
        ).repeat(self.max_batch_size, 1)

    def _raise_if_padded_drafter_batch_disabled(self):
        if self.speculative_config.disable_padded_drafter_batch:
            raise NotImplementedError(
                "Speculative Decoding with draft models or parallel drafting only "
                "supports padded drafter batch. Please unset "
                "disable_padded_drafter_batch in the speculative_config."
            )

    def _raise_if_multimodal(self):
        if self.supports_mm_inputs:
            raise NotImplementedError(
                "Speculative Decoding with draft models or parallel drafting "
                "does not support multimodal models yet"
            )

    def _raise_if_mrope(self):
        if self.draft_model_config.uses_mrope:
            raise NotImplementedError(
                "Speculative Decoding with draft models or parallel drafting "
                "does not support M-RoPE yet"
            )

    def _init_parallel_drafting_params(self):
        # For parallel drafting, we need the token ID to use for masked slots
        # And for EAGLE + parallel drafting, we need the hidden state tensor to use
        # for those masked slots.

        model_hf_config = self.draft_model_config.hf_config
        # DFlash stores mask_token_id in dflash_config
        dflash_config = getattr(model_hf_config, "dflash_config", None)
        if dflash_config and "mask_token_id" in dflash_config:
            self.parallel_drafting_token_id = dflash_config["mask_token_id"]
        elif hasattr(model_hf_config, "pard_token"):
            self.parallel_drafting_token_id = model_hf_config.pard_token
        elif hasattr(model_hf_config, "ptd_token_id"):
            self.parallel_drafting_token_id = model_hf_config.ptd_token_id
        else:
            raise ValueError(
                "For parallel drafting, the draft model config must have "
                "`pard_token`, `ptd_token_id`, or "
                "`dflash_config.mask_token_id` specified in its config.json."
            )

        if self.pass_hidden_states_to_model:
            self.parallel_drafting_hidden_state_tensor = torch.empty(
                self.hidden_size, dtype=self.dtype, device=self.device
            )

    def _get_positions(self, num_tokens: int):
        if self.uses_mrope:
            return self.mrope_positions[:, :num_tokens]
        if self.uses_xdrope_dim > 0 and self.draft_uses_xdrope_dim > 0:
            return self.xdrope_positions[:, :num_tokens]
        return self.positions[:num_tokens]

    def _set_positions(self, num_tokens: int, positions: torch.Tensor):
        if self.uses_mrope:
            self.mrope_positions[:, :num_tokens] = positions
        elif self.uses_xdrope_dim > 0 and self.draft_uses_xdrope_dim > 0:
            self.xdrope_positions[:, :num_tokens] = positions
        else:
            # Convert M-RoPE positions if target model uses M-RoPE
            # but draft doesn't, For text inputs, all M-RoPE
            # dimensions are identical
            if self.vllm_config.model_config.uses_mrope:
                positions = positions[0]
            self.positions[:num_tokens] = positions

    def _get_slot_mapping(
        self,
        num_tokens: int,
        slot_mapping: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return slot_mapping dict for EAGLE layers.

        If slot_mapping is provided, copies it into the buffer first.
        """
        if slot_mapping is not None:
            num_actual = slot_mapping.shape[0]
            self._slot_mapping_buffer[:num_actual].copy_(slot_mapping)
            if num_tokens > num_actual:
                self._slot_mapping_buffer[num_actual:num_tokens].fill_(PADDING_SLOT_ID)

        view = self._slot_mapping_buffer[:num_tokens]
        return {name: view for name in self._draft_attn_layer_names}

    def initialize_cudagraph_keys(self, cudagraph_mode: CUDAGraphMode) -> None:
        """Initialize cudagraph dispatcher keys for eagle.

        Eagle only supports PIECEWISE cudagraphs (via mixed_mode) by default.
        This should be called after adjust_cudagraph_sizes_for_spec_decode.
        """
        mtp_draft_full_cudagraph = (
            self.method == "mtp"
            and os.environ.get("VLLM_OSCAR_MTP_DRAFT_FULL_CUDAGRAPH", "0") == "1"
        )
        if (
            not self.speculative_config.enforce_eager
            and cudagraph_mode.mixed_mode()
            in [CUDAGraphMode.PIECEWISE, CUDAGraphMode.FULL]
        ):
            eagle_cudagraph_mode = (
                cudagraph_mode if mtp_draft_full_cudagraph else CUDAGraphMode.PIECEWISE
            )
        else:
            eagle_cudagraph_mode = CUDAGraphMode.NONE

        if mtp_draft_full_cudagraph and eagle_cudagraph_mode:
            self.cudagraph_dispatcher.uniform_decode_query_len = 1
            target_uniform_decode_query_len = 1 + self.num_speculative_tokens
            capture_sizes = self.compilation_config.cudagraph_capture_sizes
            assert capture_sizes is not None
            draft_decode_capture_sizes = {
                size // target_uniform_decode_query_len
                for size in capture_sizes
                if size % target_uniform_decode_query_len == 0
                and size // target_uniform_decode_query_len <= self.max_batch_size
            }
            if (
                os.environ.get("VLLM_OSCAR_MTP_DRAFT_MARGIN_EARLY_STOP")
                or os.environ.get("VLLM_OSCAR_MTP_ACCEPTANCE_ADAPTIVE")
                or os.environ.get("VLLM_OSCAR_MTP_PERIODIC_DRAFT")
            ):
                draft_decode_capture_sizes.update(
                    range(1, self.num_speculative_tokens + 2)
                )
            draft_capture_sizes = sorted(
                set(capture_sizes) | draft_decode_capture_sizes
            )
            self._oscar_mtp_draft_decode_capture_sizes = tuple(
                size for size in draft_capture_sizes if size <= self.max_batch_size
            )
            self.cudagraph_dispatcher.initialize_cudagraph_keys(
                eagle_cudagraph_mode,
                uniform_decode_query_len=1,
                cudagraph_capture_sizes=draft_capture_sizes,
            )
        else:
            self.cudagraph_dispatcher.initialize_cudagraph_keys(eagle_cudagraph_mode)

    def _greedy_sample(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Greedy-sample draft tokens from hidden states."""
        self._oscar_mtp_stop_after_current_draft = False
        if self.use_local_argmax_reduction:
            return self.model.get_top_tokens(hidden_states)
        logits = self.model.compute_logits(hidden_states)
        sampled_ids = logits.argmax(dim=-1)
        threshold = self._oscar_mtp_draft_margin_threshold
        if (
            self._oscar_mtp_dynamic_draft_enabled
            and threshold is not None
            and self.method == "mtp"
            and self.num_speculative_tokens == 5
            and logits.ndim == 2
            and logits.shape[0] == 1
            and logits.shape[1] >= 2
        ):
            top_values = torch.topk(logits, 2, dim=-1).values
            margin = top_values[:, 0] - top_values[:, 1]
            self._oscar_mtp_stop_after_current_draft = bool(
                torch.all(margin < threshold).item()
            )
        return sampled_ids

    def _sample_from_logits(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if not self._enable_probabilistic_draft_probs:
            return logits.argmax(dim=-1), None
        if sampling_metadata.all_greedy:
            return logits.argmax(dim=-1), None

        # Parallel drafting has K request-major rows per request while the
        # sampling metadata is per request. MTP is sequential, but preserving
        # this upstream behavior keeps the helper correct for both layouts.
        temperature = sampling_metadata.temperature
        if temperature is not None and temperature.shape[0] != logits.shape[0]:
            assert logits.shape[0] % temperature.shape[0] == 0
            factor = logits.shape[0] // temperature.shape[0]
            sampling_metadata = dataclasses.replace(
                sampling_metadata,
                temperature=temperature.repeat_interleave(factor, dim=0),
            )

        return compute_probs_and_sample_next_token(logits, sampling_metadata)

    def _sample_draft_tokens(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if (
            not self._enable_probabilistic_draft_probs
            or sampling_metadata.all_greedy
        ):
            return self._greedy_sample(hidden_states), None
        logits = self.model.compute_logits(hidden_states)
        return self._sample_from_logits(logits, sampling_metadata)

    def take_last_draft_probs(self) -> torch.Tensor | None:
        """Transfer ownership of probabilities from the latest proposal."""
        return self._last_draft_probs

    def propose(
        self,
        # [num_tokens]
        target_token_ids: torch.Tensor,
        # [num_tokens] or [3, num_tokens] when M-RoPE is enabled
        target_positions: torch.Tensor,
        # [num_tokens, hidden_size]
        target_hidden_states: torch.Tensor,
        # [batch_size]
        next_token_ids: torch.Tensor,
        token_indices_to_sample: torch.Tensor | None,
        common_attn_metadata: CommonAttentionMetadata,
        sampling_metadata: SamplingMetadata,
        mm_embed_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        num_rejected_tokens_gpu: torch.Tensor | None = None,
        slot_mappings: dict[str, torch.Tensor]
        | list[dict[str, torch.Tensor]]
        | None = None,
        draft_token_limit: int | None = None,
    ) -> torch.Tensor:
        self._last_draft_probs = None
        batch_size = common_attn_metadata.batch_size()
        num_speculative_tokens = self.num_speculative_tokens
        if draft_token_limit is not None:
            if (
                self.method != "mtp"
                or batch_size != 1
                or self.parallel_drafting
                or not 1 <= draft_token_limit <= self.num_speculative_tokens
            ):
                raise ValueError(
                    "draft_token_limit requires batch-1 sequential MTP and must "
                    "not exceed the configured speculative token count"
                )
            num_speculative_tokens = draft_token_limit
        if (
            not self._oscar_mtp_dynamic_draft_enabled
            and self._oscar_mtp_draft_margin_threshold is not None
            and self._oscar_mtp_draft_margin_sentinel is not None
            and self._oscar_mtp_draft_margin_sentinel.is_file()
        ):
            self._oscar_mtp_dynamic_draft_enabled = True
            logger.info(
                "OSCAR MTP5 dynamic draft early-stop enabled at margin %.6f",
                self._oscar_mtp_draft_margin_threshold,
            )
        mtp_propose_debug = (
            self.method == "mtp"
            and os.environ.get("VLLM_MTP_SAMPLE_TIMING_LOGS", "0") == "1"
        )

        def _tensor_shape(value: Any) -> tuple[int, ...] | None:
            return tuple(value.shape) if torch.is_tensor(value) else None

        def log_mtp_proposer_event(
            phase: str,
            start: float | None = None,
            **extra: Any,
        ) -> None:
            if not mtp_propose_debug:
                return
            pp = get_pp_group()
            elapsed_ms = None
            if start is not None:
                elapsed_ms = (time.perf_counter() - start) * 1000.0
            logger.warning(
                "MTP proposer event: phase=%s pp_rank=%s pp_world=%s "
                "is_last=%s elapsed_ms=%s batch_size=%s max_seq_len=%s "
                "num_actual_tokens=%s max_query_len=%s target_token_shape=%s "
                "target_position_shape=%s target_hidden_shape=%s "
                "next_token_shape=%s token_indices_shape=%s extra=%s",
                phase,
                pp.rank,
                pp.world_size,
                pp.is_last_rank,
                None if elapsed_ms is None else round(elapsed_ms, 3),
                batch_size,
                common_attn_metadata.max_seq_len,
                common_attn_metadata.num_actual_tokens,
                common_attn_metadata.max_query_len,
                _tensor_shape(target_token_ids),
                _tensor_shape(target_positions),
                _tensor_shape(target_hidden_states),
                _tensor_shape(next_token_ids),
                _tensor_shape(token_indices_to_sample),
                extra,
            )

        log_mtp_proposer_event(
            "enter",
            max_model_len=self.max_model_len,
            max_num_tokens=self.max_num_tokens,
            num_speculative_tokens=self.num_speculative_tokens,
            needs_extra_input_slots=self.needs_extra_input_slots,
            parallel_drafting=self.parallel_drafting,
        )

        if self.method in ("eagle3", "dflash"):
            assert isinstance(
                self.model,
                (
                    Eagle3LlamaForCausalLM,
                    Eagle3DeepseekV2ForCausalLM,
                    DFlashQwen3ForCausalLM,
                ),
            )
            target_hidden_states = self.model.combine_hidden_states(
                target_hidden_states
            )
            assert target_hidden_states.shape[-1] == self.hidden_size
            log_mtp_proposer_event("combine_hidden_states_done")

        phase_start = time.perf_counter()
        num_tokens, token_indices_to_sample, common_attn_metadata = (
            self.set_inputs_first_pass(
                target_token_ids=target_token_ids,
                next_token_ids=next_token_ids,
                target_positions=target_positions,
                target_hidden_states=target_hidden_states,
                token_indices_to_sample=token_indices_to_sample,
                cad=common_attn_metadata,
                num_rejected_tokens_gpu=num_rejected_tokens_gpu,
            )
        )
        log_mtp_proposer_event(
            "set_inputs_first_pass_done",
            phase_start,
            num_tokens=num_tokens,
            token_indices_shape=_tensor_shape(token_indices_to_sample),
            updated_max_seq_len=common_attn_metadata.max_seq_len,
            updated_num_actual_tokens=common_attn_metadata.num_actual_tokens,
        )

        phase_start = time.perf_counter()
        per_group_attn_metadata, per_layer_attn_metadata = (
            self.build_per_group_and_layer_attn_metadata(common_attn_metadata)
        )
        log_mtp_proposer_event(
            "build_attn_metadata_done",
            phase_start,
            per_group_types=[type(md).__name__ for md in per_group_attn_metadata],
            per_layer_count=len(per_layer_attn_metadata),
        )

        phase_start = time.perf_counter()
        cudagraph_runtime_mode, num_input_tokens, num_tokens_across_dp = (
            self._determine_batch_execution_and_padding(num_tokens)
        )
        log_mtp_proposer_event(
            "determine_batch_execution_done",
            phase_start,
            cudagraph_runtime_mode=str(cudagraph_runtime_mode),
            num_input_tokens=num_input_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
        )

        phase_start = time.perf_counter()
        model_kwargs, slot_mapping_size = self.build_model_inputs_first_pass(
            num_tokens, num_input_tokens, mm_embed_inputs
        )
        log_mtp_proposer_event(
            "build_model_inputs_first_pass_done",
            phase_start,
            slot_mapping_size=slot_mapping_size,
            input_ids_shape=_tensor_shape(model_kwargs.get("input_ids")),
            positions_shape=_tensor_shape(model_kwargs.get("positions")),
            inputs_embeds_shape=_tensor_shape(model_kwargs.get("inputs_embeds")),
            hidden_states_shape=_tensor_shape(model_kwargs.get("hidden_states")),
        )

        phase_start = time.perf_counter()
        log_mtp_proposer_event(
            "model_forward_first_pass_enter",
            num_input_tokens=num_input_tokens,
            slot_mapping_size=slot_mapping_size,
        )
        if self._share_mtp_indices and hasattr(self.model.model, "set_skip_topk"):
            self.model.model.set_skip_topk(False)
        with set_forward_context(
            per_layer_attn_metadata,
            self.vllm_config,
            num_tokens=num_input_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            slot_mapping=self._get_slot_mapping(
                slot_mapping_size, common_attn_metadata.slot_mapping
            ),
        ):
            ret_hidden_states = self.model(**model_kwargs)
            if not self.model_returns_tuple():
                last_hidden_states = ret_hidden_states
                hidden_states = last_hidden_states
            else:
                last_hidden_states, hidden_states = ret_hidden_states
        if self._share_mtp_indices and hasattr(self.model.model, "set_skip_topk"):
            # Compact sparse-index rows before later MTP steps reuse them.
            self.model.model.set_skip_topk(True)
            if hasattr(self.model.model, "compact_topk_indices"):
                self.model.model.compact_topk_indices(
                    token_indices_to_sample
                )
        log_mtp_proposer_event(
            "model_forward_first_pass_done",
            phase_start,
            last_hidden_shape=_tensor_shape(last_hidden_states),
            hidden_shape=_tensor_shape(hidden_states),
        )

        phase_start = time.perf_counter()
        sample_hidden_states = last_hidden_states[token_indices_to_sample]
        log_mtp_proposer_event(
            "select_sample_hidden_states_done",
            phase_start,
            sample_hidden_shape=_tensor_shape(sample_hidden_states),
        )

        # Early exit if there is only one draft token to be generated.
        if num_speculative_tokens == 1 or self.parallel_drafting:
            phase_start = time.perf_counter()
            log_mtp_proposer_event("draft_sample_enter")
            draft_token_ids, draft_probs = self._sample_draft_tokens(
                sample_hidden_states, sampling_metadata
            )
            log_mtp_proposer_event(
                "draft_sample_done",
                phase_start,
                draft_token_shape=_tensor_shape(draft_token_ids),
            )
            if draft_probs is not None:
                self._last_draft_probs = draft_probs.view(
                    batch_size, num_speculative_tokens, -1
                ).contiguous()
            return draft_token_ids.view(-1, num_speculative_tokens)

        if self.uses_mrope:
            positions = self.mrope_positions[:, token_indices_to_sample]
        else:
            positions = self.positions[token_indices_to_sample]
        hidden_states = hidden_states[token_indices_to_sample]

        if any(isinstance(md, TreeAttentionMetadata) for md in per_group_attn_metadata):
            # Draft using tree attention - requires full logits for top-k
            logits = self.model.compute_logits(sample_hidden_states)
            draft_token_ids_list = self.propose_tree(
                batch_size=batch_size,
                logits=logits,
                positions=positions,
                hidden_states=hidden_states,
                common_attn_metadata=common_attn_metadata,
                slot_mappings=slot_mappings,
            )
            # [batch_size, num_tree_tokens]
            return torch.cat(draft_token_ids_list, dim=1)

        draft_token_ids, draft_probs = self._sample_draft_tokens(
            sample_hidden_states, sampling_metadata
        )
        draft_probs_list = None if draft_probs is None else [draft_probs]

        if self.allowed_attn_types is not None:
            for group_md in per_group_attn_metadata:
                if not isinstance(group_md, self.allowed_attn_types):
                    raise ValueError(
                        f"Unsupported attention metadata type for speculative "
                        "decoding with num_speculative_tokens > 1: "
                        f"{type(group_md)}. Supported types are: "
                        f"{self.allowed_attn_types}"
                    )

        # Generate the remaining draft tokens.
        draft_token_ids_list = [draft_token_ids]

        mtp_draft_full_cudagraph = (
            self.method == "mtp"
            and os.environ.get("VLLM_OSCAR_MTP_DRAFT_FULL_CUDAGRAPH", "0") == "1"
        )
        cudagraph_runtime_mode, input_batch_size, batch_size_across_dp = (
            self._determine_batch_execution_and_padding(
                batch_size,
                uniform_decode=mtp_draft_full_cudagraph,
            )
        )

        common_attn_metadata.num_actual_tokens = batch_size
        common_attn_metadata.max_query_len = 1
        common_attn_metadata.query_start_loc = self.arange[: batch_size + 1]
        common_attn_metadata.query_start_loc_cpu = torch.from_numpy(
            self.token_arange_np[: batch_size + 1]
        ).clone()

        # In padded drafter batch, we need to adjust the sequence lengths
        # to remove the "padding" (i.e. rejected tokens).
        # Only apply this adjustment when we have rejected tokens
        # (i.e., not the first proposal).
        if self.num_speculative_tokens > 1 and num_rejected_tokens_gpu is not None:
            common_attn_metadata.seq_lens -= num_rejected_tokens_gpu
            # Invalidate the CPU-side shadows to avoid H<>D sync.
            common_attn_metadata._seq_lens_cpu = None
            common_attn_metadata._num_computed_tokens_cpu = None

        if (
            mtp_draft_full_cudagraph
            and os.environ.get("VLLM_OSCAR_MTP_DRAFT_INT2", "0") == "1"
            and cudagraph_runtime_mode == CUDAGraphMode.FULL
        ):
            self._bind_oscar_mtp_draft_graph_metadata(
                common_attn_metadata,
                input_batch_size,
            )

        block_size = self.block_size
        assert block_size > 0, "block_size has not been initialized."
        for token_index in range(num_speculative_tokens - 1):
            if self._oscar_mtp_stop_after_current_draft:
                break
            # Update the inputs.
            # cast to int32 is crucial when eagle model is compiled.
            # tensor.argmax() returns int64 by default.
            input_ids = draft_token_ids_list[-1].int()
            # Use fused kernel for slot mapping and metadata updates.
            # Write clamped positions directly into the positions buffer to
            # avoid an extra D2D copy for the common (non-mrope) case.
            positions_1d = positions[0] if self.uses_mrope else positions
            if self.uses_mrope:
                out_pos = self.mrope_positions[0, :batch_size]
            elif self.uses_xdrope_dim > 0 and self.draft_uses_xdrope_dim > 0:
                out_pos = self.xdrope_positions[0, :batch_size]
            else:
                out_pos = self.positions[:batch_size]
            eagle_step_update_slot_mapping_and_metadata(
                positions_1d=positions_1d,
                block_table_tensor=common_attn_metadata.block_table_tensor,
                seq_lens=common_attn_metadata.seq_lens,
                block_size=block_size,
                max_model_len=self.max_model_len,
                out_clamped_positions=out_pos,
                out_slot_mapping=self._slot_mapping_buffer[:input_batch_size],
                input_batch_size=input_batch_size,
            )
            common_attn_metadata.slot_mapping = self._slot_mapping_buffer[:batch_size]
            if self.uses_mrope:
                self.mrope_positions[1:, :batch_size] = self.mrope_positions[
                    0, :batch_size
                ]
                positions = self.mrope_positions[:, :batch_size]
            elif self.uses_xdrope_dim > 0 and self.draft_uses_xdrope_dim > 0:
                self.xdrope_positions[1:, :batch_size] = self.xdrope_positions[
                    0, :batch_size
                ]
                positions = self.xdrope_positions[0, :batch_size]
            else:
                positions = self.positions[:batch_size]
            # Increment the maximum sequence length. We increment max_seq_len
            # unconditionally even though some seq_lens may have been capped above,
            # as max_seq_len serves as an upper bound for sequence lengths.
            common_attn_metadata.max_seq_len = min(
                common_attn_metadata.max_seq_len + 1, self.max_model_len
            )

            # Also update the CPU-side shadow; NOTE: this is hacky and should be
            # removed in when common_attn_metadata.seq_lens_cpu is deprecated.
            if common_attn_metadata._seq_lens_cpu is not None:
                common_attn_metadata._seq_lens_cpu += 1
            if common_attn_metadata._num_computed_tokens_cpu is not None:
                common_attn_metadata._num_computed_tokens_cpu += 1
            if common_attn_metadata.seq_lens_cpu_upper_bound is not None:
                common_attn_metadata.seq_lens_cpu_upper_bound += 1

            # Rebuild attention metadata
            _, per_layer_attn_metadata = self.build_per_group_and_layer_attn_metadata(
                common_attn_metadata, draft_index=token_index + 1
            )

            # copy inputs to buffer for cudagraph
            self.input_ids[:batch_size] = input_ids
            self.hidden_states[:batch_size] = hidden_states
            if self.supports_mm_inputs:
                self.inputs_embeds[:batch_size] = self.model.embed_input_ids(input_ids)

                input_ids = None
                inputs_embeds = self.inputs_embeds[:input_batch_size]
            else:
                input_ids = self.input_ids[:input_batch_size]
                inputs_embeds = None

            # Run the model.
            model_kwargs = {
                "input_ids": input_ids,
                "positions": self._get_positions(input_batch_size),
                "inputs_embeds": inputs_embeds,
            }
            if self.pass_hidden_states_to_model:
                model_kwargs["hidden_states"] = self.hidden_states[:input_batch_size]

            with set_forward_context(
                per_layer_attn_metadata,
                self.vllm_config,
                num_tokens=input_batch_size,
                num_tokens_across_dp=batch_size_across_dp,
                cudagraph_runtime_mode=cudagraph_runtime_mode,
                slot_mapping=self._get_slot_mapping(input_batch_size),
            ):
                ret_hidden_states = self.model(**model_kwargs)
                if not self.model_returns_tuple():
                    last_hidden_states = ret_hidden_states
                    hidden_states = ret_hidden_states
                else:
                    last_hidden_states, hidden_states = ret_hidden_states

            hidden_states = hidden_states[:batch_size]
            draft_token_ids, draft_probs = self._sample_draft_tokens(
                last_hidden_states[:batch_size], sampling_metadata
            )
            if draft_probs is not None:
                assert draft_probs_list is not None
                draft_probs_list.append(draft_probs)
            draft_token_ids_list.append(draft_token_ids)

        # [batch_size, num_speculative_tokens]
        draft_token_ids = torch.stack(draft_token_ids_list, dim=1)
        if draft_probs_list is not None:
            self._last_draft_probs = torch.stack(
                draft_probs_list, dim=1
            ).contiguous()
        return draft_token_ids

    def set_inputs_first_pass(
        self,
        target_token_ids: torch.Tensor,
        next_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        token_indices_to_sample: torch.Tensor | None,
        cad: CommonAttentionMetadata,
        num_rejected_tokens_gpu: torch.Tensor | None,
    ) -> tuple[int, torch.Tensor, CommonAttentionMetadata]:
        if not self.needs_extra_input_slots:
            # Default EAGLE pathway: no reshaping of input tensors needed.
            # Simply rotate the input ids and leave the positions unchanged,
            # Inserting the next token ids at the last slot in each request.
            if token_indices_to_sample is None:
                token_indices_to_sample = cad.query_start_loc[1:] - 1

            num_tokens = target_token_ids.shape[0]
            # Shift the input ids by one token.
            # E.g., [a1, b1, b2, c1, c2, c3] -> [b1, b2, c1, c2, c3, c3]
            self.input_ids[: num_tokens - 1] = target_token_ids[1:]
            # Replace the last token with the next token.
            # E.g., [b1, b2, c1, c2, c3, c3] -> [a2, b2, b3, c2, c3, c4]
            self.input_ids[token_indices_to_sample] = next_token_ids

            # copy inputs to buffer for cudagraph
            if self.uses_xdrope_dim > 0 and self.draft_uses_xdrope_dim == 0:
                target_positions = target_positions[0]
            self._set_positions(num_tokens, target_positions)

            self.hidden_states[:num_tokens] = target_hidden_states

            return num_tokens, token_indices_to_sample, cad
        else:
            assert self.is_rejected_token_mask is not None
            assert self.is_masked_token_mask is not None
            # 1.
            # Call a custom triton kernel to copy input_ids and positions
            # into the correct slots in the preallocated buffers self.input_ids,
            # self.positions.
            batch_size = cad.batch_size()
            # Since we might have to copy a lot of data for prefills, we select the
            # block size based on the max query length and limit to max 256 slots/block.
            max_num_tokens_per_request = (
                cad.max_query_len + self.net_num_new_slots_per_request
            )
            BLOCK_SIZE_TOKENS = min(256, next_power_of_2(max_num_tokens_per_request))
            num_blocks = (
                max_num_tokens_per_request + BLOCK_SIZE_TOKENS - 1
            ) // BLOCK_SIZE_TOKENS
            total_num_input_tokens = target_token_ids.shape[0]
            total_num_output_tokens = total_num_input_tokens + (
                self.net_num_new_slots_per_request * batch_size
            )

            token_indices_to_sample = torch.empty(
                batch_size * self.extra_slots_per_request,
                dtype=torch.int32,
                device=self.device,
            )

            # Destination indices to write target_hidden_states into drafting buffer.
            out_hidden_state_mapping = torch.empty(
                total_num_input_tokens, dtype=torch.int32, device=self.device
            )

            # Kernel grid: one program per request (row)
            grid = (batch_size, num_blocks)
            query_start_loc = cad.query_start_loc
            query_end_loc = cad.query_start_loc[1:] - 1
            if num_rejected_tokens_gpu is not None:
                query_end_loc = query_end_loc - num_rejected_tokens_gpu

            copy_and_expand_eagle_inputs_kernel[grid](
                # (Padded) Inputs from the target model
                target_token_ids_ptr=target_token_ids,
                target_positions_ptr=target_positions,
                next_token_ids_ptr=next_token_ids,  # sampled tokens, one per request
                # Outputs to the drafting buffers
                out_input_ids_ptr=self.input_ids,
                out_positions_ptr=self.positions,  # Doesn't support mrope for now
                out_is_rejected_token_mask_ptr=self.is_rejected_token_mask,
                out_is_masked_token_mask_ptr=self.is_masked_token_mask,
                out_new_token_indices_ptr=token_indices_to_sample,
                out_hidden_state_mapping_ptr=out_hidden_state_mapping,
                # Input metadata
                query_start_loc_ptr=query_start_loc,
                query_end_loc_ptr=query_end_loc,
                padding_token_id=0,
                parallel_drafting_token_id=self.parallel_drafting_token_id,
                # Sizing info
                # Note that we can deduce batch_size for free from the grid size
                total_input_tokens=total_num_input_tokens,
                num_padding_slots_per_request=self.extra_slots_per_request,
                shift_input_ids=self.pass_hidden_states_to_model,
                BLOCK_SIZE_TOKENS=BLOCK_SIZE_TOKENS,
            )
            if self.pass_hidden_states_to_model:
                assert self.parallel_drafting_hidden_state_tensor is not None
                self.hidden_states[out_hidden_state_mapping] = target_hidden_states
                # Use torch.where to avoid DtoH sync from boolean indexing
                mask = self.is_masked_token_mask[:total_num_output_tokens]
                torch.where(
                    mask.unsqueeze(1),
                    self.parallel_drafting_hidden_state_tensor,
                    self.hidden_states[:total_num_output_tokens],
                    out=self.hidden_states[:total_num_output_tokens],
                )

            # 2.
            # Recompute the slot mapping based on the new positions and
            # rejection mask.
            assert self.block_size > 0, "block_size has not been initialized."
            new_slot_mapping = compute_new_slot_mapping(
                cad=cad,
                new_positions=self.positions[:total_num_output_tokens],
                is_rejected_token_mask=self.is_rejected_token_mask[
                    :total_num_output_tokens
                ],
                block_size=self.block_size,
                num_new_tokens=self.net_num_new_slots_per_request,
                max_model_len=self.max_model_len,
            )

            # 3. Update the common attention metadata with the new (meta)data
            new_cad = extend_all_queries_by_N(
                cad,
                N=self.net_num_new_slots_per_request,
                arange=self.arange,
                new_slot_mapping=new_slot_mapping,
            )

            return total_num_output_tokens, token_indices_to_sample, new_cad

    def build_model_inputs_first_pass(
        self,
        num_tokens: int,
        num_input_tokens: int,
        mm_embed_inputs: tuple[list[torch.Tensor], torch.Tensor] | None,
    ) -> tuple[dict[str, Any], int]:
        if self.supports_mm_inputs:
            mm_embeds, is_mm_embed = mm_embed_inputs or (None, None)

            self.inputs_embeds[:num_tokens] = self.model.embed_input_ids(
                self.input_ids[:num_tokens],
                multimodal_embeddings=mm_embeds,
                is_multimodal=is_mm_embed,
            )

            input_ids = None
            inputs_embeds = self.inputs_embeds[:num_input_tokens]
        else:
            input_ids = self.input_ids[:num_input_tokens]
            inputs_embeds = None

        model_kwargs = {
            "input_ids": input_ids,
            "positions": self._get_positions(num_input_tokens),
            "inputs_embeds": inputs_embeds,
        }
        if self.pass_hidden_states_to_model:
            model_kwargs["hidden_states"] = self.hidden_states[:num_input_tokens]

        return model_kwargs, num_input_tokens

    def build_per_group_and_layer_attn_metadata(
        self, common_attn_metadata: CommonAttentionMetadata, draft_index: int = 0
    ) -> tuple[list[object], dict[str, object]]:
        per_group_attn_metadata: list[object] = []
        per_layer_attn_metadata: dict[str, object] = {}
        for attn_group in self.draft_attn_groups:
            attn_metadata = attn_group.get_metadata_builder().build_for_drafting(
                common_attn_metadata=common_attn_metadata, draft_index=draft_index
            )
            per_group_attn_metadata.append(attn_metadata)
            for layer_name in attn_group.layer_names:
                per_layer_attn_metadata[layer_name] = attn_metadata
        return per_group_attn_metadata, per_layer_attn_metadata

    def _remember_oscar_mtp_draft_graph_metadata(
        self,
        input_batch_size: int,
        per_layer_attn_metadata: dict[str, object],
    ) -> None:
        for metadata in per_layer_attn_metadata.values():
            oscar_mla = getattr(metadata, "oscar_mla", None)
            if oscar_mla is not None:
                capture_metadata = getattr(
                    self,
                    "_oscar_mtp_draft_graph_capture_metadata",
                    None,
                )
                if capture_metadata is None:
                    capture_metadata = {}
                    self._oscar_mtp_draft_graph_capture_metadata = capture_metadata
                capture_metadata[input_batch_size] = oscar_mla
                return

    def _bind_oscar_mtp_draft_graph_metadata(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        input_batch_size: int,
    ) -> None:
        runtime_oscar = common_attn_metadata.oscar_mla
        if runtime_oscar is None:
            return
        captured_oscar = self._oscar_mtp_draft_graph_capture_metadata.get(
            input_batch_size
        )
        if captured_oscar is None:
            raise RuntimeError(
                "OSCAR MTP draft FULL cudagraph runtime has no capture-bound "
                f"metadata for batch size {input_batch_size}"
            )
        if captured_oscar is runtime_oscar:
            return

        def _copy_vector(name: str, padding_value: int) -> None:
            source = getattr(runtime_oscar, name)
            target = getattr(captured_oscar, name)
            if source.ndim != 1 or target.ndim != 1:
                raise RuntimeError(f"OSCAR MTP {name} must be one-dimensional")
            if source.numel() > target.numel():
                raise RuntimeError(f"OSCAR MTP runtime {name} exceeds capture capacity")
            if source.dtype != target.dtype or source.device != target.device:
                raise RuntimeError(
                    f"OSCAR MTP runtime {name} does not match capture storage"
                )
            if source.data_ptr() == target.data_ptr():
                if source.shape == target.shape:
                    return
                source = source.clone()
            target.fill_(padding_value)
            target[: source.numel()].copy_(source)

        _copy_vector("hp_rows", -1)
        _copy_vector("decode_positions", -1)
        _copy_vector("final_seq_lens", 0)
        _copy_vector("previous_seq_lens", 0)

        source_history = runtime_oscar.history_page_table
        target_history = captured_oscar.history_page_table
        if source_history.ndim != 2 or target_history.ndim != 2:
            raise RuntimeError("OSCAR MTP history page tables must be two-dimensional")
        if any(
            source_size > target_size
            for source_size, target_size in zip(
                source_history.shape,
                target_history.shape,
            )
        ):
            raise RuntimeError(
                "OSCAR MTP runtime history page table exceeds capture capacity"
            )
        if (
            source_history.dtype != target_history.dtype
            or source_history.device != target_history.device
        ):
            raise RuntimeError(
                "OSCAR MTP runtime history page table does not match capture storage"
            )
        if source_history.data_ptr() != target_history.data_ptr():
            target_history.zero_()
            target_history[
                : source_history.shape[0],
                : source_history.shape[1],
            ].copy_(source_history)
        elif source_history.shape != target_history.shape:
            source_history = source_history.clone()
            target_history.zero_()
            target_history[
                : source_history.shape[0],
                : source_history.shape[1],
            ].copy_(source_history)

        common_attn_metadata.oscar_mla = captured_oscar

    def model_returns_tuple(self) -> bool:
        if self.method == "mtp":
            # GLM-5.2's speculative setup rewrites the draft architecture to
            # DeepSeekMTPModel.  That model returns (pre_norm, post_norm),
            # while other MTP variants retain their legacy single-tensor ABI.
            return "DeepSeekMTPModel" in (
                self.draft_model_config.hf_config.architectures or []
            )
        return self.method not in ("mtp", "draft_model", "dflash")

    def prepare_next_token_ids_cpu(
        self,
        sampled_token_ids: list[list[int]],
        requests: dict[str, CachedRequestState],
        gpu_input_batch: InputBatch,
        num_scheduled_tokens: dict[str, int],
    ) -> torch.Tensor:
        """
        This function is used to prepare the inputs for speculative decoding.
        It calculates the next token ids for each request based on the sampled
        token ids from the CPU. If a request has no sampled token ids (e.g.,
        during the initial decoding steps), it falls back to using the request
        state to get the next token id.
        """
        req_ids = gpu_input_batch.req_ids
        next_token_ids: list[int] = []
        for i, token_ids in enumerate(sampled_token_ids):
            if token_ids:
                # Common case.
                next_token_id = token_ids[-1]
            else:
                # Partial prefill (rare case).
                # Get the next token id from the request state.
                req_id = req_ids[i]
                req_state = requests[req_id]
                seq_len = req_state.num_computed_tokens + num_scheduled_tokens[req_id]
                next_token_id = req_state.get_token_id(seq_len)
            next_token_ids.append(next_token_id)
        next_token_ids = torch.tensor(
            next_token_ids, dtype=torch.int32, device=self.input_ids.device
        )
        return next_token_ids

    def prepare_next_token_ids_padded(
        self,
        sampled_token_ids: torch.Tensor,
        requests: dict[str, CachedRequestState],
        gpu_input_batch: InputBatch,
        discard_request_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        This function is used to prepare the inputs for speculative decoding.
        It calculates the next token ids and the number of valid sampled tokens
        for each request, considering the "discarded" requests whose next token
        is not sampled and comes from `request.get_token_id()` instead. This is denoted
        the "backup" token id. It also counts rejected tokens via `sampled_token_ids`.
        """
        # Precompute get_token_id for when there is no valid next token
        num_reqs = gpu_input_batch.num_reqs
        seq_lens_list = (gpu_input_batch.num_tokens_no_spec[:num_reqs] - 1).tolist()
        self.backup_next_token_ids.np[:num_reqs] = np.array(
            [
                requests[gpu_input_batch.req_ids[i]].get_token_id(seq_lens_list[i])
                for i in range(num_reqs)
            ],
            dtype=np.int32,
        )
        self.backup_next_token_ids.copy_to_gpu(num_reqs)
        backup_tokens_gpu = self.backup_next_token_ids.gpu

        batch_size, num_tokens = sampled_token_ids.shape
        device = sampled_token_ids.device

        assert discard_request_mask.dtype == torch.bool
        assert backup_tokens_gpu.dtype == torch.int32

        next_token_ids = torch.empty(batch_size, dtype=torch.int32, device=device)
        valid_sampled_tokens_count = next_token_ids.new_empty(batch_size)

        # Kernel grid: one program per request (row)
        grid = (batch_size,)

        # Find the next power of 2 for block sizes
        BLOCK_SIZE_TOKENS = next_power_of_2(num_tokens)
        eagle_prepare_next_token_padded_kernel[grid](
            sampled_token_ids,
            discard_request_mask,
            backup_tokens_gpu,
            next_token_ids,
            valid_sampled_tokens_count,
            gpu_input_batch.vocab_size,
            num_tokens,
            batch_size,
            sampled_token_ids.stride(0),
            BLOCK_SIZE_TOKENS=BLOCK_SIZE_TOKENS,
        )

        return next_token_ids, valid_sampled_tokens_count

    def prepare_inputs_padded(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        spec_decode_metadata: SpecDecodeMetadata,
        valid_sampled_tokens_count: torch.Tensor,
    ) -> tuple[CommonAttentionMetadata, torch.Tensor, torch.Tensor]:
        """
        This function is used to prepare the inputs for speculative decoding
        It updates the common_attn_metadata for speculative decoding,
        but does not consider the rejected tokens. Instead, all tokens
        are included as inputs to the speculator, with the rejected tokens
        used as padding and filtered out later by `token_indices_to_sample`.
        No blocking CPU operations should be introduced in this function.
        """
        num_reqs = common_attn_metadata.num_reqs
        device = valid_sampled_tokens_count.device

        token_indices_to_sample = torch.empty(
            (num_reqs,), dtype=torch.int32, device=device
        )
        num_rejected_tokens_gpu = torch.empty(
            (num_reqs,), dtype=torch.int32, device=device
        )

        grid = (num_reqs,)
        eagle_prepare_inputs_padded_kernel[grid](
            spec_decode_metadata.cu_num_draft_tokens,
            valid_sampled_tokens_count,
            common_attn_metadata.query_start_loc,
            token_indices_to_sample,
            num_rejected_tokens_gpu,
            num_reqs,
        )

        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
        new_query_len_per_req = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]

        total_num_tokens = query_start_loc_cpu[-1].item()

        spec_common_attn_metadata = CommonAttentionMetadata(
            query_start_loc=common_attn_metadata.query_start_loc,
            seq_lens=common_attn_metadata.seq_lens,
            query_start_loc_cpu=query_start_loc_cpu,
            _seq_lens_cpu=common_attn_metadata._seq_lens_cpu,
            _num_computed_tokens_cpu=common_attn_metadata._num_computed_tokens_cpu,
            seq_lens_cpu_upper_bound=common_attn_metadata.seq_lens_cpu_upper_bound,
            num_reqs=common_attn_metadata.num_reqs,
            num_actual_tokens=total_num_tokens,
            max_query_len=new_query_len_per_req.max().item(),
            max_seq_len=common_attn_metadata.max_seq_len,
            block_table_tensor=common_attn_metadata.block_table_tensor,
            slot_mapping=common_attn_metadata.slot_mapping[:total_num_tokens],
            oscar_mla=common_attn_metadata.oscar_mla,
            causal=True,
            dcp_local_seq_lens=common_attn_metadata.dcp_local_seq_lens,
        )

        return (
            spec_common_attn_metadata,
            token_indices_to_sample,
            num_rejected_tokens_gpu,
        )

    def propose_tree(
        self,
        batch_size: int,
        # [num_tokens, vocab_size]
        logits: torch.Tensor,
        # [num_tokens]
        positions: torch.Tensor,
        # [num_tokens, hidden_size]
        hidden_states: torch.Tensor,
        common_attn_metadata: CommonAttentionMetadata,
        slot_mappings: dict[str, torch.Tensor]
        | list[dict[str, torch.Tensor]]
        | None = None,
    ) -> list[torch.Tensor]:
        tree_attn_metadata_builder = self.draft_attn_groups[0].get_metadata_builder()
        assert isinstance(tree_attn_metadata_builder, TreeAttentionMetadataBuilder)

        total_num_drafts = self.cu_drafts_per_level[0]
        level_num_drafts = total_num_drafts
        # Sample a draft token for each child at the tree root level.
        num_children = self.child_drafts_per_level[0]
        if num_children == 1:
            draft_token_ids = logits.argmax(dim=-1).view(batch_size, -1)
        else:
            draft_token_ids = torch.topk(logits, num_children, dim=-1).indices.view(
                batch_size, -1
            )
        draft_token_ids_list = [draft_token_ids]
        draft_hidden_states = hidden_states.view(batch_size, 1, -1)

        # Initialize empty tensors for concatenation with the level outputs.
        tree_input_ids = torch.empty(
            0, device=self.input_ids.device, dtype=self.input_ids.dtype
        )
        tree_positions = torch.empty(
            0, device=self.positions.device, dtype=self.positions.dtype
        )
        tree_hidden_states = torch.empty(
            0, device=self.hidden_states.device, dtype=self.hidden_states.dtype
        )
        # Precompute the draft token positions.
        flattened_draft_positions = (
            positions.view(batch_size, -1) + self.tree_draft_pos_offsets[:batch_size, :]
        )
        tree_depth = len(self.cu_drafts_per_level)
        for level in range(tree_depth - 1):
            # Get draft positions for RoPE.
            draft_positions = positions + (level + 1)
            exceeds_max_model_len = (positions + total_num_drafts) >= self.max_model_len
            # Mask out the position ids that exceed the max model length.
            # Otherwise, we may get out-of-range error in RoPE.
            draft_positions = torch.where(
                exceeds_max_model_len,
                0,
                draft_positions,
            ).view(batch_size, -1)

            if level_num_drafts > 1:
                # Repeat the positions for each draft at this level.
                draft_positions = draft_positions.repeat_interleave(
                    level_num_drafts, dim=1
                )

            if num_children > 1:
                # Repeat draft hidden states for each child.
                draft_hidden_states = draft_hidden_states.repeat_interleave(
                    num_children, dim=1
                )

            # Concatenate the draft tokens, positions, and hidden states.
            tree_input_ids = torch.cat([tree_input_ids, draft_token_ids], dim=1)
            tree_positions = torch.cat([tree_positions, draft_positions], dim=1)
            tree_hidden_states = torch.cat(
                [tree_hidden_states, draft_hidden_states], dim=1
            )

            # Build new attention metadata for the next level of drafts.
            # This is necessary to support tree attention.
            query_len = total_num_drafts
            common_attn_metadata = replace(
                common_attn_metadata,
                query_start_loc=query_len * self.arange[: batch_size + 1],
                seq_lens=common_attn_metadata.seq_lens + level_num_drafts,
                num_actual_tokens=batch_size * query_len,
                max_query_len=query_len,
            )
            attn_metadata = tree_attn_metadata_builder.build_for_drafting(
                common_attn_metadata=common_attn_metadata, draft_index=level + 1
            )

            # Apply new attention metadata to all draft layers.
            per_layer_attn_metadata = {}
            for attn_group in self.draft_attn_groups:
                for layer_name in attn_group.layer_names:
                    per_layer_attn_metadata[layer_name] = attn_metadata

            # Consider max model length.
            attn_metadata.max_seq_len = min(
                attn_metadata.max_seq_len, self.max_model_len
            )
            # For the requests that exceed the max model length, we set the
            # sequence length to 1 to minimize their overheads in attention.
            attn_metadata.seq_lens.masked_fill_(exceeds_max_model_len, 1)

            # Compute the slot mapping.
            block_size = tree_attn_metadata_builder.kv_cache_spec.block_size
            query_positions = flattened_draft_positions[:, level : level + query_len]
            block_numbers = query_positions // block_size
            block_ids = attn_metadata.block_table.gather(dim=1, index=block_numbers)
            slot_mapping = block_ids * block_size + query_positions % block_size
            # Mask out the slot mappings that exceed the max model length.
            # Otherwise, the KV cache will be inadvertently updated with the
            # padding tokens.
            slot_mapping[exceeds_max_model_len] = PADDING_SLOT_ID
            attn_metadata.slot_mapping = slot_mapping.view(-1)

            # Copy inputs to buffer for cudagraph.
            num_tokens = attn_metadata.num_actual_tokens
            input_ids = tree_input_ids.view(-1)
            self.input_ids[:num_tokens] = input_ids
            self.positions[:num_tokens] = tree_positions.view(-1)
            self.hidden_states[:num_tokens] = tree_hidden_states.view(num_tokens, -1)

            cudagraph_runtime_mode, batch_desc = self.cudagraph_dispatcher.dispatch(
                num_tokens
            )
            num_input_tokens = batch_desc.num_tokens
            # Run the model.
            with set_forward_context(
                per_layer_attn_metadata,
                self.vllm_config,
                num_tokens=num_input_tokens,
                cudagraph_runtime_mode=cudagraph_runtime_mode,
                slot_mapping=self._get_slot_mapping(
                    num_input_tokens, attn_metadata.slot_mapping
                ),
            ):
                last_hidden_states, hidden_states = self.model(
                    input_ids=self.input_ids[:num_input_tokens],
                    positions=self.positions[:num_input_tokens],
                    hidden_states=self.hidden_states[:num_input_tokens],
                    inputs_embeds=None,
                )

            # Get the output hidden states for the draft tokens.
            draft_hidden_states = hidden_states[:num_tokens].view(
                batch_size, query_len, -1
            )[:, -level_num_drafts:]
            draft_last_hidden_states = last_hidden_states[:num_tokens].view(
                batch_size, query_len, -1
            )[:, -level_num_drafts:]

            # Get the output logits for the draft tokens.
            logits = self.model.compute_logits(
                draft_last_hidden_states.reshape(batch_size * level_num_drafts, -1)
            )

            # Sample a draft token for each child at the next tree level.
            num_children = self.child_drafts_per_level[level + 1]
            if num_children == 1:
                draft_token_ids = logits.argmax(dim=-1).view(batch_size, -1)
            else:
                draft_token_ids = torch.topk(logits, num_children, dim=-1).indices.view(
                    batch_size, -1
                )
            draft_token_ids_list.append(draft_token_ids)

            # Update the # drafts counters for the next tree level.
            level_num_drafts = self.cu_drafts_per_level[level + 1] - total_num_drafts
            total_num_drafts = self.cu_drafts_per_level[level + 1]
        return draft_token_ids_list

    def prepare_inputs(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        sampled_token_ids: list[list[int]],
        num_draft_tokens: list[int],
    ) -> tuple[CommonAttentionMetadata, torch.Tensor]:
        """
        This function is used to prepare the inputs for speculative decoding.
        It updates to the common_attn_metadata to account for the rejected
        tokens (and newly sampled tokens). It also returns the token indices
        of the tokens that should be fed to the speculator.
        """
        # E.g.
        #  common_attn_metadata.query_start_loc{_cpu}:
        #       [0, q1, q1 + q2, q1 + q2 + q3]
        #  common_attn_metadata.seq_lens{_cpu}: [s1, s2, s3]
        #  num_rejected_tokens: [n1, n2, n3]
        # This function computes the intermediate values:
        #  num_tokens_per_req: [q1 - n1, q2 - n2, q3 - n3]
        # And returns:
        #  common_attn_metadata.query_start_loc{_cpu}:
        #       [0, q1 - n1, q1 + q2 - n1 - n2, q1 + q2 + q3 - n1 - n2 - n3]
        #  common_attn_metadata.seq_lens{_cpu}:
        #       [s1 - n1 + 1, s2 - n2 + 1, s3 - n3 + 1]
        #  token_indices: [0, 1, ..., q1 - n1 - 1,
        #                 q1, q1 + 1, ..., q1 + q2 - n2 - 1,
        #                 q1 + q2, q1 + q2 + 1, ..., q1 + q2 + q3 - n3 - 1]

        num_rejected_tokens = [
            n + 1 - len(sampled_token_ids[i]) if n > 0 else 0
            for i, n in enumerate(num_draft_tokens)
        ]
        num_rejected_tokens = torch.tensor(num_rejected_tokens, dtype=torch.int32)

        device = common_attn_metadata.query_start_loc.device
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
        # upper_bound - rejected = actual post-rejection seq_lens (no D2H sync).
        assert common_attn_metadata.seq_lens_cpu_upper_bound is not None
        new_seq_lens_cpu = (
            common_attn_metadata.seq_lens_cpu_upper_bound - num_rejected_tokens
        )

        # [0, q1, q1 + q2, q1 + q2 + q3] -> [q1, q2, q3]
        new_query_len_per_req = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
        # [q1, q2, q3] -> [q1 - n1, q2 - n2, q3 - n3]
        new_num_tokens_per_req = new_query_len_per_req - num_rejected_tokens
        new_num_tokens_per_req_np = new_num_tokens_per_req.numpy()

        # [q1 - n1, q2 - n2, q3 - n3] ->
        # [0, q1 - n1, q1 + q2 - n1 - n2, q1 + q2 + q3 - n1 - n2 - n3]
        new_query_start_loc_cpu = torch.zeros(
            query_start_loc_cpu.shape,
            dtype=torch.int32,
            pin_memory=is_pin_memory_available(),
        )
        new_query_start_loc_np = new_query_start_loc_cpu.numpy()
        np.cumsum(new_num_tokens_per_req_np, out=new_query_start_loc_np[1:])

        total_num_tokens = new_query_start_loc_np[-1]
        # Example assuming num_tokens_per_req_np = [2, 4, 3]
        # this implies that `new_query_start_locs` is:
        # [0, 2, 6, 9] ->
        # [0, 0, 2, 2, 2, 2, 6, 6, 6]
        #  _r1_  ____r2____  ___r3__
        new_query_start_locs_expanded = np.repeat(
            new_query_start_loc_np[:-1], new_num_tokens_per_req_np
        )
        # [0, 1, 2, 3, 4, 5, 6, 7, 8] ->
        # [0, 1, 0, 1, 2, 3, 0, 1, 2]
        #  _r1_  ____r2____  ___r3__
        token_offsets = (
            self.token_arange_np[:total_num_tokens] - new_query_start_locs_expanded
        )

        # Expand starting positions to match token pattern
        # [0, q1, q1 + q2] ->
        # [0, 0, q1, q1, q1, q1, q1 + q2, q1 + q2, q1 + q2]
        #  _r1_  _____r2_______  ___________r3____________
        old_query_start_locs_expanded = np.repeat(
            query_start_loc_cpu[:-1].numpy(), new_num_tokens_per_req_np
        )
        # Final token indices are:
        # [0, 1,                                // req 1
        #  q1 + 0, q1 + 1, q1 + 2, q1 + 3,       // req 2
        #  q1 + q2 + 0, q1 + q2 + 1, q1 + q2 + 2] // req 3
        token_indices_np = token_offsets + old_query_start_locs_expanded
        token_indices = torch.from_numpy(token_indices_np).to(device, non_blocking=True)

        spec_common_attn_metadata = CommonAttentionMetadata(
            query_start_loc=new_query_start_loc_cpu.to(device, non_blocking=True),
            seq_lens=new_seq_lens_cpu.to(device, non_blocking=True),
            query_start_loc_cpu=new_query_start_loc_cpu,
            _seq_lens_cpu=new_seq_lens_cpu,
            _num_computed_tokens_cpu=common_attn_metadata._num_computed_tokens_cpu,
            seq_lens_cpu_upper_bound=new_seq_lens_cpu,
            num_reqs=common_attn_metadata.num_reqs,
            num_actual_tokens=total_num_tokens,
            max_query_len=new_query_len_per_req.max().item(),
            max_seq_len=new_seq_lens_cpu.max().item(),
            block_table_tensor=common_attn_metadata.block_table_tensor,
            slot_mapping=common_attn_metadata.slot_mapping[token_indices],
            oscar_mla=common_attn_metadata.oscar_mla,
            causal=True,
            dcp_local_seq_lens=common_attn_metadata.dcp_local_seq_lens,
        )

        return spec_common_attn_metadata, token_indices

    def get_model_name(self, model: nn.Module) -> str:
        if hasattr(model, "module"):  # multi-GPU
            model = model.module
        return model.__class__.__name__

    def _create_draft_vllm_config(self) -> VllmConfig:
        """Return a VllmConfig with kernel-level overrides for the proposer.
        Subclasses may override to apply additional config changes.
        """
        spec_cfg = self.speculative_config
        if spec_cfg.moe_backend is not None:
            return replace(
                self.vllm_config,
                kernel_config=replace(
                    self.vllm_config.kernel_config,
                    moe_backend=spec_cfg.moe_backend,
                ),
            )
        return self.vllm_config

    def _get_model(self) -> nn.Module:
        """
        Default method to call get_model(). Can be overridden by subclasses which
        need to customize model loading.
        """
        from vllm.compilation.backends import set_model_tag

        draft_vllm_config = self._create_draft_vllm_config()
        with set_model_tag("eagle_head"):
            model = get_model(
                vllm_config=draft_vllm_config,
                model_config=self.speculative_config.draft_model_config,
                load_config=self.speculative_config.draft_load_config,
            )
        return model

    def load_model(self, target_model: nn.Module) -> None:
        target_attn_layer_names = set(
            get_layers_from_vllm_config(
                self.vllm_config,
                AttentionLayerBase,  # type: ignore[type-abstract]
            ).keys()
        )

        self.model = self._get_model()

        # Find draft layers (attention layers added by draft model)
        all_attn_layers = get_layers_from_vllm_config(
            self.vllm_config,
            AttentionLayerBase,  # type: ignore[type-abstract]
        )
        self._draft_attn_layer_names = (
            set(all_attn_layers.keys()) - target_attn_layer_names
        )

        if self.supports_mm_inputs:
            # Even if the target model is multimodal, we can also use
            # text-only draft models
            try:
                dummy_input_ids = torch.tensor([[1]], device=self.input_ids.device)
                self.model.embed_input_ids(dummy_input_ids, multimodal_embeddings=None)
            except (NotImplementedError, AttributeError, TypeError):
                logger.warning(
                    "Draft model does not support multimodal inputs, "
                    "falling back to text-only mode"
                )
                self.supports_mm_inputs = False

        if supports_multimodal(target_model):
            # handle multimodality
            assert hasattr(target_model, "config")
            if self.get_model_name(target_model) in [
                "Exaone4_5_ForConditionalGeneration",
                "GlmOcrForConditionalGeneration",
                "HunYuanVLForConditionalGeneration",
                "Qwen2_5_VLForConditionalGeneration",
                "Qwen3_5ForConditionalGeneration",
                "Qwen3_5MoeForConditionalGeneration",
                "Qwen3VLForConditionalGeneration",
                "Qwen3VLMoeForConditionalGeneration",
                "Gemma4ForConditionalGeneration",
            ]:
                self.model.config.image_token_index = target_model.config.image_token_id
            elif self.get_model_name(target_model) == "PixtralForConditionalGeneration":
                self.model.config.image_token_index = (
                    target_model.config.vision_config.image_token_id
                )
            elif self.get_model_name(target_model) == "KimiK25ForConditionalGeneration":
                self.model.config.image_token_index = (
                    target_model.config.media_placeholder_token_id
                )
            else:
                self.model.config.image_token_index = (
                    target_model.config.image_token_index
                )
            target_language_model = cast(
                SupportsMultiModal, target_model
            ).get_language_model()
        else:
            target_language_model = target_model

        self._maybe_share_embeddings(target_language_model)
        self._maybe_share_lm_head(target_language_model)

        if (
            self.parallel_drafting
            and self.pass_hidden_states_to_model
            and self.parallel_drafting_hidden_state_tensor is not None
        ):
            flat_mask = self.model.mask_hidden.view(-1)
            if self.eagle3_use_aux_hidden_state:
                # EAGLE3: mask_hidden stores all aux hidden states,
                # project through combine_hidden_states
                self.parallel_drafting_hidden_state_tensor.copy_(
                    self.model.combine_hidden_states(flat_mask)
                )
            else:
                self.parallel_drafting_hidden_state_tensor.copy_(flat_mask)

        self._maybe_wrap_oscar_mtp_draft_full_cudagraph()

    def _maybe_wrap_oscar_mtp_draft_full_cudagraph(self) -> None:
        cudagraph_mode = self.vllm_config.compilation_config.cudagraph_mode
        if (
            self.method != "mtp"
            or os.environ.get("VLLM_OSCAR_MTP_DRAFT_FULL_CUDAGRAPH", "0") != "1"
            or self.speculative_config.enforce_eager
            or cudagraph_mode is None
            or not cudagraph_mode.has_full_cudagraphs()
            or self.vllm_config.parallel_config.use_ubatching
            or isinstance(self.model, CUDAGraphWrapper)
        ):
            return

        self.model = CUDAGraphWrapper(
            self.model,
            self.vllm_config,
            runtime_mode=CUDAGraphMode.FULL,
            cudagraph_options=CUDAGraphOptions(
                weak_ref_output=False,
                strong_ref_entry_output=True,
            ),
        )

    def _maybe_share_embeddings(self, target_language_model: nn.Module) -> None:
        """
        Some draft models may not have their own embedding layers, and some may
        have a duplicate copy of the target model's embedding layers. In these cases,
        we share the target model's embedding layers with the draft model to save
        memory.
        """
        if get_pp_group().world_size == 1:
            inner_model = getattr(target_language_model, "model", None)
            if inner_model is None:
                raise AttributeError("Target model does not have 'model' attribute")
            if hasattr(inner_model, "embed_tokens"):
                target_embed_tokens = inner_model.embed_tokens
            elif hasattr(inner_model, "embedding"):
                target_embed_tokens = inner_model.embedding
            else:
                raise AttributeError(
                    "Target model does not have 'embed_tokens' or 'embedding' attribute"
                )

            share_embeddings = False
            if hasattr(self.model, "has_own_embed_tokens"):
                # EAGLE model
                if not self.model.has_own_embed_tokens:
                    share_embeddings = True
                    logger.info(
                        "Detected EAGLE model without its own embed_tokens in the"
                        " checkpoint. Sharing target model embedding weights with the"
                        " draft model."
                    )
                elif (
                    isinstance(target_embed_tokens.weight, torch.Tensor)
                    and isinstance(self.model.model.embed_tokens.weight, torch.Tensor)
                    # TODO: Offload to CPU for comparison to avoid extra GPU memory
                    # usage in CI testing environments with limited GPU memory
                    and torch.equal(
                        target_embed_tokens.weight.cpu(),
                        self.model.model.embed_tokens.weight.cpu(),
                    )
                ):
                    share_embeddings = True
                    logger.info(
                        "Detected EAGLE model with embed_tokens identical to the target"
                        " model. Sharing target model embedding weights with the draft"
                        " model."
                    )
                else:
                    logger.info(
                        "Detected EAGLE model with distinct embed_tokens weights. "
                        "Keeping separate embedding weights from the target model."
                    )
            else:
                # MTP model
                share_embeddings = True
                logger.info(
                    "Detected MTP model. "
                    "Sharing target model embedding weights with the draft model."
                )

            if share_embeddings:
                if hasattr(self.model.model, "embed_tokens"):
                    del self.model.model.embed_tokens
                self.model.model.embed_tokens = target_embed_tokens
        else:
            logger.info(
                "The draft model's vocab embedding will be loaded separately"
                " from the target model."
            )

    def _maybe_share_lm_head(self, target_language_model: nn.Module) -> None:
        """
        Some draft models may not have their own LM head, and some may have a
        duplicate copy of the target model's LM head. In these cases, we share
        the target model's LM head with the draft model to save memory.
        """
        share_lm_head = False
        if hasattr(self.model, "has_own_lm_head"):
            # EAGLE model
            if not self.model.has_own_lm_head:
                share_lm_head = True
                logger.info(
                    "Detected EAGLE model without its own lm_head in the checkpoint. "
                    "Sharing target model lm_head weights with the draft model."
                )
            elif (
                hasattr(target_language_model, "lm_head")
                and hasattr(target_language_model.lm_head, "weight")
                and hasattr(self.model.lm_head, "weight")
                and isinstance(target_language_model.lm_head.weight, torch.Tensor)
                and isinstance(self.model.lm_head.weight, torch.Tensor)
                # TODO: Offload to CPU for comparison to avoid extra GPU memory
                # usage in CI testing environments with limited GPU memory
                and torch.equal(
                    target_language_model.lm_head.weight.cpu(),
                    self.model.lm_head.weight.cpu(),
                )
            ):
                share_lm_head = True
                logger.info(
                    "Detected EAGLE model with lm_head identical to the target model. "
                    "Sharing target model lm_head weights with the draft model."
                )
            else:
                logger.info(
                    "Detected EAGLE model with distinct lm_head weights. "
                    "Keeping separate lm_head weights from the target model."
                )
        else:
            # MTP model
            share_lm_head = True
            logger.info(
                "Detected MTP model. "
                "Sharing target model lm_head weights with the draft model."
            )

        if share_lm_head and hasattr(target_language_model, "lm_head"):
            if hasattr(self.model, "lm_head"):
                del self.model.lm_head
            self.model.lm_head = target_language_model.lm_head

            # MTP models call compute_logits via shared_head.head (a
            # ParallelLMHead inside each MTP layer), not self.model.lm_head.
            # If the checkpoint omits a copy of the lm_head weights at the
            # MTP layer path, shared_head.head stays uninitialised and
            # produces NaN logits. Always share it explicitly.
            inner = getattr(self.model, "model", None)
            layers = getattr(inner, "layers", None) if inner else None
            if layers is not None:
                items = layers.values() if isinstance(layers, nn.ModuleDict) else layers
                for layer in items:
                    sh = getattr(layer, "shared_head", None)
                    if sh is not None and hasattr(sh, "head"):
                        del sh.head
                        sh.head = target_language_model.lm_head
                        logger.info(
                            "Shared target model lm_head with MTP shared_head.head."
                        )

        if self.use_local_argmax_reduction:
            if not hasattr(self.model, "get_top_tokens"):
                raise ValueError(
                    "use_local_argmax_reduction is enabled but draft model "
                    f"{self.model.__class__.__name__} does not implement "
                    "get_top_tokens()."
                )
            # Warn if draft model has vocab remapping, which forces fallback
            # to the full-logits path (negating the optimization).
            if (
                hasattr(self.model, "draft_id_to_target_id")
                and self.model.draft_id_to_target_id is not None
            ):
                logger.warning(
                    "use_local_argmax_reduction is enabled but draft model "
                    "uses draft_id_to_target_id vocab remapping. The "
                    "optimization will be bypassed (falling back to full "
                    "logits gather + argmax)."
                )
            else:
                logger.info(
                    "Using local argmax reduction for draft token generation "
                    "(communication: O(2*tp_size) vs O(vocab_size))."
                )

    @torch.inference_mode()
    def dummy_run(
        self,
        num_tokens: int,
        use_cudagraphs: bool = True,
        is_graph_capturing: bool = False,
        slot_mappings: dict[str, torch.Tensor] | None = None,
        common_attn_metadata: CommonAttentionMetadata | None = None,
    ) -> None:
        # FIXME: when using tree-based specdec, adjust number of forward-passes
        # according to the depth of the tree.
        capture_mtp_draft_full_cudagraph = (
            is_graph_capturing
            and use_cudagraphs
            and self.method == "mtp"
            and os.environ.get("VLLM_OSCAR_MTP_DRAFT_FULL_CUDAGRAPH", "0") == "1"
        )
        only_one_forward_pass = (
            is_graph_capturing and not capture_mtp_draft_full_cudagraph
        ) or self.parallel_drafting
        num_forward_passes = 1 if only_one_forward_pass else self.num_speculative_tokens
        if capture_mtp_draft_full_cudagraph:
            assert common_attn_metadata is not None, (
                "MTP draft FULL cudagraph capture requires attention metadata"
            )
            draft_capture_sizes = self._oscar_mtp_draft_decode_capture_sizes
            if not draft_capture_sizes:
                draft_capture_sizes = (common_attn_metadata.batch_size(),)
            capture_batch_size = common_attn_metadata.batch_size()
            captured_sizes = getattr(
                self,
                "_oscar_mtp_draft_graph_capture_metadata",
                {},
            )
            draft_capture_sizes = tuple(
                sorted(
                    (
                        size
                        for size in draft_capture_sizes
                        if size <= capture_batch_size and size not in captured_sizes
                    ),
                    reverse=True,
                )
            )
            num_forward_passes = 1 + len(draft_capture_sizes)

        for fwd_idx in range(num_forward_passes):
            if capture_mtp_draft_full_cudagraph:
                dispatch_num_tokens = (
                    num_tokens if fwd_idx == 0 else draft_capture_sizes[fwd_idx - 1]
                )
                cudagraph_runtime_mode, num_input_tokens, num_tokens_across_dp = (
                    self._determine_batch_execution_and_padding(
                        dispatch_num_tokens,
                        use_cudagraphs=use_cudagraphs,
                        uniform_decode=fwd_idx > 0,
                    )
                )
            elif fwd_idx <= 1:
                dispatch_num_tokens = num_tokens
                cudagraph_runtime_mode, num_input_tokens, num_tokens_across_dp = (
                    self._determine_batch_execution_and_padding(
                        dispatch_num_tokens,
                        use_cudagraphs=use_cudagraphs,
                        uniform_decode=False,
                    )
                )

            if (
                capture_mtp_draft_full_cudagraph
                and self._share_mtp_indices
                and hasattr(self.model.model, "set_skip_topk")
            ):
                # GLM-5.2 computes sparse indices on the first MTP pass and
                # reuses them on later draft passes. Freeze each FULL graph
                # with the same branch state used by real requests.
                self.model.model.set_skip_topk(fwd_idx > 0)

            per_layer_attn_metadata = None
            if capture_mtp_draft_full_cudagraph:
                assert common_attn_metadata is not None
                capture_common_attn_metadata = common_attn_metadata
                if fwd_idx > 0:
                    batch_size = draft_capture_sizes[fwd_idx - 1]
                    assert num_input_tokens == batch_size
                    capture_oscar_mla = common_attn_metadata.oscar_mla
                    if capture_oscar_mla is not None:
                        capture_oscar_mla = replace(
                            capture_oscar_mla,
                            hp_rows=capture_oscar_mla.hp_rows[:batch_size],
                            decode_positions=capture_oscar_mla.decode_positions[
                                :batch_size
                            ],
                            final_seq_lens=capture_oscar_mla.final_seq_lens[
                                :batch_size
                            ],
                            history_page_table=capture_oscar_mla.history_page_table[
                                :batch_size
                            ],
                            previous_seq_lens=capture_oscar_mla.previous_seq_lens[
                                :batch_size
                            ],
                        )
                    capture_common_attn_metadata = common_attn_metadata.unpadded(
                        batch_size,
                        batch_size,
                    ).replace(
                        oscar_mla=capture_oscar_mla,
                        max_query_len=1,
                        query_start_loc=self.arange[: batch_size + 1],
                        query_start_loc_cpu=torch.from_numpy(
                            self.token_arange_np[: batch_size + 1]
                        ).clone(),
                        slot_mapping=self._slot_mapping_buffer[:batch_size],
                    )
                _, per_layer_attn_metadata = (
                    self.build_per_group_and_layer_attn_metadata(
                        capture_common_attn_metadata,
                        draft_index=min(fwd_idx, 1),
                    )
                )

            # Make sure to use EAGLE's own buffer during cudagraph capture.
            if (
                self._draft_attn_layer_names
                and slot_mappings is not None
                and next(iter(self._draft_attn_layer_names)) in slot_mappings
            ):
                slot_mapping_dict = self._get_slot_mapping(num_input_tokens)
            else:
                slot_mapping_dict = slot_mappings or {}

            if self.supports_mm_inputs:
                input_ids = None
                inputs_embeds = self.inputs_embeds[:num_input_tokens]
            else:
                input_ids = self.input_ids[:num_input_tokens]
                inputs_embeds = None

            kwargs = dict(
                input_ids=input_ids,
                positions=self._get_positions(num_input_tokens),
                inputs_embeds=inputs_embeds,
            )
            if self.pass_hidden_states_to_model:
                kwargs["hidden_states"] = self.hidden_states[:num_input_tokens]

            if (
                capture_mtp_draft_full_cudagraph
                and fwd_idx > 0
                and os.environ.get("VLLM_USE_NCCL_SYMM_MEM", "0") == "1"
            ):
                # NCCL symmetric-memory buffers must be allocated and their
                # communication windows registered before CUDA graph capture.
                # An eager pass with the exact draft shape lets the memory pool
                # reuse those registered segments during the following capture.
                with set_forward_context(
                    per_layer_attn_metadata,
                    self.vllm_config,
                    num_tokens=num_input_tokens,
                    num_tokens_across_dp=num_tokens_across_dp,
                    cudagraph_runtime_mode=CUDAGraphMode.NONE,
                    slot_mapping=slot_mapping_dict,
                ):
                    self.model(**kwargs)

            with set_forward_context(
                per_layer_attn_metadata,
                self.vllm_config,
                num_tokens=num_input_tokens,
                num_tokens_across_dp=num_tokens_across_dp,
                cudagraph_runtime_mode=cudagraph_runtime_mode,
                slot_mapping=slot_mapping_dict,
            ):
                if (
                    capture_mtp_draft_full_cudagraph
                    and os.environ.get("VLLM_OSCAR_MTP_DRAFT_INT2", "0") == "1"
                    and fwd_idx > 0
                ):
                    self._remember_oscar_mtp_draft_graph_metadata(
                        num_input_tokens,
                        per_layer_attn_metadata or {},
                    )
                self.model(**kwargs)

    def _get_eagle3_use_aux_hidden_state_from_config(self) -> bool:
        """
        Some eagle3 heads (e.g., nvidia/gpt-oss-120b-Eagle3-v2) do not use auxiliary
        hidden states and directly uses the last layer output just like eagle1.
        They might indicate this by setting "use_aux_hidden_state" to False
        inside the "eagle_config" dict of their hf_config.
        """
        if self.method != "eagle3":
            return False
        # Assume that eagle3 heads use aux hidden states by default
        use_aux_hidden_state = True
        eagle_config = getattr(self.draft_model_config.hf_config, "eagle_config", None)
        if eagle_config is not None:
            use_aux_hidden_state = eagle_config.get("use_aux_hidden_state", True)
        return use_aux_hidden_state

    def validate_same_kv_cache_group(self, kv_cache_config: KVCacheConfig) -> None:
        """
        Validate that all drafting layers belong to the same KVCacheGroup.
        Need this assumption to ensure all drafting layers can use the
        same AttentionMetadata.
        May extend to multiple AttentionMetadata in the future.
        """
        kv_cache_groups: dict[str, int] = {}
        for id, kv_cache_group in enumerate(kv_cache_config.kv_cache_groups):
            for layer_name in kv_cache_group.layer_names:
                kv_cache_groups[layer_name] = id
        assert (
            len(
                set(
                    [
                        kv_cache_groups[layer_name]
                        for layer_name in self._draft_attn_layer_names
                    ]
                )
            )
            == 1
        ), "All drafting layers should belong to the same kv cache group"

    def initialize_attn_backend(
        self,
        kv_cache_config: KVCacheConfig,
        kernel_block_sizes: list[int] | None = None,
    ) -> None:
        """
        Initialize AttentionGroups for draft layers using kv_cache_config.
        Called from the model runner's initialize_metadata_builders.
        """
        all_attn_layers = get_layers_from_vllm_config(
            self.vllm_config,
            AttentionLayerBase,  # type: ignore[type-abstract]
        )

        # Find which kv_cache_group the draft layers belong to
        self.validate_same_kv_cache_group(kv_cache_config)
        kv_cache_spec = None
        for gid, group in enumerate(kv_cache_config.kv_cache_groups):
            if self._draft_attn_layer_names & set(group.layer_names):
                self.kv_cache_gid = gid
                kv_cache_spec = group.kv_cache_spec
                break

        attention_groups: dict[tuple[str, str], AttentionGroup] = {}
        if kv_cache_spec is not None:
            for layer_name in self._draft_attn_layer_names:
                attn_backend = all_attn_layers[layer_name].get_attn_backend()
                backend_key = attn_backend.full_cls_name()
                if backend_key not in attention_groups:
                    layer_kv_cache_spec = kv_cache_spec
                    if isinstance(layer_kv_cache_spec, UniformTypeKVCacheSpecs):
                        layer_kv_cache_spec = layer_kv_cache_spec.kv_cache_specs[
                            layer_name
                        ]

                    kernel_block_size = (
                        kernel_block_sizes[self.kv_cache_gid]
                        if kernel_block_sizes is not None
                        and self.kv_cache_gid < len(kernel_block_sizes)
                        else None
                    )
                    attn_group = AttentionGroup(
                        backend=attn_backend,
                        layer_names=[layer_name],
                        kv_cache_spec=layer_kv_cache_spec,
                        kv_cache_group_id=self.kv_cache_gid,
                    )
                    attn_group.create_metadata_builders(
                        self.vllm_config,
                        self.device,
                        kernel_block_size=kernel_block_size,
                    )
                    attention_groups[backend_key] = attn_group
                else:
                    attention_groups[backend_key].layer_names.append(layer_name)

        self.draft_attn_groups = list(attention_groups.values())
        self.block_size = (
            self.draft_attn_groups[0].get_metadata_builder().kv_cache_spec.block_size
        )
        logger.debug("Using block size %d for drafting layers", self.block_size)

    def _determine_batch_execution_and_padding(
        self,
        num_tokens: int,
        use_cudagraphs: bool = True,
        uniform_decode: bool = False,
    ) -> tuple[CUDAGraphMode, int, torch.Tensor | None]:
        cudagraph_mode, batch_desc = self.cudagraph_dispatcher.dispatch(
            num_tokens,
            uniform_decode=uniform_decode,
            valid_modes=({CUDAGraphMode.NONE} if not use_cudagraphs else None),
        )
        num_tokens_padded = batch_desc.num_tokens

        # Extra coordination when running data-parallel since we need to
        # coordinate across ranks
        # TODO(Flechman): support DBO ubatching
        should_ubatch, num_tokens_across_dp = False, None
        if self.vllm_config.parallel_config.data_parallel_size > 1:
            should_ubatch, num_tokens_across_dp, synced_cudagraph_mode = (
                coordinate_batch_across_dp(
                    num_tokens_unpadded=num_tokens,
                    parallel_config=self.vllm_config.parallel_config,
                    allow_microbatching=False,
                    num_tokens_padded=num_tokens_padded,
                    cudagraph_mode=cudagraph_mode.value,
                )
            )
            assert not should_ubatch, "DBO ubatching not implemented for EAGLE"

            # Extract DP-synced values
            if num_tokens_across_dp is not None:
                dp_rank = self.dp_rank
                num_tokens_padded = int(num_tokens_across_dp[dp_rank].item())
                # Re-dispatch with DP padding so we have the correct
                # batch_descriptor
                cudagraph_mode, batch_desc = self.cudagraph_dispatcher.dispatch(
                    num_tokens_padded,
                    uniform_decode=uniform_decode,
                    valid_modes={CUDAGraphMode(synced_cudagraph_mode)},
                )
                # Assert to make sure the agreed upon token count is correct
                # otherwise num_tokens_across_dp will no-longer be valid
                assert batch_desc.num_tokens == num_tokens_padded
                num_tokens_across_dp[dp_rank] = num_tokens_padded

        return cudagraph_mode, num_tokens_padded, num_tokens_across_dp


# NOTE(woosuk): Currently, the below code is not used and we always use argmax
# to sample the draft tokens. We will use this after we find a way to manage
# the draft prob tensor.
# Refer to https://github.com/vllm-project/vllm/pull/16899 for the details.
# FIXME(woosuk): The logic here is duplicated with the main sampling code.
# We should refactor this to reuse the same sampling implementation.
def compute_probs_and_sample_next_token(
    logits: torch.Tensor,
    sampling_metadata: SamplingMetadata,
) -> tuple[torch.Tensor, torch.Tensor]:
    if sampling_metadata.all_greedy:
        # For greedy requests, draft_probs is not used in rejection sampling.
        # Therefore, we can just return the logits.
        probs = logits
        next_token_ids = logits.argmax(dim=-1)
        return next_token_ids, probs

    assert sampling_metadata.temperature is not None

    # Use epsilon comparison to detect greedy sampling (temperature ~ 0.0)
    # consistent with sampler.py's _SAMPLING_EPS threshold
    temperature = sampling_metadata.temperature
    # Avoid division by zero if there are greedy requests.
    if not sampling_metadata.all_random:
        is_greedy = temperature < _SAMPLING_EPS
        temperature = torch.where(is_greedy, 1.0, temperature)
    logits.div_(temperature.view(-1, 1))
    probs = logits.softmax(dim=-1, dtype=torch.float32)

    # NOTE(woosuk): Currently, we ignore most of the sampling parameters in
    # generating the draft tokens. We only use the temperature. While this
    # could degrade the acceptance rate, it does not affect the distribution
    # of the generated tokens after rejection sampling.

    # TODO(woosuk): Consider seeds.
    q = torch.empty_like(probs)
    q.exponential_()
    # NOTE(woosuk): We shouldn't use `probs.div_(q)` because the draft_probs
    # will be used later for rejection sampling.
    next_token_ids = probs.div(q).argmax(dim=-1).view(-1)
    if not sampling_metadata.all_random:
        greedy_token_ids = probs.argmax(dim=-1)
        next_token_ids = torch.where(is_greedy, greedy_token_ids, next_token_ids)
    return next_token_ids, probs
