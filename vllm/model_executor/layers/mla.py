# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
import time
from dataclasses import dataclass

import torch

from vllm.config import CacheConfig
from vllm.logger import init_logger
from vllm.model_executor.custom_op import PluggableLayer
from vllm.model_executor.layers.attention import MLAAttention
from vllm.model_executor.layers.quantization import QuantizationConfig

logger = init_logger(__name__)


def _mtp_mla_debug_enabled(
    prefix: str,
    positions: torch.Tensor,
) -> tuple[bool, int | None]:
    if os.environ.get("VLLM_MTP_LAYER_DEBUG", "0") != "1":
        return False, None
    if ".layers." not in prefix:
        return False, None
    try:
        layer_idx = int(prefix.split(".layers.", 1)[1].split(".", 1)[0])
        min_layer = int(os.environ.get("VLLM_MTP_LAYER_DEBUG_MIN_LAYER", "80") or "80")
    except (IndexError, ValueError):
        return False, None
    if layer_idx < min_layer:
        return False, None
    try:
        min_pos = int(os.environ.get("VLLM_MTP_LAYER_DEBUG_MIN_POS", "49900") or "0")
    except ValueError:
        min_pos = 49900
    max_pos = None
    try:
        flat = positions.reshape(-1)
        if flat.numel() > 0:
            max_pos = int(flat.max().detach().cpu().item())
    except Exception:
        max_pos = None
    if max_pos is not None and max_pos < min_pos:
        return False, max_pos
    return True, max_pos


def _mtp_mla_debug_start() -> float | None:
    if os.environ.get("VLLM_MTP_LAYER_DEBUG", "0") != "1":
        return None
    return time.perf_counter()


def _mtp_mla_debug_log(
    phase: str,
    prefix: str,
    positions: torch.Tensor,
    start: float | None = None,
    **extra,
) -> None:
    enabled, max_pos = _mtp_mla_debug_enabled(prefix, positions)
    if not enabled:
        return
    elapsed_ms = None if start is None else (time.perf_counter() - start) * 1000.0
    logger.warning(
        "MTP MLA debug: phase=%s prefix=%s max_pos=%s elapsed_ms=%s extra=%s",
        phase,
        prefix,
        max_pos,
        None if elapsed_ms is None else round(elapsed_ms, 3),
        extra,
    )


@dataclass
class MLAModules:
    """Modules used in MLA."""

    kv_a_layernorm: torch.nn.Module
    kv_b_proj: torch.nn.Module
    rotary_emb: torch.nn.Module
    o_proj: torch.nn.Module
    fused_qkv_a_proj: torch.nn.Module | None
    kv_a_proj_with_mqa: torch.nn.Module | None
    q_a_layernorm: torch.nn.Module | None
    q_b_proj: torch.nn.Module | None
    q_proj: torch.nn.Module | None
    indexer: torch.nn.Module | None
    is_sparse: bool
    topk_indices_buffer: torch.Tensor | None
    indexer_rotary_emb: torch.nn.Module | None = None


# --8<-- [start:multi_head_latent_attention]
@PluggableLayer.register("multi_head_latent_attention")
class MultiHeadLatentAttentionWrapper(PluggableLayer):
    """Pluggable MLA layer which allows OOT backends to add
    custom implementations of the outer MLA layer (including rope & o_proj).
    Note that currently oot platforms can still use CustomOp.register_oot to
    replace MLA layer entirely, although we use PluggableLayer to register
    this layer now.

    This class takes positions and hidden_states as input.
    The input tensors can either contain prefill tokens or decode tokens.
    The class does the following:

    1. MLA Preprocess.
    2. Perform multi-head attention to prefill tokens and
       multi-query attention to decode tokens separately.
    3. Return the output tensor.
    """

    # --8<-- [end:multi_head_latent_attention]

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        scale: float,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int | None,
        kv_lora_rank: int,
        mla_modules: MLAModules,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.num_heads = num_heads
        self.fused_qkv_a_proj = mla_modules.fused_qkv_a_proj
        self.kv_a_proj_with_mqa = mla_modules.kv_a_proj_with_mqa
        self.q_a_layernorm = mla_modules.q_a_layernorm
        self.q_b_proj = mla_modules.q_b_proj
        self.q_proj = mla_modules.q_proj
        self.kv_a_layernorm = mla_modules.kv_a_layernorm
        self.kv_b_proj = mla_modules.kv_b_proj
        self.rotary_emb = mla_modules.rotary_emb
        self.o_proj = mla_modules.o_proj
        self.indexer = mla_modules.indexer
        self.indexer_rope_emb = mla_modules.indexer_rotary_emb
        self.is_sparse = mla_modules.is_sparse

        if self.indexer is not None:
            assert hasattr(self.indexer, "topk_tokens")
            self.topk_tokens = self.indexer.topk_tokens
            self.topk_indices_buffer = mla_modules.topk_indices_buffer

        self.mla_attn = MLAAttention(
            num_heads=self.num_heads,
            scale=scale,
            qk_nope_head_dim=self.qk_nope_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
            v_head_dim=self.v_head_dim,
            q_lora_rank=self.q_lora_rank,
            kv_lora_rank=self.kv_lora_rank,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            kv_b_proj=self.kv_b_proj,
            use_sparse=self.is_sparse,
            indexer=self.indexer,
        )

        self.prefix = prefix

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _mtp_mla_debug_log(
            "enter",
            self.prefix,
            positions,
            hidden_shape=tuple(hidden_states.shape),
            sparse=self.is_sparse,
            has_indexer=self.indexer is not None,
        )
        q_c = None
        kv_lora = None

        if self.q_lora_rank is not None:
            assert self.fused_qkv_a_proj is not None, (
                "fused_qkv_a_proj is required when q_lora_rank is not None"
            )
            assert self.q_a_layernorm is not None, (
                "q_a_layernorm is required when q_lora_rank is not None"
            )
            assert self.q_b_proj is not None, (
                "q_b_proj is required when q_lora_rank is not None"
            )

            phase_start = _mtp_mla_debug_start()
            qkv_lora = self.fused_qkv_a_proj(hidden_states)[0]
            _mtp_mla_debug_log(
                "fused_qkv_a_proj_done",
                self.prefix,
                positions,
                phase_start,
                qkv_shape=tuple(qkv_lora.shape),
            )
            q_c, kv_lora = qkv_lora.split(
                [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
                dim=-1,
            )
            phase_start = _mtp_mla_debug_start()
            q_c = self.q_a_layernorm(q_c)
            _mtp_mla_debug_log(
                "q_a_layernorm_done",
                self.prefix,
                positions,
                phase_start,
                q_c_shape=tuple(q_c.shape),
            )
            phase_start = _mtp_mla_debug_start()
            q = self.q_b_proj(q_c)[0]
            _mtp_mla_debug_log(
                "q_b_proj_done",
                self.prefix,
                positions,
                phase_start,
                q_shape=tuple(q.shape),
            )
        else:
            assert self.kv_a_proj_with_mqa is not None, (
                "kv_a_proj_with_mqa is required when q_lora_rank is None"
            )
            assert self.q_proj is not None, (
                "q_proj is required when q_lora_rank is None"
            )
            kv_lora = self.kv_a_proj_with_mqa(hidden_states)[0]
            q = self.q_proj(hidden_states)[0]

        kv_c, k_pe = kv_lora.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kv_c_normed = self.kv_a_layernorm(kv_c)

        q = q.view(-1, self.num_heads, self.qk_head_dim)
        # Add head dim of 1 to k_pe
        k_pe = k_pe.unsqueeze(1)

        if self.rotary_emb is not None:
            phase_start = _mtp_mla_debug_start()
            q[..., self.qk_nope_head_dim :], k_pe = self.rotary_emb(
                positions, q[..., self.qk_nope_head_dim :], k_pe
            )
            _mtp_mla_debug_log("rotary_done", self.prefix, positions, phase_start)

        if self.indexer and self.is_sparse:
            phase_start = _mtp_mla_debug_start()
            _mtp_mla_debug_log("indexer_enter", self.prefix, positions)
            _topk_indices = self.indexer(
                hidden_states, q_c, positions, self.indexer_rope_emb
            )
            _mtp_mla_debug_log("indexer_done", self.prefix, positions, phase_start)

        if llama_4_scaling is not None:
            q *= llama_4_scaling

        phase_start = _mtp_mla_debug_start()
        _mtp_mla_debug_log("mla_attn_enter", self.prefix, positions)
        attn_out = self.mla_attn(
            q,
            kv_c_normed,
            k_pe,
            output_shape=(hidden_states.shape[0], self.num_heads * self.v_head_dim),
        )
        _mtp_mla_debug_log(
            "mla_attn_done",
            self.prefix,
            positions,
            phase_start,
            attn_out_shape=tuple(attn_out.shape),
        )

        phase_start = _mtp_mla_debug_start()
        output = self.o_proj(attn_out)[0]
        _mtp_mla_debug_log(
            "o_proj_done",
            self.prefix,
            positions,
            phase_start,
            output_shape=tuple(output.shape),
        )
        return output
