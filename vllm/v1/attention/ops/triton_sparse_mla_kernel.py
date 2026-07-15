# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton sparse MLA attention with split-KV for low-batch decode.

Stage 1 runs the sparse attention over a contiguous slice of the topk
axis (or the full axis when `num_kv_splits=1`) and writes a partial
`(out/e_sum, lse)` tile to a mid buffer. Stage 2 merges the splits via
online-softmax rescaling — pattern from `triton_decode_attention.py`.
"""

import functools
import logging
import os

import torch

from vllm.triton_utils import LOG2E, LOGE2, tl, triton
from vllm.utils.platform_utils import num_compute_units

logger = logging.getLogger(__name__)

# DeepSeek-V3.2 / GLM-5 sparse MLA shape constants.
_BLOCK_DMODEL = 512
_BLOCK_DPE = 64
_BLOCK_DV = 512
_DIM_QK = _BLOCK_DMODEL + _BLOCK_DPE  # 576

_BLOCK_H = int(os.getenv("VLLM_SPARSE_MLA_BLOCK_H", "32"))


def _env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default) == "1"


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)) or str(default))


# Smallest BLOCK_N the autotune sweep offers; only used for the topk-divisibility
# check at dispatch time.
_MIN_BLOCK_N = 16

# Merge kernel is launch-bound on a (1, 1) grid — one CTA per token starves the
# SMs. Spread across heads and DV tiles (pattern from FlashMLA's combine kernel
# at FlashMLA/csrc/smxx/decode/combine/combine.cu:22-27). BLOCK_H=1 so each
# of the 8 per-rank heads runs concurrently; BLOCK_DV_TILE=128 splits the 512
# output lanes into 4 tiles.
_MERGE_BLOCK_H = 1
_MERGE_BLOCK_DV_TILE = 128
assert _BLOCK_DV % _MERGE_BLOCK_DV_TILE == 0
_NUM_MERGE_DV_TILES = _BLOCK_DV // _MERGE_BLOCK_DV_TILE

# Separate config sweeps for the single-pass (prefill) and split-KV (decode)
# entry points. Per A100/SM80 sweeps:
#   - Single-pass ("final") kernel at prefill M>>1 prefers BLOCK_N=16 with few
#     warps; the wider configs in the combined sweep were landing 1.3–1.5×
#     slower because autotune's key omits M and the cached pick was a bad
#     compromise.
#   - Split kernel at decode M=1 prefers BLOCK_N=32/num_warps=4 across every
#     split count we tested.
# Each kernel only ever runs in its own regime (see `_choose_num_kv_splits`),
# so we can tune each independently.
def _get_final_autotune_configs() -> list[triton.Config]:
    forced_config = os.getenv("VLLM_SPARSE_MLA_FINAL_CONFIG")
    if forced_config:
        block_n, num_warps, num_stages = (
            int(part) for part in forced_config.split(",")
        )
        return [
            triton.Config(
                {"BLOCK_N": block_n},
                num_warps=num_warps,
                num_stages=num_stages,
            )
        ]
    if os.getenv("VLLM_SPARSE_MLA_FINAL_FAST_SWEEP", "0") == "1":
        return [
            triton.Config({"BLOCK_N": 16}, num_warps=4, num_stages=4),
            triton.Config({"BLOCK_N": 32}, num_warps=4, num_stages=1),
        ]
    return [
        triton.Config({"BLOCK_N": bn}, num_warps=nw, num_stages=ns)
        for bn in (16, 32)
        for nw in (1, 2, 4)
        for ns in (1, 2, 4)
    ]


_FINAL_AUTOTUNE_CONFIGS = _get_final_autotune_configs()
_SPLIT_AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_N": 32}, num_warps=4, num_stages=ns) for ns in (2, 4)
]

# Split count candidates that `_choose_num_kv_splits` can return; also the set
# `_warmup_autotune` pre-compiles so the first decode does not pay the sweep cost.
KV_SPLITS_CANDIDATES = (1, 2, 4, 8, 16)

# Split-KV heuristic tuning.
# At topk=2048 (DSv3.2/GLM-5.1) this unlocks 16-way split for decode, which
# benches ~1.3× faster than 8-way on A100 SM80 at BLOCK_N=32/num_warps=4.
_MIN_TOPK_PER_SPLIT = 128  # below this, per-split work is too small to amortize
_SPLIT_MAX_OCCUPANCY = 4  # skip split when baseline grid fills >=1/4 of SMs

# Stage TPOT services restart per candidate, so these sparse DSA runtime
# controls are fixed for the process lifetime.
_FORCE_KV_SPLITS = os.getenv("VLLM_SPARSE_MLA_FORCE_KV_SPLITS")
_REUSE_K_AS_V = _env_flag("VLLM_SPARSE_MLA_REUSE_K_AS_V")
_FULL_BLOCK_H_ENABLED = _env_flag("VLLM_SPARSE_MLA_FULL_BLOCK_H")
_FULL_BLOCK_H_MAX_TOKENS = _env_int("VLLM_SPARSE_MLA_FULL_BLOCK_H_MAX_TOKENS", 0)
_ASSUME_VALID_NOMASK_ENABLED = _env_flag("VLLM_SPARSE_MLA_ASSUME_VALID_NOMASK")
_ASSUME_VALID_AFTER_TOPK_NOMASK_ENABLED = _env_flag(
    "VLLM_SPARSE_MLA_ASSUME_VALID_AFTER_TOPK_NOMASK"
)
_ASSUME_PREFIX_LEN_MASK_ENABLED = _env_flag(
    "VLLM_SPARSE_MLA_ASSUME_PREFIX_LEN_MASK"
)
_FINAL_DYNAMIC_CONFIG_ENABLED = _env_flag("VLLM_SPARSE_MLA_FINAL_DYNAMIC_CONFIG")
_FINAL_STATIC_BY_TOKENS_ENABLED = _env_flag(
    "VLLM_SPARSE_MLA_FINAL_STATIC_BY_TOKENS"
)
_DECODE_M1_DV_TILE_FINAL_ENABLED = _env_flag(
    "VLLM_SPARSE_MLA_DECODE_M1_DV_TILE_FINAL"
)
_DECODE_M1_DV_TILE = _env_int("VLLM_SPARSE_MLA_DECODE_M1_DV_TILE", 128)
_DECODE_M1_DV_TILE_BLOCK_H = _env_int(
    "VLLM_SPARSE_MLA_DECODE_M1_DV_TILE_BLOCK_H", _BLOCK_H
)
_DECODE_M1_DV_TILE_BLOCK_N = _env_int(
    "VLLM_SPARSE_MLA_DECODE_M1_DV_TILE_BLOCK_N", 32
)
_DECODE_M1_DV_TILE_NUM_WARPS = _env_int(
    "VLLM_SPARSE_MLA_DECODE_M1_DV_TILE_NUM_WARPS", 4
)
_DECODE_M1_DV_TILE_NUM_STAGES = _env_int(
    "VLLM_SPARSE_MLA_DECODE_M1_DV_TILE_NUM_STAGES", 1
)
_M1_COOP_FINAL_ENABLED = _env_flag("VLLM_SPARSE_MLA_M1_COOP_FINAL")
_M1_COOP_FINAL_NUM_SPLITS = _env_int(
    "VLLM_SPARSE_MLA_M1_COOP_FINAL_NUM_SPLITS", 32
)
_M1_COOP_FINAL_LIB_PATH = os.getenv("VLLM_SPARSE_MLA_M1_COOP_FINAL_LIB", "")
_M1_COOP_FINAL_DEBUG_LOGS_REMAINING = _env_int(
    "VLLM_SPARSE_MLA_M1_COOP_FINAL_DEBUG_LOGS", 0
)
_M1_COOP_FINAL_SYNC_DEBUG = _env_flag(
    "VLLM_SPARSE_MLA_M1_COOP_FINAL_SYNC_DEBUG"
)
_M1_SPLITMERGE_FINAL_ENABLED = _env_flag(
    "VLLM_SPARSE_MLA_M1_SPLITMERGE_FINAL"
) and _env_flag(
    "VLLM_SPARSE_MLA_M1_SPLITMERGE_FINAL_UNSAFE_ENABLE"
)
_M1_SPLITMERGE_FINAL_NUM_SPLITS = _env_int(
    "VLLM_SPARSE_MLA_M1_SPLITMERGE_FINAL_NUM_SPLITS", 32
)
_M1_SPLITMERGE_FINAL_LIB_PATH = os.getenv(
    "VLLM_SPARSE_MLA_M1_SPLITMERGE_FINAL_LIB", ""
)
_M1_SPLITMERGE_FINAL_DEBUG_LOGS_REMAINING = _env_int(
    "VLLM_SPARSE_MLA_M1_SPLITMERGE_FINAL_DEBUG_LOGS", 0
)
_M1_SPLITMERGE_FINAL_SYNC_DEBUG = _env_flag(
    "VLLM_SPARSE_MLA_M1_SPLITMERGE_FINAL_SYNC_DEBUG"
)


def _current_workspace_manager():
    from vllm.v1.worker.workspace import current_workspace_manager

    return current_workspace_manager()


def _log_m1_splitmerge_final_dispatch(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    out: torch.Tensor,
    num_heads_q: int,
    index_topk: int,
    convert_req_to_global: bool,
) -> None:
    global _M1_SPLITMERGE_FINAL_DEBUG_LOGS_REMAINING
    if _M1_SPLITMERGE_FINAL_DEBUG_LOGS_REMAINING <= 0:
        return
    _M1_SPLITMERGE_FINAL_DEBUG_LOGS_REMAINING -= 1
    logger.warning(
        "Stage50 sparse_mla_m1_splitmerge_final dispatch: q_nope=%s "
        "stride=%s q_pe=%s stride=%s kv=%s stride=%s indices=%s stride=%s "
        "out=%s stride=%s heads=%s topk=%s convert_req_to_global=%s",
        tuple(q_nope.shape),
        q_nope.stride(),
        tuple(q_pe.shape),
        q_pe.stride(),
        tuple(kv.shape),
        kv.stride(),
        tuple(indices.shape),
        indices.stride(),
        tuple(out.shape),
        out.stride(),
        num_heads_q,
        index_topk,
        convert_req_to_global,
    )


def _has_m1_splitmerge_final_op() -> bool:
    try:
        return bool(
            torch._C._dispatch_has_kernel_for_dispatch_key(
                "_C::sparse_mla_m1_splitmerge_final_cuda", "CUDA"
            )
        )
    except RuntimeError:
        return False


@functools.lru_cache(maxsize=1)
def _load_m1_splitmerge_final_op() -> None:
    if _has_m1_splitmerge_final_op():
        return
    if not _M1_SPLITMERGE_FINAL_LIB_PATH:
        raise RuntimeError(
            "VLLM_SPARSE_MLA_M1_SPLITMERGE_FINAL=1 requires "
            "VLLM_SPARSE_MLA_M1_SPLITMERGE_FINAL_LIB"
        )
    if not os.path.exists(_M1_SPLITMERGE_FINAL_LIB_PATH):
        raise RuntimeError(
            "VLLM_SPARSE_MLA_M1_SPLITMERGE_FINAL_LIB does not exist: "
            f"{_M1_SPLITMERGE_FINAL_LIB_PATH}"
        )
    torch.ops.load_library(_M1_SPLITMERGE_FINAL_LIB_PATH)
    if not _has_m1_splitmerge_final_op():
        raise RuntimeError(
            "loaded VLLM_SPARSE_MLA_M1_SPLITMERGE_FINAL_LIB but "
            "_C::sparse_mla_m1_splitmerge_final_cuda CUDA kernel is unavailable"
        )


def _can_use_m1_splitmerge_final(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    out: torch.Tensor,
    num_tokens: int,
    num_heads_q: int,
    index_topk: int,
    num_kv_splits: int,
    assume_valid_indices: bool,
    return_lse: bool,
    convert_req_to_global: bool,
) -> bool:
    if not _M1_SPLITMERGE_FINAL_ENABLED:
        return False
    if _M1_SPLITMERGE_FINAL_NUM_SPLITS not in (4, 8, 16, 32):
        raise RuntimeError(
            "VLLM_SPARSE_MLA_M1_SPLITMERGE_FINAL_NUM_SPLITS must be "
            "4, 8, 16, or 32"
        )
    return (
        num_kv_splits == 1
        and num_tokens == 1
        and num_heads_q in (4, 8, 16)
        and index_topk == 2048
        and kv.shape[1] == 1
        and kv.shape[2] == _DIM_QK
        and kv.shape[0] >= index_topk
        and assume_valid_indices
        and not return_lse
        and not convert_req_to_global
        and q_nope.dtype == torch.bfloat16
        and q_pe.dtype == torch.bfloat16
        and kv.dtype == torch.bfloat16
        and indices.dtype == torch.int32
        and out.dtype == torch.bfloat16
        and q_nope.device.type == "cuda"
        and q_pe.device == q_nope.device
        and kv.device == q_nope.device
        and indices.device == q_nope.device
        and out.device == q_nope.device
        and q_nope.stride(2) == 1
        and q_pe.stride(2) == 1
        and kv.stride(2) == 1
        and indices.stride(2) == 1
        and out.stride(2) == 1
    )


def _log_m1_coop_final_dispatch(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    out: torch.Tensor,
    num_heads_q: int,
    index_topk: int,
    convert_req_to_global: bool,
) -> None:
    global _M1_COOP_FINAL_DEBUG_LOGS_REMAINING
    if _M1_COOP_FINAL_DEBUG_LOGS_REMAINING <= 0:
        return
    _M1_COOP_FINAL_DEBUG_LOGS_REMAINING -= 1
    logger.warning(
        "Stage50 sparse_mla_m1_coop_final dispatch: q_nope=%s stride=%s "
        "q_pe=%s stride=%s kv=%s stride=%s indices=%s stride=%s out=%s "
        "stride=%s heads=%s topk=%s convert_req_to_global=%s",
        tuple(q_nope.shape),
        q_nope.stride(),
        tuple(q_pe.shape),
        q_pe.stride(),
        tuple(kv.shape),
        kv.stride(),
        tuple(indices.shape),
        indices.stride(),
        tuple(out.shape),
        out.stride(),
        num_heads_q,
        index_topk,
        convert_req_to_global,
    )


def _has_m1_coop_final_op() -> bool:
    try:
        return bool(
            torch._C._dispatch_has_kernel_for_dispatch_key(
                "_C::sparse_mla_m1_coop_final_cuda", "CUDA"
            )
        )
    except RuntimeError:
        return False


@functools.lru_cache(maxsize=1)
def _load_m1_coop_final_op() -> None:
    if _has_m1_coop_final_op():
        return
    if not _M1_COOP_FINAL_LIB_PATH:
        raise RuntimeError(
            "VLLM_SPARSE_MLA_M1_COOP_FINAL=1 requires "
            "VLLM_SPARSE_MLA_M1_COOP_FINAL_LIB"
        )
    if not os.path.exists(_M1_COOP_FINAL_LIB_PATH):
        raise RuntimeError(
            "VLLM_SPARSE_MLA_M1_COOP_FINAL_LIB does not exist: "
            f"{_M1_COOP_FINAL_LIB_PATH}"
        )
    torch.ops.load_library(_M1_COOP_FINAL_LIB_PATH)
    if not _has_m1_coop_final_op():
        raise RuntimeError(
            "loaded VLLM_SPARSE_MLA_M1_COOP_FINAL_LIB but "
            "_C::sparse_mla_m1_coop_final_cuda CUDA kernel is unavailable"
        )


def _can_use_m1_coop_final(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    out: torch.Tensor,
    num_tokens: int,
    num_heads_q: int,
    index_topk: int,
    num_kv_splits: int,
    assume_valid_indices: bool,
    return_lse: bool,
    convert_req_to_global: bool,
) -> bool:
    if not _M1_COOP_FINAL_ENABLED:
        return False
    if _M1_COOP_FINAL_NUM_SPLITS not in (4, 8, 16, 32):
        raise RuntimeError(
            "VLLM_SPARSE_MLA_M1_COOP_FINAL_NUM_SPLITS must be 4, 8, 16, or 32"
        )
    return (
        num_kv_splits == 1
        and num_tokens == 1
        and num_heads_q in (4, 8, 16)
        and index_topk == 2048
        and kv.shape[1] == 1
        and kv.shape[2] == _DIM_QK
        and kv.shape[0] >= index_topk
        and assume_valid_indices
        and not return_lse
        and not convert_req_to_global
        and q_nope.dtype == torch.bfloat16
        and q_pe.dtype == torch.bfloat16
        and kv.dtype == torch.bfloat16
        and indices.dtype == torch.int32
        and out.dtype == torch.bfloat16
        and q_nope.device.type == "cuda"
        and q_pe.device == q_nope.device
        and kv.device == q_nope.device
        and indices.device == q_nope.device
        and out.device == q_nope.device
        and q_nope.stride(2) == 1
        and q_pe.stride(2) == 1
        and kv.stride(2) == 1
        and indices.stride(2) == 1
        and out.stride(2) == 1
    )


def _parse_final_static_config(env_name: str, default: str) -> tuple[int, int, int]:
    config = os.getenv(env_name, default)
    block_n, num_warps, num_stages = (int(part) for part in config.split(","))
    return block_n, num_warps, num_stages


def _choose_dynamic_final_static_config(num_tokens: int) -> tuple[int, int, int]:
    small_m = int(os.getenv("VLLM_SPARSE_MLA_FINAL_DYNAMIC_SMALL_M", "1024"))
    large_m = int(os.getenv("VLLM_SPARSE_MLA_FINAL_DYNAMIC_LARGE_M", "4096"))
    if num_tokens < small_m:
        return _parse_final_static_config(
            "VLLM_SPARSE_MLA_FINAL_DYNAMIC_SMALL_CONFIG", "16,2,4"
        )
    if num_tokens >= large_m:
        return _parse_final_static_config(
            "VLLM_SPARSE_MLA_FINAL_DYNAMIC_LARGE_CONFIG", "16,2,2"
        )
    return _parse_final_static_config(
        "VLLM_SPARSE_MLA_FINAL_DYNAMIC_MID_CONFIG", "16,2,1"
    )


@triton.jit
def _sparse_mla_compute_tile(
    q_nope_buffer,
    q_pe_buffer,
    k_buffer,  # V is the first BLOCK_DV lanes of each row of k_buffer.
    indices_ptr,
    req_id_ptr,
    block_table_ptr,
    cur_q,
    cur_head,
    cur_kv_head_id,
    mask_h,
    split_start,
    split_end,
    seq_kv,
    stride_q_nope_token,
    stride_q_nope_head,
    stride_q_pe_token,
    stride_q_pe_head,
    stride_kv_token,
    stride_kv_head,
    stride_indices_token,
    stride_indices_head,
    stride_block_table_req,
    stride_block_table_block,
    sm_scale,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DPE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    MAX_NUM_BLOCKS_PER_REQ: tl.constexpr,
    REUSE_K_AS_V: tl.constexpr,
    FULL_BLOCK_H: tl.constexpr,
    CONVERT_REQ_TO_GLOBAL: tl.constexpr,
    ASSUME_VALID_INDICES: tl.constexpr,
    ASSUME_VALID_NOMASK: tl.constexpr,
    ASSUME_VALID_AFTER_TOPK: tl.constexpr,
    ASSUME_VALID_AFTER_TOPK_NOMASK: tl.constexpr,
    ASSUME_PREFIX_LEN_MASK: tl.constexpr,
    VALID_INDEX_BASE_SEQ_LEN,
    INDEX_TOPK: tl.constexpr,
):
    """Shared stage-1 body: load Q, run the sparse online-softmax loop over
    `[split_start, split_end)` of the topk axis, return accumulators."""
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dpe = tl.arange(0, BLOCK_DPE)
    offs_dpe_k = BLOCK_DMODEL + offs_dpe
    offs_dv = tl.arange(0, BLOCK_DV)

    if FULL_BLOCK_H:
        q = tl.load(
            q_nope_buffer
            + cur_q * stride_q_nope_token
            + cur_head[:, None] * stride_q_nope_head
            + offs_d[None, :],
        )
        qpe = tl.load(
            q_pe_buffer
            + cur_q * stride_q_pe_token
            + cur_head[:, None] * stride_q_pe_head
            + offs_dpe[None, :],
        )
    else:
        q = tl.load(
            q_nope_buffer
            + cur_q * stride_q_nope_token
            + cur_head[:, None] * stride_q_nope_head
            + offs_d[None, :],
            mask=mask_h[:, None],
            other=0.0,
        )
        qpe = tl.load(
            q_pe_buffer
            + cur_q * stride_q_pe_token
            + cur_head[:, None] * stride_q_pe_head
            + offs_dpe[None, :],
            mask=mask_h[:, None],
            other=0.0,
        )

    # Large negative but finite sentinel for masked-out positions. `-inf`
    # would give `-inf - -inf = NaN` when a whole BLOCK_N tile is masked
    # (common in short prefill where most topk slots are -1); a finite value
    # keeps `sentinel - sentinel = 0` and `exp2(0) = 1`, and the
    # corresponding v slots are already loaded as 0.
    NEG_LARGE = -1.0e30
    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) + NEG_LARGE
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, BLOCK_DV], dtype=tl.float32)

    for start_indice in range(split_start, split_end, BLOCK_N):
        offs_indice = start_indice + tl.arange(0, BLOCK_N)
        mask_indice = offs_indice < split_end
        convert_valid = mask_indice
        prefix_mask_kv = (
            mask_indice
            & (offs_indice < (VALID_INDEX_BASE_SEQ_LEN + cur_q + 1))
        )
        if ASSUME_VALID_NOMASK or (
            ASSUME_VALID_AFTER_TOPK_NOMASK
            and (VALID_INDEX_BASE_SEQ_LEN + cur_q + 1 >= INDEX_TOPK)
        ):
            indices = tl.load(
                indices_ptr
                + cur_q * stride_indices_token
                + cur_kv_head_id * stride_indices_head
                + offs_indice,
            )
            if CONVERT_REQ_TO_GLOBAL:
                req_id = tl.load(req_id_ptr + cur_q)
                block_id = indices // BLOCK_SIZE
                inblock_off = indices - block_id * BLOCK_SIZE
                convert_valid = (indices >= 0) & (
                    block_id < MAX_NUM_BLOCKS_PER_REQ
                )
                block_idx = tl.load(
                    block_table_ptr
                    + req_id * stride_block_table_req
                    + block_id * stride_block_table_block,
                    mask=convert_valid,
                    other=0,
                )
                indices = tl.where(
                    convert_valid,
                    block_idx * BLOCK_SIZE + inblock_off,
                    0,
                )
        elif ASSUME_PREFIX_LEN_MASK:
            indices = tl.load(
                indices_ptr
                + cur_q * stride_indices_token
                + cur_kv_head_id * stride_indices_head
                + offs_indice,
                mask=prefix_mask_kv,
                other=0,
            )
            if CONVERT_REQ_TO_GLOBAL:
                req_id = tl.load(req_id_ptr + cur_q)
                block_id = indices // BLOCK_SIZE
                inblock_off = indices - block_id * BLOCK_SIZE
                valid_block = (block_id >= 0) & (
                    block_id < MAX_NUM_BLOCKS_PER_REQ
                )
                convert_valid = prefix_mask_kv & valid_block
                block_idx = tl.load(
                    block_table_ptr
                    + req_id * stride_block_table_req
                    + block_id * stride_block_table_block,
                    mask=convert_valid,
                    other=0,
                )
                indices = tl.where(
                    convert_valid,
                    block_idx * BLOCK_SIZE + inblock_off,
                    0,
                )
        else:
            indices = tl.load(
                indices_ptr
                + cur_q * stride_indices_token
                + cur_kv_head_id * stride_indices_head
                + offs_indice,
                mask=mask_indice,
                other=-1,
            )
            if CONVERT_REQ_TO_GLOBAL:
                req_id = tl.load(req_id_ptr + cur_q)
                block_id = indices // BLOCK_SIZE
                inblock_off = indices - block_id * BLOCK_SIZE
                valid_block = (block_id >= 0) & (
                    block_id < MAX_NUM_BLOCKS_PER_REQ
                )
                convert_valid = mask_indice & valid_block
                block_idx = tl.load(
                    block_table_ptr
                    + req_id * stride_block_table_req
                    + block_id * stride_block_table_block,
                    mask=convert_valid,
                    other=0,
                )
                indices = tl.where(
                    convert_valid,
                    block_idx * BLOCK_SIZE + inblock_off,
                    0,
                )

        if ASSUME_VALID_NOMASK or (
            ASSUME_VALID_AFTER_TOPK_NOMASK
            and (VALID_INDEX_BASE_SEQ_LEN + cur_q + 1 >= INDEX_TOPK)
        ):
            if CONVERT_REQ_TO_GLOBAL:
                mask_kv = convert_valid
            else:
                mask_kv = mask_indice
        elif ASSUME_VALID_INDICES:
            mask_kv = mask_indice
        elif ASSUME_VALID_AFTER_TOPK and (
            VALID_INDEX_BASE_SEQ_LEN + cur_q + 1 >= INDEX_TOPK
        ):
            mask_kv = mask_indice
        elif ASSUME_PREFIX_LEN_MASK:
            mask_kv = prefix_mask_kv
        else:
            mask_kv = (indices >= 0) & (indices < seq_kv)

        offs_k = (
            indices[None, :] * stride_kv_token
            + cur_kv_head_id * stride_kv_head
            + offs_d[:, None]
        )
        if ASSUME_VALID_NOMASK or (
            ASSUME_VALID_AFTER_TOPK_NOMASK
            and (VALID_INDEX_BASE_SEQ_LEN + cur_q + 1 >= INDEX_TOPK)
        ):
            if CONVERT_REQ_TO_GLOBAL:
                k = tl.load(k_buffer + offs_k, mask=mask_kv[None, :], other=0.0)
            else:
                k = tl.load(k_buffer + offs_k)
        else:
            k = tl.load(
                k_buffer + offs_k,
                mask=mask_kv[None, :],
                other=0.0,
            )
        qk = tl.dot(q, k.to(q.dtype))

        offs_kpe = (
            indices[None, :] * stride_kv_token
            + cur_kv_head_id * stride_kv_head
            + offs_dpe_k[:, None]
        )
        if ASSUME_VALID_NOMASK or (
            ASSUME_VALID_AFTER_TOPK_NOMASK
            and (VALID_INDEX_BASE_SEQ_LEN + cur_q + 1 >= INDEX_TOPK)
        ):
            if CONVERT_REQ_TO_GLOBAL:
                kpe = tl.load(
                    k_buffer + offs_kpe, mask=mask_kv[None, :], other=0.0
                )
            else:
                kpe = tl.load(k_buffer + offs_kpe)
        else:
            kpe = tl.load(
                k_buffer + offs_kpe,
                mask=mask_kv[None, :],
                other=0.0,
            )
        qk += tl.dot(qpe, kpe.to(q.dtype))

        qk *= sm_scale
        if ASSUME_VALID_NOMASK or (
            ASSUME_VALID_AFTER_TOPK_NOMASK
            and (VALID_INDEX_BASE_SEQ_LEN + cur_q + 1 >= INDEX_TOPK)
        ):
            if CONVERT_REQ_TO_GLOBAL:
                if FULL_BLOCK_H:
                    qk = tl.where(mask_kv[None, :], qk, NEG_LARGE)
                else:
                    qk = tl.where(
                        (mask_h[:, None]) & (mask_kv[None, :]), qk, NEG_LARGE
                    )
            elif not FULL_BLOCK_H:
                qk = tl.where(mask_h[:, None], qk, NEG_LARGE)
        else:
            if FULL_BLOCK_H:
                qk = tl.where(mask_kv[None, :], qk, NEG_LARGE)
            else:
                qk = tl.where(
                    (mask_h[:, None]) & (mask_kv[None, :]), qk, NEG_LARGE
                )

        if REUSE_K_AS_V:
            v = tl.trans(k)
        else:
            offs_v = (
                indices[:, None] * stride_kv_token
                + cur_kv_head_id * stride_kv_head
                + offs_dv[None, :]
            )
            if ASSUME_VALID_NOMASK or (
                ASSUME_VALID_AFTER_TOPK_NOMASK
                and (VALID_INDEX_BASE_SEQ_LEN + cur_q + 1 >= INDEX_TOPK)
            ):
                if CONVERT_REQ_TO_GLOBAL:
                    v = tl.load(k_buffer + offs_v, mask=mask_kv[:, None], other=0.0)
                else:
                    v = tl.load(k_buffer + offs_v)
            else:
                v = tl.load(k_buffer + offs_v, mask=mask_kv[:, None], other=0.0)

        n_e_max = tl.maximum(tl.max(qk, 1), e_max)
        re_scale = tl.exp2(e_max - n_e_max)
        p = tl.exp2(qk - n_e_max[:, None])
        acc *= re_scale[:, None]
        acc += tl.dot(p.to(v.dtype), v)
        e_sum = e_sum * re_scale + tl.sum(p, 1)
        e_max = n_e_max

    return acc, e_max, e_sum


@triton.autotune(
    configs=_FINAL_AUTOTUNE_CONFIGS,
    key=["num_tokens", "index_topk", "kv_group_num"],
)
@triton.jit
def _sparse_mla_kernel_final(
    q_nope_buffer,
    q_pe_buffer,
    k_buffer,
    indices_ptr,
    req_id_ptr,
    block_table_ptr,
    out_ptr,
    lse_ptr,
    seq_kv,
    h_q,
    num_tokens: tl.constexpr,
    stride_q_nope_token,
    stride_q_nope_head,
    stride_q_pe_token,
    stride_q_pe_head,
    stride_kv_token,
    stride_kv_head,
    stride_out_token,
    stride_out_head,
    stride_lse_token,
    stride_indices_token,
    stride_indices_head,
    stride_block_table_req,
    stride_block_table_block,
    sm_scale,
    index_topk: tl.constexpr,
    kv_group_num: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DPE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    MAX_NUM_BLOCKS_PER_REQ: tl.constexpr,
    LOGE2: tl.constexpr,
    REUSE_K_AS_V: tl.constexpr,
    FULL_BLOCK_H: tl.constexpr,
    RETURN_LSE: tl.constexpr,
    CONVERT_REQ_TO_GLOBAL: tl.constexpr,
    ASSUME_VALID_INDICES: tl.constexpr,
    ASSUME_VALID_NOMASK: tl.constexpr,
    ASSUME_VALID_AFTER_TOPK: tl.constexpr,
    ASSUME_VALID_AFTER_TOPK_NOMASK: tl.constexpr,
    ASSUME_PREFIX_LEN_MASK: tl.constexpr,
    VALID_INDEX_BASE_SEQ_LEN,
):
    """Single-pass fast path: full topk, write final bf16 output directly."""
    cur_q = tl.program_id(0)
    cur_head_id = tl.program_id(1)
    cur_kv_head_id = cur_head_id // tl.cdiv(kv_group_num, BLOCK_H)

    VALID_BLOCK_H: tl.constexpr = BLOCK_H if kv_group_num > BLOCK_H else kv_group_num
    cur_head = cur_head_id * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = (cur_head < (cur_head_id + 1) * VALID_BLOCK_H) & (cur_head < h_q)

    acc, e_max, e_sum = _sparse_mla_compute_tile(
        q_nope_buffer,
        q_pe_buffer,
        k_buffer,
        indices_ptr,
        req_id_ptr,
        block_table_ptr,
        cur_q,
        cur_head,
        cur_kv_head_id,
        mask_h,
        0,
        index_topk,
        seq_kv,
        stride_q_nope_token,
        stride_q_nope_head,
        stride_q_pe_token,
        stride_q_pe_head,
        stride_kv_token,
        stride_kv_head,
        stride_indices_token,
        stride_indices_head,
        stride_block_table_req,
        stride_block_table_block,
        sm_scale,
        BLOCK_H,
        BLOCK_N,
        BLOCK_DV,
        BLOCK_DMODEL,
        BLOCK_DPE,
        BLOCK_SIZE,
        MAX_NUM_BLOCKS_PER_REQ,
        REUSE_K_AS_V,
        FULL_BLOCK_H,
        CONVERT_REQ_TO_GLOBAL,
        ASSUME_VALID_INDICES,
        ASSUME_VALID_NOMASK,
        ASSUME_VALID_AFTER_TOPK,
        ASSUME_VALID_AFTER_TOPK_NOMASK,
        ASSUME_PREFIX_LEN_MASK,
        VALID_INDEX_BASE_SEQ_LEN,
        index_topk,
    )

    # Guard against queries with zero valid KV (e_sum == 0 → NaN from 0/0).
    e_sum_safe = tl.where(e_sum > 0, e_sum, 1.0)
    offs_dv = tl.arange(0, BLOCK_DV)
    if FULL_BLOCK_H:
        tl.store(
            out_ptr
            + cur_q * stride_out_token
            + cur_head[:, None] * stride_out_head
            + offs_dv[None, :],
            (acc / e_sum_safe[:, None]).to(tl.bfloat16),
        )
    else:
        tl.store(
            out_ptr
            + cur_q * stride_out_token
            + cur_head[:, None] * stride_out_head
            + offs_dv[None, :],
            (acc / e_sum_safe[:, None]).to(tl.bfloat16),
            mask=mask_h[:, None],
        )
    if RETURN_LSE:
        lse = (e_max + tl.log2(e_sum)) * LOGE2
        if FULL_BLOCK_H:
            tl.store(lse_ptr + cur_q * stride_lse_token + cur_head, lse)
        else:
            tl.store(
                lse_ptr + cur_q * stride_lse_token + cur_head,
                lse,
                mask=mask_h,
            )


@triton.jit
def _sparse_mla_kernel_final_static(
    q_nope_buffer,
    q_pe_buffer,
    k_buffer,
    indices_ptr,
    req_id_ptr,
    block_table_ptr,
    out_ptr,
    lse_ptr,
    seq_kv,
    h_q,
    stride_q_nope_token,
    stride_q_nope_head,
    stride_q_pe_token,
    stride_q_pe_head,
    stride_kv_token,
    stride_kv_head,
    stride_out_token,
    stride_out_head,
    stride_lse_token,
    stride_indices_token,
    stride_indices_head,
    stride_block_table_req,
    stride_block_table_block,
    sm_scale,
    index_topk: tl.constexpr,
    kv_group_num: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DPE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    MAX_NUM_BLOCKS_PER_REQ: tl.constexpr,
    LOGE2: tl.constexpr,
    REUSE_K_AS_V: tl.constexpr,
    FULL_BLOCK_H: tl.constexpr,
    RETURN_LSE: tl.constexpr,
    CONVERT_REQ_TO_GLOBAL: tl.constexpr,
    ASSUME_VALID_INDICES: tl.constexpr,
    ASSUME_VALID_NOMASK: tl.constexpr,
    ASSUME_VALID_AFTER_TOPK: tl.constexpr,
    ASSUME_VALID_AFTER_TOPK_NOMASK: tl.constexpr,
    ASSUME_PREFIX_LEN_MASK: tl.constexpr,
    VALID_INDEX_BASE_SEQ_LEN,
):
    """Single-pass final kernel with caller-selected static launch config."""
    cur_q = tl.program_id(0)
    cur_head_id = tl.program_id(1)
    cur_kv_head_id = cur_head_id // tl.cdiv(kv_group_num, BLOCK_H)

    VALID_BLOCK_H: tl.constexpr = BLOCK_H if kv_group_num > BLOCK_H else kv_group_num
    cur_head = cur_head_id * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = (cur_head < (cur_head_id + 1) * VALID_BLOCK_H) & (cur_head < h_q)

    acc, e_max, e_sum = _sparse_mla_compute_tile(
        q_nope_buffer,
        q_pe_buffer,
        k_buffer,
        indices_ptr,
        req_id_ptr,
        block_table_ptr,
        cur_q,
        cur_head,
        cur_kv_head_id,
        mask_h,
        0,
        index_topk,
        seq_kv,
        stride_q_nope_token,
        stride_q_nope_head,
        stride_q_pe_token,
        stride_q_pe_head,
        stride_kv_token,
        stride_kv_head,
        stride_indices_token,
        stride_indices_head,
        stride_block_table_req,
        stride_block_table_block,
        sm_scale,
        BLOCK_H,
        BLOCK_N,
        BLOCK_DV,
        BLOCK_DMODEL,
        BLOCK_DPE,
        BLOCK_SIZE,
        MAX_NUM_BLOCKS_PER_REQ,
        REUSE_K_AS_V,
        FULL_BLOCK_H,
        CONVERT_REQ_TO_GLOBAL,
        ASSUME_VALID_INDICES,
        ASSUME_VALID_NOMASK,
        ASSUME_VALID_AFTER_TOPK,
        ASSUME_VALID_AFTER_TOPK_NOMASK,
        ASSUME_PREFIX_LEN_MASK,
        VALID_INDEX_BASE_SEQ_LEN,
        index_topk,
    )

    e_sum_safe = tl.where(e_sum > 0, e_sum, 1.0)
    offs_dv = tl.arange(0, BLOCK_DV)
    if FULL_BLOCK_H:
        tl.store(
            out_ptr
            + cur_q * stride_out_token
            + cur_head[:, None] * stride_out_head
            + offs_dv[None, :],
            (acc / e_sum_safe[:, None]).to(tl.bfloat16),
        )
    else:
        tl.store(
            out_ptr
            + cur_q * stride_out_token
            + cur_head[:, None] * stride_out_head
            + offs_dv[None, :],
            (acc / e_sum_safe[:, None]).to(tl.bfloat16),
            mask=mask_h[:, None],
        )
    if RETURN_LSE:
        lse = (e_max + tl.log2(e_sum)) * LOGE2
        if FULL_BLOCK_H:
            tl.store(lse_ptr + cur_q * stride_lse_token + cur_head, lse)
        else:
            tl.store(
                lse_ptr + cur_q * stride_lse_token + cur_head,
                lse,
                mask=mask_h,
            )


@triton.jit
def _sparse_mla_kernel_final_dv_tile_static(
    q_nope_buffer,
    q_pe_buffer,
    k_buffer,
    indices_ptr,
    req_id_ptr,
    block_table_ptr,
    out_ptr,
    lse_ptr,
    seq_kv,
    h_q,
    stride_q_nope_token,
    stride_q_nope_head,
    stride_q_pe_token,
    stride_q_pe_head,
    stride_kv_token,
    stride_kv_head,
    stride_out_token,
    stride_out_head,
    stride_lse_token,
    stride_indices_token,
    stride_indices_head,
    stride_block_table_req,
    stride_block_table_block,
    sm_scale,
    index_topk: tl.constexpr,
    kv_group_num: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_DV_TILE: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DPE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    MAX_NUM_BLOCKS_PER_REQ: tl.constexpr,
    LOGE2: tl.constexpr,
    FULL_BLOCK_H: tl.constexpr,
    RETURN_LSE: tl.constexpr,
    CONVERT_REQ_TO_GLOBAL: tl.constexpr,
    ASSUME_VALID_INDICES: tl.constexpr,
    ASSUME_VALID_NOMASK: tl.constexpr,
    ASSUME_VALID_AFTER_TOPK: tl.constexpr,
    ASSUME_VALID_AFTER_TOPK_NOMASK: tl.constexpr,
    ASSUME_PREFIX_LEN_MASK: tl.constexpr,
    VALID_INDEX_BASE_SEQ_LEN,
):
    """Decode-M=1 final kernel split across output-DV tiles."""
    cur_q = tl.program_id(0)
    cur_head_id = tl.program_id(1)
    cur_dv_tile = tl.program_id(2)
    cur_kv_head_id = cur_head_id // tl.cdiv(kv_group_num, BLOCK_H)

    VALID_BLOCK_H: tl.constexpr = BLOCK_H if kv_group_num > BLOCK_H else kv_group_num
    cur_head = cur_head_id * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = (cur_head < (cur_head_id + 1) * VALID_BLOCK_H) & (cur_head < h_q)

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dpe = tl.arange(0, BLOCK_DPE)
    offs_dpe_k = BLOCK_DMODEL + offs_dpe
    offs_dv = cur_dv_tile * BLOCK_DV_TILE + tl.arange(0, BLOCK_DV_TILE)
    mask_dv = offs_dv < BLOCK_DV

    if FULL_BLOCK_H:
        q = tl.load(
            q_nope_buffer
            + cur_q * stride_q_nope_token
            + cur_head[:, None] * stride_q_nope_head
            + offs_d[None, :],
        )
        qpe = tl.load(
            q_pe_buffer
            + cur_q * stride_q_pe_token
            + cur_head[:, None] * stride_q_pe_head
            + offs_dpe[None, :],
        )
    else:
        q = tl.load(
            q_nope_buffer
            + cur_q * stride_q_nope_token
            + cur_head[:, None] * stride_q_nope_head
            + offs_d[None, :],
            mask=mask_h[:, None],
            other=0.0,
        )
        qpe = tl.load(
            q_pe_buffer
            + cur_q * stride_q_pe_token
            + cur_head[:, None] * stride_q_pe_head
            + offs_dpe[None, :],
            mask=mask_h[:, None],
            other=0.0,
        )

    NEG_LARGE = -1.0e30
    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) + NEG_LARGE
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, BLOCK_DV_TILE], dtype=tl.float32)

    for start_indice in range(0, index_topk, BLOCK_N):
        offs_indice = start_indice + tl.arange(0, BLOCK_N)
        mask_indice = offs_indice < index_topk
        convert_valid = mask_indice
        prefix_mask_kv = (
            mask_indice
            & (offs_indice < (VALID_INDEX_BASE_SEQ_LEN + cur_q + 1))
        )
        if ASSUME_VALID_NOMASK or (
            ASSUME_VALID_AFTER_TOPK_NOMASK
            and (VALID_INDEX_BASE_SEQ_LEN + cur_q + 1 >= index_topk)
        ):
            indices = tl.load(
                indices_ptr
                + cur_q * stride_indices_token
                + cur_kv_head_id * stride_indices_head
                + offs_indice,
            )
            if CONVERT_REQ_TO_GLOBAL:
                req_id = tl.load(req_id_ptr + cur_q)
                block_id = indices // BLOCK_SIZE
                inblock_off = indices - block_id * BLOCK_SIZE
                convert_valid = (indices >= 0) & (
                    block_id < MAX_NUM_BLOCKS_PER_REQ
                )
                block_idx = tl.load(
                    block_table_ptr
                    + req_id * stride_block_table_req
                    + block_id * stride_block_table_block,
                    mask=convert_valid,
                    other=0,
                )
                indices = tl.where(
                    convert_valid,
                    block_idx * BLOCK_SIZE + inblock_off,
                    0,
                )
        elif ASSUME_PREFIX_LEN_MASK:
            indices = tl.load(
                indices_ptr
                + cur_q * stride_indices_token
                + cur_kv_head_id * stride_indices_head
                + offs_indice,
                mask=prefix_mask_kv,
                other=0,
            )
            if CONVERT_REQ_TO_GLOBAL:
                req_id = tl.load(req_id_ptr + cur_q)
                block_id = indices // BLOCK_SIZE
                inblock_off = indices - block_id * BLOCK_SIZE
                valid_block = (block_id >= 0) & (
                    block_id < MAX_NUM_BLOCKS_PER_REQ
                )
                convert_valid = prefix_mask_kv & valid_block
                block_idx = tl.load(
                    block_table_ptr
                    + req_id * stride_block_table_req
                    + block_id * stride_block_table_block,
                    mask=convert_valid,
                    other=0,
                )
                indices = tl.where(
                    convert_valid,
                    block_idx * BLOCK_SIZE + inblock_off,
                    0,
                )
        else:
            indices = tl.load(
                indices_ptr
                + cur_q * stride_indices_token
                + cur_kv_head_id * stride_indices_head
                + offs_indice,
                mask=mask_indice,
                other=-1,
            )
            if CONVERT_REQ_TO_GLOBAL:
                req_id = tl.load(req_id_ptr + cur_q)
                block_id = indices // BLOCK_SIZE
                inblock_off = indices - block_id * BLOCK_SIZE
                valid_block = (block_id >= 0) & (
                    block_id < MAX_NUM_BLOCKS_PER_REQ
                )
                convert_valid = mask_indice & valid_block
                block_idx = tl.load(
                    block_table_ptr
                    + req_id * stride_block_table_req
                    + block_id * stride_block_table_block,
                    mask=convert_valid,
                    other=0,
                )
                indices = tl.where(
                    convert_valid,
                    block_idx * BLOCK_SIZE + inblock_off,
                    0,
                )

        if ASSUME_VALID_NOMASK or (
            ASSUME_VALID_AFTER_TOPK_NOMASK
            and (VALID_INDEX_BASE_SEQ_LEN + cur_q + 1 >= index_topk)
        ):
            if CONVERT_REQ_TO_GLOBAL:
                mask_kv = convert_valid
            else:
                mask_kv = mask_indice
        elif ASSUME_VALID_INDICES:
            mask_kv = mask_indice
        elif ASSUME_VALID_AFTER_TOPK and (
            VALID_INDEX_BASE_SEQ_LEN + cur_q + 1 >= index_topk
        ):
            mask_kv = mask_indice
        elif ASSUME_PREFIX_LEN_MASK:
            mask_kv = prefix_mask_kv
        else:
            mask_kv = (indices >= 0) & (indices < seq_kv)

        offs_k = (
            indices[None, :] * stride_kv_token
            + cur_kv_head_id * stride_kv_head
            + offs_d[:, None]
        )
        if ASSUME_VALID_NOMASK or (
            ASSUME_VALID_AFTER_TOPK_NOMASK
            and (VALID_INDEX_BASE_SEQ_LEN + cur_q + 1 >= index_topk)
        ):
            if CONVERT_REQ_TO_GLOBAL:
                k = tl.load(k_buffer + offs_k, mask=mask_kv[None, :], other=0.0)
            else:
                k = tl.load(k_buffer + offs_k)
        else:
            k = tl.load(
                k_buffer + offs_k,
                mask=mask_kv[None, :],
                other=0.0,
            )
        qk = tl.dot(q, k.to(q.dtype))

        offs_kpe = (
            indices[None, :] * stride_kv_token
            + cur_kv_head_id * stride_kv_head
            + offs_dpe_k[:, None]
        )
        if ASSUME_VALID_NOMASK or (
            ASSUME_VALID_AFTER_TOPK_NOMASK
            and (VALID_INDEX_BASE_SEQ_LEN + cur_q + 1 >= index_topk)
        ):
            if CONVERT_REQ_TO_GLOBAL:
                kpe = tl.load(
                    k_buffer + offs_kpe, mask=mask_kv[None, :], other=0.0
                )
            else:
                kpe = tl.load(k_buffer + offs_kpe)
        else:
            kpe = tl.load(
                k_buffer + offs_kpe,
                mask=mask_kv[None, :],
                other=0.0,
            )
        qk += tl.dot(qpe, kpe.to(q.dtype))

        qk *= sm_scale
        if ASSUME_VALID_NOMASK or (
            ASSUME_VALID_AFTER_TOPK_NOMASK
            and (VALID_INDEX_BASE_SEQ_LEN + cur_q + 1 >= index_topk)
        ):
            if CONVERT_REQ_TO_GLOBAL:
                if FULL_BLOCK_H:
                    qk = tl.where(mask_kv[None, :], qk, NEG_LARGE)
                else:
                    qk = tl.where(
                        (mask_h[:, None]) & (mask_kv[None, :]), qk, NEG_LARGE
                    )
            elif not FULL_BLOCK_H:
                qk = tl.where(mask_h[:, None], qk, NEG_LARGE)
        else:
            if FULL_BLOCK_H:
                qk = tl.where(mask_kv[None, :], qk, NEG_LARGE)
            else:
                qk = tl.where(
                    (mask_h[:, None]) & (mask_kv[None, :]), qk, NEG_LARGE
                )

        offs_v = (
            indices[:, None] * stride_kv_token
            + cur_kv_head_id * stride_kv_head
            + offs_dv[None, :]
        )
        if ASSUME_VALID_NOMASK or (
            ASSUME_VALID_AFTER_TOPK_NOMASK
            and (VALID_INDEX_BASE_SEQ_LEN + cur_q + 1 >= index_topk)
        ):
            if CONVERT_REQ_TO_GLOBAL:
                v = tl.load(
                    k_buffer + offs_v,
                    mask=mask_kv[:, None] & mask_dv[None, :],
                    other=0.0,
                )
            else:
                v = tl.load(k_buffer + offs_v, mask=mask_dv[None, :], other=0.0)
        else:
            v = tl.load(
                k_buffer + offs_v,
                mask=mask_kv[:, None] & mask_dv[None, :],
                other=0.0,
            )

        n_e_max = tl.maximum(tl.max(qk, 1), e_max)
        re_scale = tl.exp2(e_max - n_e_max)
        p = tl.exp2(qk - n_e_max[:, None])
        acc *= re_scale[:, None]
        acc += tl.dot(p.to(v.dtype), v)
        e_sum = e_sum * re_scale + tl.sum(p, 1)
        e_max = n_e_max

    e_sum_safe = tl.where(e_sum > 0, e_sum, 1.0)
    if FULL_BLOCK_H:
        tl.store(
            out_ptr
            + cur_q * stride_out_token
            + cur_head[:, None] * stride_out_head
            + offs_dv[None, :],
            (acc / e_sum_safe[:, None]).to(tl.bfloat16),
            mask=mask_dv[None, :],
        )
    else:
        tl.store(
            out_ptr
            + cur_q * stride_out_token
            + cur_head[:, None] * stride_out_head
            + offs_dv[None, :],
            (acc / e_sum_safe[:, None]).to(tl.bfloat16),
            mask=mask_h[:, None] & mask_dv[None, :],
        )
    if RETURN_LSE and cur_dv_tile == 0:
        lse = (e_max + tl.log2(e_sum)) * LOGE2
        if FULL_BLOCK_H:
            tl.store(lse_ptr + cur_q * stride_lse_token + cur_head, lse)
        else:
            tl.store(
                lse_ptr + cur_q * stride_lse_token + cur_head,
                lse,
                mask=mask_h,
            )


@triton.autotune(
    configs=_SPLIT_AUTOTUNE_CONFIGS,
    key=["index_topk", "NUM_KV_SPLITS", "kv_group_num"],
)
@triton.jit
def _sparse_mla_kernel_split(
    q_nope_buffer,
    q_pe_buffer,
    k_buffer,
    indices_ptr,
    req_id_ptr,
    block_table_ptr,
    mid_out_ptr,
    seq_kv,
    h_q,
    stride_q_nope_token,
    stride_q_nope_head,
    stride_q_pe_token,
    stride_q_pe_head,
    stride_kv_token,
    stride_kv_head,
    stride_mid_token,
    stride_mid_head,
    stride_mid_split,
    stride_indices_token,
    stride_indices_head,
    stride_block_table_req,
    stride_block_table_block,
    sm_scale,
    index_topk: tl.constexpr,
    NUM_KV_SPLITS: tl.constexpr,
    kv_group_num: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DPE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    MAX_NUM_BLOCKS_PER_REQ: tl.constexpr,
    LOGE2: tl.constexpr,
    REUSE_K_AS_V: tl.constexpr,
    FULL_BLOCK_H: tl.constexpr,
    CONVERT_REQ_TO_GLOBAL: tl.constexpr,
    ASSUME_VALID_INDICES: tl.constexpr,
    ASSUME_VALID_NOMASK: tl.constexpr,
    ASSUME_VALID_AFTER_TOPK: tl.constexpr,
    ASSUME_VALID_AFTER_TOPK_NOMASK: tl.constexpr,
    ASSUME_PREFIX_LEN_MASK: tl.constexpr,
    VALID_INDEX_BASE_SEQ_LEN,
):
    """Stage 1 of split-KV: process one slice of the topk axis and write
    its `(out_partial, lse_partial)` into the mid buffer."""
    cur_q = tl.program_id(0)
    cur_head_id = tl.program_id(1)
    split_kv_id = tl.program_id(2)
    cur_kv_head_id = cur_head_id // tl.cdiv(kv_group_num, BLOCK_H)

    VALID_BLOCK_H: tl.constexpr = BLOCK_H if kv_group_num > BLOCK_H else kv_group_num
    cur_head = cur_head_id * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = (cur_head < (cur_head_id + 1) * VALID_BLOCK_H) & (cur_head < h_q)

    split_topk: tl.constexpr = tl.cdiv(index_topk, NUM_KV_SPLITS)
    split_start = split_kv_id * split_topk
    split_end = tl.minimum(split_start + split_topk, index_topk)

    acc, e_max, e_sum = _sparse_mla_compute_tile(
        q_nope_buffer,
        q_pe_buffer,
        k_buffer,
        indices_ptr,
        req_id_ptr,
        block_table_ptr,
        cur_q,
        cur_head,
        cur_kv_head_id,
        mask_h,
        split_start,
        split_end,
        seq_kv,
        stride_q_nope_token,
        stride_q_nope_head,
        stride_q_pe_token,
        stride_q_pe_head,
        stride_kv_token,
        stride_kv_head,
        stride_indices_token,
        stride_indices_head,
        stride_block_table_req,
        stride_block_table_block,
        sm_scale,
        BLOCK_H,
        BLOCK_N,
        BLOCK_DV,
        BLOCK_DMODEL,
        BLOCK_DPE,
        BLOCK_SIZE,
        MAX_NUM_BLOCKS_PER_REQ,
        REUSE_K_AS_V,
        FULL_BLOCK_H,
        CONVERT_REQ_TO_GLOBAL,
        ASSUME_VALID_INDICES,
        ASSUME_VALID_NOMASK,
        ASSUME_VALID_AFTER_TOPK,
        ASSUME_VALID_AFTER_TOPK_NOMASK,
        ASSUME_PREFIX_LEN_MASK,
        VALID_INDEX_BASE_SEQ_LEN,
        index_topk,
    )

    # Partial output and natural-log LSE for stage-2 merge.
    # When a split has no valid KV (`e_sum == 0`), guard the divide so the
    # mid buffer holds 0 instead of NaN; otherwise the `0 * NaN = NaN` term
    # in stage 2 would poison every other split.
    e_sum_safe = tl.where(e_sum > 0, e_sum, 1.0)
    offs_dv = tl.arange(0, BLOCK_DV)
    mid_base_2d = (
        mid_out_ptr
        + cur_q * stride_mid_token
        + cur_head[:, None] * stride_mid_head
        + split_kv_id * stride_mid_split
    )
    if FULL_BLOCK_H:
        tl.store(
            mid_base_2d + offs_dv[None, :],
            acc / e_sum_safe[:, None],
        )
    else:
        tl.store(
            mid_base_2d + offs_dv[None, :],
            acc / e_sum_safe[:, None],
            mask=mask_h[:, None],
        )
    mid_lse_ptr = (
        mid_out_ptr
        + cur_q * stride_mid_token
        + cur_head * stride_mid_head
        + split_kv_id * stride_mid_split
        + BLOCK_DV
    )
    if FULL_BLOCK_H:
        tl.store(mid_lse_ptr, (e_max + tl.log2(e_sum)) * LOGE2)
    else:
        tl.store(mid_lse_ptr, (e_max + tl.log2(e_sum)) * LOGE2, mask=mask_h)


@triton.jit
def _sparse_mla_merge_kernel(
    mid_out_ptr,
    out_ptr,
    lse_ptr,
    h_q,
    stride_mid_token,
    stride_mid_head,
    stride_mid_split,
    stride_out_token,
    stride_out_head,
    stride_lse_token,
    NUM_KV_SPLITS: tl.constexpr,
    kv_group_num: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_DV_TILE: tl.constexpr,
    RETURN_LSE: tl.constexpr,
):
    """Stage 2: N-way online-softmax merge of per-split `(out, lse)` tiles.

    Grid is `(num_tokens, num_head_groups, num_dv_tiles)`. Each program handles
    `BLOCK_H` heads × `BLOCK_DV_TILE` output-dim lanes. The LSE reduction is
    identical across DV tiles for the same (token, head) — each program
    recomputes it locally, which is cheap (O(NUM_KV_SPLITS) scalars) and
    avoids inter-CTA synchronization.
    """
    cur_q = tl.program_id(0)
    cur_head_id = tl.program_id(1)
    cur_dv_tile = tl.program_id(2)

    VALID_BLOCK_H: tl.constexpr = BLOCK_H if kv_group_num > BLOCK_H else kv_group_num
    cur_head = cur_head_id * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = (cur_head < (cur_head_id + 1) * VALID_BLOCK_H) & (cur_head < h_q)

    offs_dv = cur_dv_tile * BLOCK_DV_TILE + tl.arange(0, BLOCK_DV_TILE)
    mask_dv = offs_dv < BLOCK_DV
    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) - float("inf")
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, BLOCK_DV_TILE], dtype=tl.float32)

    mid_base_2d = (
        mid_out_ptr + cur_q * stride_mid_token + cur_head[:, None] * stride_mid_head
    )
    mid_lse_1d = (
        mid_out_ptr + cur_q * stride_mid_token + cur_head * stride_mid_head + BLOCK_DV
    )

    for split_kv_id in range(NUM_KV_SPLITS):
        tv = tl.load(
            mid_base_2d + split_kv_id * stride_mid_split + offs_dv[None, :],
            mask=mask_h[:, None] & mask_dv[None, :],
            other=0.0,
        )
        tlogic = tl.load(
            mid_lse_1d + split_kv_id * stride_mid_split,
            mask=mask_h,
            other=-float("inf"),
        )
        n_e_max = tl.maximum(tlogic, e_max)
        old_scale = tl.exp(e_max - n_e_max)
        exp_logic = tl.exp(tlogic - n_e_max)
        acc = acc * old_scale[:, None] + exp_logic[:, None] * tv
        e_sum = e_sum * old_scale + exp_logic
        e_max = n_e_max

    e_sum_safe = tl.where(e_sum > 0, e_sum, 1.0)
    tl.store(
        out_ptr
        + cur_q * stride_out_token
        + cur_head[:, None] * stride_out_head
        + offs_dv[None, :],
        (acc / e_sum_safe[:, None]).to(tl.bfloat16),
        mask=mask_h[:, None] & mask_dv[None, :],
    )
    if RETURN_LSE and cur_dv_tile == 0:
        lse = e_max + tl.log(e_sum)
        tl.store(
            lse_ptr + cur_q * stride_lse_token + cur_head,
            lse,
            mask=mask_h,
        )


@functools.lru_cache(maxsize=256)
def _choose_num_kv_splits(
    num_tokens: int, num_head_groups: int, index_topk: int, sm_count: int
) -> int:
    """Pick a power-of-2 split count that fills the device without dropping
    per-split work below _MIN_TOPK_PER_SPLIT. Returns 1 when the single-pass
    grid already reaches ~1/_SPLIT_MAX_OCCUPANCY utilization.
    """
    baseline = num_tokens * num_head_groups
    if baseline == 0 or baseline * _SPLIT_MAX_OCCUPANCY >= sm_count:
        return 1
    ideal = triton.next_power_of_2(max(1, index_topk // _MIN_TOPK_PER_SPLIT))
    max_splits = max(1, sm_count // baseline)
    max_splits = 1 << (max_splits.bit_length() - 1)  # floor to power of 2
    num_kv_splits = min(ideal, max_splits)
    while num_kv_splits > 1 and index_topk % num_kv_splits != 0:
        num_kv_splits //= 2
    return max(1, num_kv_splits)


def triton_sparse_mla_attention(
    q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    num_kv_splits: int | None = None,
    sm_count: int | None = None,
    out: torch.Tensor | None = None,
    assume_valid_indices: bool = False,
    valid_index_base_seq_len: int | None = None,
    return_lse: bool = False,
    req_id: torch.Tensor | None = None,
    block_table: torch.Tensor | None = None,
    block_size: int = 64,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Sparse MLA attention over topk indices.

    Args:
        q:         [num_tokens, num_heads_q, dim_qk] bf16 or
                   (q_nope [num_tokens, num_heads_q, 512],
                    q_pe [num_tokens, num_heads_q, 64])
        kv:        [seq_kv, num_heads_kv=1, dim_qk] bf16
        indices:   [num_tokens, num_heads_kv=1, topk] int32. When
            block_table is provided, these are request-local indices that are
            converted to physical KV slots inside the sparse DSA kernel.
        sm_scale:  softmax scale
        num_kv_splits: override auto-heuristic; None/0 = auto, 1 = force single-pass.
        sm_count:  device SM count, used by the split heuristic. If None,
            queried from the device — pass a cached value to avoid a dict
            lookup on every decode step.

    Returns:
        out:   [num_tokens, num_heads_q, _BLOCK_DV] bf16
    """
    if isinstance(q, tuple):
        q_nope, q_pe = q
    else:
        q_nope = q[..., :_BLOCK_DMODEL]
        q_pe = q[..., _BLOCK_DMODEL:]
    num_tokens, num_heads_q, dim_q_nope = q_nope.shape
    assert dim_q_nope == _BLOCK_DMODEL, (
        f"sparse MLA q_nope requires dim={_BLOCK_DMODEL}, got {dim_q_nope}"
    )
    assert q_pe.shape == (num_tokens, num_heads_q, _BLOCK_DPE), (
        "sparse MLA q_pe shape mismatch: "
        f"expected {(num_tokens, num_heads_q, _BLOCK_DPE)}, got {q_pe.shape}"
    )
    assert kv.shape[1] == 1 and kv.shape[2] == _DIM_QK
    index_topk = indices.shape[2]
    assert index_topk % _MIN_BLOCK_N == 0, (
        f"topk ({index_topk}) must be a multiple of the smallest autotune "
        f"BLOCK_N ({_MIN_BLOCK_N})"
    )

    kv_group_num = num_heads_q
    num_head_groups = triton.cdiv(num_heads_q, min(_BLOCK_H, kv_group_num))

    if num_kv_splits is None or num_kv_splits == 0:
        if _FORCE_KV_SPLITS:
            num_kv_splits = int(_FORCE_KV_SPLITS)
        else:
            if sm_count is None:
                sm_count = num_compute_units(q_nope.device.index)
            num_kv_splits = _choose_num_kv_splits(
                num_tokens, num_head_groups, index_topk, sm_count
            )

    if out is None:
        out = torch.empty(
            (num_tokens, num_heads_q, _BLOCK_DV),
            dtype=torch.bfloat16,
            device=q_nope.device,
        )
    elif out.shape != (num_tokens, num_heads_q, _BLOCK_DV):
        raise RuntimeError(
            "triton_sparse_mla_attention out shape mismatch: "
            f"expected {(num_tokens, num_heads_q, _BLOCK_DV)}, got {out.shape}"
        )
    reuse_k_as_v = _REUSE_K_AS_V
    full_block_h = (
        _FULL_BLOCK_H_ENABLED
        and num_heads_q % _BLOCK_H == 0
        and kv_group_num >= _BLOCK_H
    )
    full_block_h_max_tokens = _FULL_BLOCK_H_MAX_TOKENS
    if full_block_h and full_block_h_max_tokens > 0:
        full_block_h = num_tokens <= full_block_h_max_tokens
    lse = None
    if return_lse:
        lse = torch.empty(
            (num_tokens, num_heads_q),
            dtype=torch.float32,
            device=q_nope.device,
        )
    lse_ptr = lse if lse is not None else out
    assume_valid_nomask_enabled = _ASSUME_VALID_NOMASK_ENABLED
    assume_valid_nomask = assume_valid_nomask_enabled and assume_valid_indices
    assume_valid_after_topk_nomask = (
        _ASSUME_VALID_AFTER_TOPK_NOMASK_ENABLED
        and assume_valid_nomask_enabled
        and valid_index_base_seq_len is not None
    )
    assume_prefix_len_mask = (
        _ASSUME_PREFIX_LEN_MASK_ENABLED
        and valid_index_base_seq_len is not None
    )
    convert_req_to_global = req_id is not None and block_table is not None
    if convert_req_to_global:
        assert req_id is not None
        assert block_table is not None
        assert req_id.dtype == torch.int32
        assert block_table.dtype == torch.int32
        assert req_id.device == indices.device
        assert block_table.device == indices.device
        assert req_id.shape[0] >= num_tokens
        req_id_ptr = req_id
        block_table_ptr = block_table
        stride_block_table_req, stride_block_table_block = block_table.stride()
        max_num_blocks_per_req = block_table.shape[1]
    else:
        req_id_ptr = indices
        block_table_ptr = indices
        stride_block_table_req = 0
        stride_block_table_block = 0
        max_num_blocks_per_req = 1

    if num_kv_splits == 1:
        if _can_use_m1_coop_final(
            q_nope=q_nope,
            q_pe=q_pe,
            kv=kv,
            indices=indices,
            out=out,
            num_tokens=num_tokens,
            num_heads_q=num_heads_q,
            index_topk=index_topk,
            num_kv_splits=num_kv_splits,
            assume_valid_indices=assume_valid_indices,
            return_lse=return_lse,
            convert_req_to_global=convert_req_to_global,
        ):
            _load_m1_coop_final_op()
            _log_m1_coop_final_dispatch(
                q_nope=q_nope,
                q_pe=q_pe,
                kv=kv,
                indices=indices,
                out=out,
                num_heads_q=num_heads_q,
                index_topk=index_topk,
                convert_req_to_global=convert_req_to_global,
            )
            partial_acc, partial_meta = (
                _current_workspace_manager().get_simultaneous(
                    (
                        (num_heads_q, _M1_COOP_FINAL_NUM_SPLITS, _BLOCK_DV),
                        torch.float32,
                    ),
                    ((num_heads_q, _M1_COOP_FINAL_NUM_SPLITS, 2), torch.float32),
                )
            )
            torch.ops._C.sparse_mla_m1_coop_final_cuda(
                q_nope,
                q_pe,
                kv,
                indices,
                partial_acc,
                partial_meta,
                out,
                float(sm_scale),
                int(_M1_COOP_FINAL_NUM_SPLITS),
            )
            if _M1_COOP_FINAL_SYNC_DEBUG:
                torch.cuda.synchronize(q_nope.device)
            return out
        if _can_use_m1_splitmerge_final(
            q_nope=q_nope,
            q_pe=q_pe,
            kv=kv,
            indices=indices,
            out=out,
            num_tokens=num_tokens,
            num_heads_q=num_heads_q,
            index_topk=index_topk,
            num_kv_splits=num_kv_splits,
            assume_valid_indices=assume_valid_indices,
            return_lse=return_lse,
            convert_req_to_global=convert_req_to_global,
        ):
            _load_m1_splitmerge_final_op()
            _log_m1_splitmerge_final_dispatch(
                q_nope=q_nope,
                q_pe=q_pe,
                kv=kv,
                indices=indices,
                out=out,
                num_heads_q=num_heads_q,
                index_topk=index_topk,
                convert_req_to_global=convert_req_to_global,
            )
            partial_acc, partial_meta = (
                _current_workspace_manager().get_simultaneous(
                    (
                        (
                            num_heads_q,
                            _M1_SPLITMERGE_FINAL_NUM_SPLITS,
                            _BLOCK_DV,
                        ),
                        torch.float32,
                    ),
                    (
                        (
                            num_heads_q,
                            _M1_SPLITMERGE_FINAL_NUM_SPLITS,
                            2,
                        ),
                        torch.float32,
                    ),
                )
            )
            torch.ops._C.sparse_mla_m1_splitmerge_final_cuda(
                q_nope,
                q_pe,
                kv,
                indices,
                partial_acc,
                partial_meta,
                out,
                float(sm_scale),
                int(_M1_SPLITMERGE_FINAL_NUM_SPLITS),
            )
            if _M1_SPLITMERGE_FINAL_SYNC_DEBUG:
                torch.cuda.synchronize(q_nope.device)
            return out
        if _DECODE_M1_DV_TILE_FINAL_ENABLED and num_tokens == 1:
            tile = _DECODE_M1_DV_TILE
            block_h = _DECODE_M1_DV_TILE_BLOCK_H
            block_n = _DECODE_M1_DV_TILE_BLOCK_N
            if tile <= 0 or _BLOCK_DV % tile != 0:
                raise RuntimeError(
                    "VLLM_SPARSE_MLA_DECODE_M1_DV_TILE must divide "
                    f"{_BLOCK_DV}, got {tile}"
                )
            if block_h <= 0 or block_n <= 0 or index_topk % block_n != 0:
                raise RuntimeError(
                    "invalid sparse MLA decode-M1 DV-tile config: "
                    f"BLOCK_H={block_h}, BLOCK_N={block_n}, topk={index_topk}"
                )
            tile_num_head_groups = triton.cdiv(
                num_heads_q, min(block_h, kv_group_num)
            )
            tile_full_block_h = (
                _FULL_BLOCK_H_ENABLED
                and num_heads_q % block_h == 0
                and kv_group_num >= block_h
            )
            if tile_full_block_h and full_block_h_max_tokens > 0:
                tile_full_block_h = num_tokens <= full_block_h_max_tokens
            _sparse_mla_kernel_final_dv_tile_static[
                (num_tokens, tile_num_head_groups, _BLOCK_DV // tile)
            ](
                q_nope_buffer=q_nope,
                q_pe_buffer=q_pe,
                k_buffer=kv,
                indices_ptr=indices,
                req_id_ptr=req_id_ptr,
                block_table_ptr=block_table_ptr,
                out_ptr=out,
                lse_ptr=lse_ptr,
                seq_kv=kv.shape[0],
                h_q=num_heads_q,
                stride_q_nope_token=q_nope.stride(0),
                stride_q_nope_head=q_nope.stride(1),
                stride_q_pe_token=q_pe.stride(0),
                stride_q_pe_head=q_pe.stride(1),
                stride_kv_token=kv.stride(0),
                stride_kv_head=kv.stride(1),
                stride_out_token=out.stride(0),
                stride_out_head=out.stride(1),
                stride_lse_token=0 if lse is None else lse.stride(0),
                stride_indices_token=indices.stride(0),
                stride_indices_head=indices.stride(1),
                stride_block_table_req=stride_block_table_req,
                stride_block_table_block=stride_block_table_block,
                sm_scale=sm_scale * LOG2E,
                index_topk=index_topk,
                kv_group_num=kv_group_num,
                BLOCK_H=block_h,
                BLOCK_N=block_n,
                BLOCK_DV=_BLOCK_DV,
                BLOCK_DV_TILE=tile,
                BLOCK_DMODEL=_BLOCK_DMODEL,
                BLOCK_DPE=_BLOCK_DPE,
                BLOCK_SIZE=block_size,
                MAX_NUM_BLOCKS_PER_REQ=max_num_blocks_per_req,
                LOGE2=LOGE2,
                FULL_BLOCK_H=tile_full_block_h,
                RETURN_LSE=return_lse,
                CONVERT_REQ_TO_GLOBAL=convert_req_to_global,
                ASSUME_VALID_INDICES=assume_valid_indices,
                ASSUME_VALID_NOMASK=assume_valid_nomask
                and index_topk % block_n == 0,
                ASSUME_VALID_AFTER_TOPK=valid_index_base_seq_len is not None,
                ASSUME_VALID_AFTER_TOPK_NOMASK=assume_valid_after_topk_nomask
                and index_topk % block_n == 0,
                ASSUME_PREFIX_LEN_MASK=assume_prefix_len_mask,
                VALID_INDEX_BASE_SEQ_LEN=0
                if valid_index_base_seq_len is None
                else valid_index_base_seq_len,
                num_warps=_DECODE_M1_DV_TILE_NUM_WARPS,
                num_stages=_DECODE_M1_DV_TILE_NUM_STAGES,
            )
            return (out, lse) if return_lse else out
        if _FINAL_DYNAMIC_CONFIG_ENABLED:
            block_n, num_warps, num_stages = _choose_dynamic_final_static_config(
                num_tokens
            )
            _sparse_mla_kernel_final_static[(num_tokens, num_head_groups)](
                q_nope_buffer=q_nope,
                q_pe_buffer=q_pe,
                k_buffer=kv,
                indices_ptr=indices,
                req_id_ptr=req_id_ptr,
                block_table_ptr=block_table_ptr,
                out_ptr=out,
                lse_ptr=lse_ptr,
                seq_kv=kv.shape[0],
                h_q=num_heads_q,
                stride_q_nope_token=q_nope.stride(0),
                stride_q_nope_head=q_nope.stride(1),
                stride_q_pe_token=q_pe.stride(0),
                stride_q_pe_head=q_pe.stride(1),
                stride_kv_token=kv.stride(0),
                stride_kv_head=kv.stride(1),
                stride_out_token=out.stride(0),
                stride_out_head=out.stride(1),
                stride_lse_token=0 if lse is None else lse.stride(0),
                stride_indices_token=indices.stride(0),
                stride_indices_head=indices.stride(1),
                stride_block_table_req=stride_block_table_req,
                stride_block_table_block=stride_block_table_block,
                sm_scale=sm_scale * LOG2E,
                index_topk=index_topk,
                kv_group_num=kv_group_num,
                BLOCK_H=_BLOCK_H,
                BLOCK_N=block_n,
                BLOCK_DV=_BLOCK_DV,
                BLOCK_DMODEL=_BLOCK_DMODEL,
                BLOCK_DPE=_BLOCK_DPE,
                BLOCK_SIZE=block_size,
                MAX_NUM_BLOCKS_PER_REQ=max_num_blocks_per_req,
                LOGE2=LOGE2,
                REUSE_K_AS_V=reuse_k_as_v,
                FULL_BLOCK_H=full_block_h,
                RETURN_LSE=return_lse,
                CONVERT_REQ_TO_GLOBAL=convert_req_to_global,
                ASSUME_VALID_INDICES=assume_valid_indices,
                ASSUME_VALID_NOMASK=assume_valid_nomask
                and index_topk % block_n == 0,
                ASSUME_VALID_AFTER_TOPK=valid_index_base_seq_len is not None,
                ASSUME_VALID_AFTER_TOPK_NOMASK=assume_valid_after_topk_nomask
                and index_topk % block_n == 0,
                ASSUME_PREFIX_LEN_MASK=assume_prefix_len_mask,
                VALID_INDEX_BASE_SEQ_LEN=0
                if valid_index_base_seq_len is None
                else valid_index_base_seq_len,
                num_warps=num_warps,
                num_stages=num_stages,
            )
            return (out, lse) if return_lse else out
        if _FINAL_STATIC_BY_TOKENS_ENABLED:
            if num_tokens >= 4096:
                block_n, num_warps, num_stages = 16, 4, 4
            else:
                block_n, num_warps, num_stages = 32, 4, 1
            _sparse_mla_kernel_final_static[(num_tokens, num_head_groups)](
                q_nope_buffer=q_nope,
                q_pe_buffer=q_pe,
                k_buffer=kv,
                indices_ptr=indices,
                req_id_ptr=req_id_ptr,
                block_table_ptr=block_table_ptr,
                out_ptr=out,
                lse_ptr=lse_ptr,
                seq_kv=kv.shape[0],
                h_q=num_heads_q,
                stride_q_nope_token=q_nope.stride(0),
                stride_q_nope_head=q_nope.stride(1),
                stride_q_pe_token=q_pe.stride(0),
                stride_q_pe_head=q_pe.stride(1),
                stride_kv_token=kv.stride(0),
                stride_kv_head=kv.stride(1),
                stride_out_token=out.stride(0),
                stride_out_head=out.stride(1),
                stride_lse_token=0 if lse is None else lse.stride(0),
                stride_indices_token=indices.stride(0),
                stride_indices_head=indices.stride(1),
                stride_block_table_req=stride_block_table_req,
                stride_block_table_block=stride_block_table_block,
                sm_scale=sm_scale * LOG2E,
                index_topk=index_topk,
                kv_group_num=kv_group_num,
                BLOCK_H=_BLOCK_H,
                BLOCK_N=block_n,
                BLOCK_DV=_BLOCK_DV,
                BLOCK_DMODEL=_BLOCK_DMODEL,
                BLOCK_DPE=_BLOCK_DPE,
                BLOCK_SIZE=block_size,
                MAX_NUM_BLOCKS_PER_REQ=max_num_blocks_per_req,
                LOGE2=LOGE2,
                REUSE_K_AS_V=reuse_k_as_v,
                FULL_BLOCK_H=full_block_h,
                RETURN_LSE=return_lse,
                CONVERT_REQ_TO_GLOBAL=convert_req_to_global,
                ASSUME_VALID_INDICES=assume_valid_indices,
                ASSUME_VALID_NOMASK=assume_valid_nomask
                and index_topk % block_n == 0,
                ASSUME_VALID_AFTER_TOPK=valid_index_base_seq_len is not None,
                ASSUME_VALID_AFTER_TOPK_NOMASK=assume_valid_after_topk_nomask
                and index_topk % block_n == 0,
                ASSUME_PREFIX_LEN_MASK=assume_prefix_len_mask,
                VALID_INDEX_BASE_SEQ_LEN=0
                if valid_index_base_seq_len is None
                else valid_index_base_seq_len,
                num_warps=num_warps,
                num_stages=num_stages,
            )
            return (out, lse) if return_lse else out
        _sparse_mla_kernel_final[(num_tokens, num_head_groups)](
            q_nope_buffer=q_nope,
            q_pe_buffer=q_pe,
            k_buffer=kv,
            indices_ptr=indices,
            req_id_ptr=req_id_ptr,
            block_table_ptr=block_table_ptr,
            out_ptr=out,
            lse_ptr=lse_ptr,
            seq_kv=kv.shape[0],
            h_q=num_heads_q,
            num_tokens=num_tokens,
            stride_q_nope_token=q_nope.stride(0),
            stride_q_nope_head=q_nope.stride(1),
            stride_q_pe_token=q_pe.stride(0),
            stride_q_pe_head=q_pe.stride(1),
            stride_kv_token=kv.stride(0),
            stride_kv_head=kv.stride(1),
            stride_out_token=out.stride(0),
            stride_out_head=out.stride(1),
            stride_lse_token=0 if lse is None else lse.stride(0),
            stride_indices_token=indices.stride(0),
            stride_indices_head=indices.stride(1),
            stride_block_table_req=stride_block_table_req,
            stride_block_table_block=stride_block_table_block,
            sm_scale=sm_scale * LOG2E,
            index_topk=index_topk,
            kv_group_num=kv_group_num,
            BLOCK_H=_BLOCK_H,
            BLOCK_DV=_BLOCK_DV,
            BLOCK_DMODEL=_BLOCK_DMODEL,
            BLOCK_DPE=_BLOCK_DPE,
            BLOCK_SIZE=block_size,
            MAX_NUM_BLOCKS_PER_REQ=max_num_blocks_per_req,
            LOGE2=LOGE2,
            REUSE_K_AS_V=reuse_k_as_v,
            FULL_BLOCK_H=full_block_h,
            RETURN_LSE=return_lse,
            CONVERT_REQ_TO_GLOBAL=convert_req_to_global,
            ASSUME_VALID_INDICES=assume_valid_indices,
            ASSUME_VALID_NOMASK=assume_valid_nomask and index_topk % 32 == 0,
            ASSUME_VALID_AFTER_TOPK=valid_index_base_seq_len is not None,
            ASSUME_VALID_AFTER_TOPK_NOMASK=assume_valid_after_topk_nomask
            and index_topk % 32 == 0,
            ASSUME_PREFIX_LEN_MASK=assume_prefix_len_mask,
            VALID_INDEX_BASE_SEQ_LEN=0
            if valid_index_base_seq_len is None
                else valid_index_base_seq_len,
        )
        return (out, lse) if return_lse else out

    # Split-KV: partial fp32 output + LSE per (token, head, split).
    mid_out = torch.empty(
        (num_tokens, num_heads_q, num_kv_splits, _BLOCK_DV + 1),
        dtype=torch.float32,
        device=q_nope.device,
    )
    _sparse_mla_kernel_split[(num_tokens, num_head_groups, num_kv_splits)](
        q_nope_buffer=q_nope,
        q_pe_buffer=q_pe,
        k_buffer=kv,
        indices_ptr=indices,
        req_id_ptr=req_id_ptr,
        block_table_ptr=block_table_ptr,
        mid_out_ptr=mid_out,
        seq_kv=kv.shape[0],
        h_q=num_heads_q,
        stride_q_nope_token=q_nope.stride(0),
        stride_q_nope_head=q_nope.stride(1),
        stride_q_pe_token=q_pe.stride(0),
        stride_q_pe_head=q_pe.stride(1),
        stride_kv_token=kv.stride(0),
        stride_kv_head=kv.stride(1),
        stride_mid_token=mid_out.stride(0),
        stride_mid_head=mid_out.stride(1),
        stride_mid_split=mid_out.stride(2),
        stride_indices_token=indices.stride(0),
        stride_indices_head=indices.stride(1),
        stride_block_table_req=stride_block_table_req,
        stride_block_table_block=stride_block_table_block,
        sm_scale=sm_scale * LOG2E,
        index_topk=index_topk,
        NUM_KV_SPLITS=num_kv_splits,
        kv_group_num=kv_group_num,
        BLOCK_H=_BLOCK_H,
        BLOCK_DV=_BLOCK_DV,
        BLOCK_DMODEL=_BLOCK_DMODEL,
        BLOCK_DPE=_BLOCK_DPE,
        BLOCK_SIZE=block_size,
        MAX_NUM_BLOCKS_PER_REQ=max_num_blocks_per_req,
        LOGE2=LOGE2,
        REUSE_K_AS_V=reuse_k_as_v,
        FULL_BLOCK_H=full_block_h,
        CONVERT_REQ_TO_GLOBAL=convert_req_to_global,
        ASSUME_VALID_INDICES=assume_valid_indices,
        ASSUME_VALID_NOMASK=False,
        ASSUME_VALID_AFTER_TOPK=False,
        ASSUME_VALID_AFTER_TOPK_NOMASK=False,
        ASSUME_PREFIX_LEN_MASK=False,
        VALID_INDEX_BASE_SEQ_LEN=0,
    )

    _sparse_mla_merge_kernel[(num_tokens, num_heads_q, _NUM_MERGE_DV_TILES)](
        mid_out_ptr=mid_out,
        out_ptr=out,
        lse_ptr=lse_ptr,
        h_q=num_heads_q,
        stride_mid_token=mid_out.stride(0),
        stride_mid_head=mid_out.stride(1),
        stride_mid_split=mid_out.stride(2),
        stride_out_token=out.stride(0),
        stride_out_head=out.stride(1),
        stride_lse_token=0 if lse is None else lse.stride(0),
        NUM_KV_SPLITS=num_kv_splits,
        kv_group_num=kv_group_num,
        BLOCK_H=_MERGE_BLOCK_H,
        BLOCK_DV=_BLOCK_DV,
        BLOCK_DV_TILE=_MERGE_BLOCK_DV_TILE,
        RETURN_LSE=return_lse,
        num_warps=2,
    )
    return (out, lse) if return_lse else out
