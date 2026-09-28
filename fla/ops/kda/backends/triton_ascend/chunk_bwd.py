# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""KDA chunk backward kernels for triton-ascend on Ascend NPU."""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.runtime import driver

from fla.ops.utils import prepare_chunk_indices, prepare_chunk_offsets
from fla.ops.utils.op import exp2
from fla.utils import ascend_compile_kwargs, input_guard
from fla.utils.ascend_ub_manager import (
    ASCEND_MAX_GRID_DIM,
    compute_row_tile_block_size,
    max_grid_axis_chunks,
)

_DAV_NUM_WARPS = 2
# the hoisted A tile stays live across V-slab iterations within the UB budget.
_DAV_MEM_MULT = 4.0
_DAV_SAFETY_MARGIN = 0.75
_DAV_FALLBACK_TILE = 8
_DAV_MAX_TILE = 128


def _get_dAv_bv(BT: int, V: int) -> int:
    return compute_row_tile_block_size(
        BT, V, _DAV_MEM_MULT,
        tiling_row=False,
        safety_margin=_DAV_SAFETY_MARGIN,
        fallback=_DAV_FALLBACK_TILE,
        min_block=8,
        max_block=min(_DAV_MAX_TILE, triton.next_power_of_2(V)),
    )


def _launch_dAv_2d_kernel(kernel, *, nt: int, bh_total: int, kernel_kwargs: dict) -> None:
    max_nt = max_grid_axis_chunks(nt, bh_total, max_grid=ASCEND_MAX_GRID_DIM)
    for nt_off in range(0, nt, max_nt):
        nt_len = min(max_nt, nt - nt_off)
        kernel_kwargs['NT_OFFSET'] = nt_off
        max_bh = max_grid_axis_chunks(bh_total, nt_len, max_grid=ASCEND_MAX_GRID_DIM)
        for bh_off in range(0, bh_total, max_bh):
            bh_len = min(max_bh, bh_total - bh_off)
            kernel_kwargs['BH_OFFSET'] = bh_off
            kernel[(nt_len, bh_len)](num_warps=_DAV_NUM_WARPS, **kernel_kwargs)


@triton.jit(do_not_specialize=['T'])
def chunk_kda_bwd_kernel_dAv_npu(
    v,
    A,
    do,
    dv,
    dA,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    HV: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    NT_OFFSET: tl.constexpr,
    BH_OFFSET: tl.constexpr,
):
    i_t = tl.program_id(0) + NT_OFFSET
    i_bh = tl.program_id(1) + BH_OFFSET
    i_b, i_hv = i_bh // HV, i_bh % HV
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = (eos - bos).to(tl.int32)
    else:
        bos = tl.cast(i_b, tl.int64) * T
        eos = bos + T

    v += (bos * HV + i_hv) * V
    do += (bos * HV + i_hv) * V
    dv += (bos * HV + i_hv) * V
    dA += (bos * HV + i_hv) * BT

    o_i = tl.arange(0, BT)
    o_t = i_t * BT + o_i
    m_t = o_t < T
    m_A = (o_t[:, None] <= o_t[None, :]) & (m_t[:, None] & m_t)

    # cast the row index before multiplication to preserve large offsets.
    p_row = tl.cast(i_t, tl.int64) * BT
    p_A = A + (bos * HV + i_hv) * BT

    b_dA = tl.zeros([BT, BT], dtype=tl.float32)
    # mask/cast the A tile once per chunk
    b_A = tl.load(p_A + p_row * (HV * BT) + o_i[None, :] * (HV * BT) + o_i[:, None],
                  mask=m_t[None, :], other=0.0)
    b_A = tl.where(m_A, b_A, 0).to(do.dtype.element_ty)
    for i_v in range(tl.cdiv(V, BV)):
        o_v = i_v * BV + tl.arange(0, BV)
        m_v = o_v < V
        b_v = tl.load(v + p_row * (HV * V) + o_i[None, :] * (HV * V) + o_v[:, None],
                      mask=m_t[None, :] & m_v[:, None], other=0.0)
        b_do = tl.load(do + p_row * (HV * V) + o_i[:, None] * (HV * V) + o_v[None, :],
                       mask=m_t[:, None] & m_v[None, :], other=0.0)
        b_do_c = b_do + 0.0
        b_dA = tl.dot(b_do, b_v, b_dA, allow_tf32=False)
        # fresh lhs each slab: tl.dot clobbers its lhs and b_A is live across iterations
        b_dv = tl.dot(b_A + 0.0, b_do_c, allow_tf32=False)
        tl.store(dv + p_row * (HV * V) + o_i[:, None] * (HV * V) + o_v[None, :],
                 b_dv.to(dv.dtype.element_ty), mask=m_t[:, None] & m_v[None, :])

    b_dA = tl.where(o_t[:, None] >= o_t, b_dA * scale, 0.)
    tl.store(dA + p_row * (HV * BT) + o_i[:, None] * (HV * BT) + o_i[None, :],
             b_dA.to(dA.dtype.element_ty), mask=m_t[:, None])


@input_guard
def chunk_kda_bwd_dAv_npu(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    do: torch.Tensor,
    A: torch.Tensor | None = None,
    scale: float = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
    use_graph: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if use_graph:
        raise NotImplementedError("use_graph is not supported on the Ascend NPU backend")
    B, T, HV, V = k.shape[0], k.shape[1], do.shape[2], do.shape[-1]
    BT = chunk_size
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    BV = _get_dAv_bv(BT, V)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    dA = v.new_empty(B, T, HV, BT, dtype=torch.float)
    # all slots are stored by the V-slab sweep
    dv = torch.empty_like(do)

    _launch_dAv_2d_kernel(
        chunk_kda_bwd_kernel_dAv_npu,
        nt=NT,
        bh_total=B * HV,
        kernel_kwargs=dict(
            v=v,
            A=A,
            do=do,
            dv=dv,
            dA=dA,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            scale=scale,
            T=T,
            HV=HV,
            V=V,
            BT=BT,
            BV=BV,
            IS_VARLEN=cu_seqlens is not None,
            NT_OFFSET=0,
            BH_OFFSET=0,
        ),
    )
    return dA, dv


_BC = 16
_BWD_MEM_MULT = 10.0
_SAFETY_MARGIN = 0.80
_FALLBACK_TILE = 16
_MAX_TILE = 128


# STATE_V_FIRST + BK=BV=128 exceeds the 192KB UB and fails to compile; tiles
# are picked without consulting state_v_first, so covered shapes must keep at
# least one tile below 128.
def _get_bk(K: int) -> int:
    return compute_row_tile_block_size(
        _BC,
        K,
        _BWD_MEM_MULT,
        tiling_row=False,
        safety_margin=_SAFETY_MARGIN,
        fallback=_FALLBACK_TILE,
        min_block=16,
        max_block=min(_MAX_TILE, triton.next_power_of_2(K)),
    )


def _get_bv(V: int) -> int:
    return compute_row_tile_block_size(
        _BC,
        V,
        _BWD_MEM_MULT,
        tiling_row=False,
        safety_margin=_SAFETY_MARGIN,
        fallback=_FALLBACK_TILE,
        min_block=16,
        max_block=min(_MAX_TILE, triton.next_power_of_2(V)),
    )


def _t_contig_arg(x: torch.Tensor, head_dim: int) -> tuple[torch.Tensor, bool]:
    """Transpose [B, T, H*, ...] → [B, H*, T, ...] so T-loads are stride-1."""
    if head_dim == 1:
        return x, False
    return x.transpose(1, 2).contiguous(), True


def get_npu_properties():
    device = torch.npu.current_device()
    return driver.active.utils.get_device_properties(device)


def _launch_wy_dA_finalize(
    kernel,
    *,
    nt: int,
    bh_total: int,
    T: int,
    BT: int,
    is_varlen: bool,
    num_core: int,
    kernel_kwargs: dict,
) -> None:
    """Host-split aligned bulk vs tail so TAIL_MODE is constexpr per launch."""
    kwargs = dict(kernel_kwargs)
    kwargs['num_core'] = num_core
    if is_varlen:
        kwargs['TAIL_MODE'] = 1
        kwargs['NT_OFFSET'] = 0
        kwargs['task_num'] = nt * bh_total
        kernel[(num_core,)](**kwargs)
        return
    n_bulk = nt if T % BT == 0 else max(nt - 1, 0)
    if n_bulk > 0:
        # Batched-x2 bulk finalize: two tasks per iteration, stage-interleaved
        # (loads | masks | dot1 x2 | dot2 x2 | stores).
        task_num = n_bulk * bh_total
        chunk_kda_bwd_kernel_wy_dA_finalize_x2_npu[(num_core,)](
            A=kwargs['A'], beta=kwargs['beta'], dA_acc=kwargs['dA_acc'],
            db_acc=kwargs['db_acc'], dA=kwargs['dA'], db=kwargs['db'],
            T=T, BH=kwargs['BH'], task_num=task_num,
            task_num2=(task_num + 1) // 2, num_core=num_core, NT_OFFSET=0,
            HV=kwargs['HV'], BT=BT, G_T_CONTIG=kwargs['G_T_CONTIG'])
    if T % BT != 0 and nt > 0:
        kwargs['TAIL_MODE'] = 1
        kwargs['NT_OFFSET'] = n_bulk
        kwargs['task_num'] = bh_total
        kernel[(num_core,)](**kwargs)


@triton.jit(do_not_specialize=['T', 'task_num', 'num_core', 'BH'])
def chunk_kda_bwd_kernel_wy_v_part_npu(
    v,
    beta,
    A,
    dv,
    dv2,
    dA_acc,
    db_acc,
    cu_seqlens,
    chunk_indices,
    T,
    BH,
    task_num,
    num_core,
    HV: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    G_T_CONTIG: tl.constexpr,
):
    core_id = tl.program_id(0)
    T_seq = T

    for task_id in tl.range(core_id, task_num, num_core):
        i_t = task_id // BH
        i_bh = task_id % BH
        i_b, i_hv = i_bh // HV, i_bh % HV

        if IS_VARLEN:
            i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
            bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
            T = (eos - bos).to(tl.int32)
        else:
            bos, eos = tl.cast(i_b, tl.int64) * T, tl.cast(i_b, tl.int64) * T + T

        if G_T_CONTIG:
            if IS_VARLEN:
                v_ptr = v + tl.cast(i_hv, tl.int64) * T_seq * V + bos * V
                dv_ptr = dv + tl.cast(i_hv, tl.int64) * T_seq * V + bos * V
                A_ptr = A + tl.cast(i_hv, tl.int64) * T_seq * BT + bos * BT
                beta_ptr = beta + tl.cast(i_hv, tl.int64) * T_seq + bos
            else:
                hv_off = tl.cast(i_b, tl.int64) * HV + i_hv
                v_ptr = v + hv_off * T_seq * V
                dv_ptr = dv + hv_off * T_seq * V
                A_ptr = A + hv_off * T_seq * BT
                beta_ptr = beta + hv_off * T_seq
            v_stride_t = V
            a_stride_t = BT
            beta_stride = 1
        else:
            v_ptr = v + (bos * HV + i_hv) * V
            dv_ptr = dv + (bos * HV + i_hv) * V
            A_ptr = A + (bos * HV + i_hv) * BT
            beta_ptr = beta + bos * HV + i_hv
            v_stride_t = HV * V
            a_stride_t = HV * BT
            beta_stride = HV

        dv2_ptr = dv2 + (bos * HV + i_hv) * V
        dA_ptr = dA_acc + (bos * HV + i_hv) * BT
        db_ptr = db_acc + bos * HV + i_hv

        p_A = tl.make_block_ptr(A_ptr, (BT, T), (1, a_stride_t), (0, i_t * BT), (BT, BT), (0, 1))
        p_beta = tl.make_block_ptr(beta_ptr, (T,), (beta_stride,), (i_t * BT,), (BT,), (0,))
        b_A = tl.load(p_A, boundary_check=(0, 1))
        b_beta = tl.load(p_beta, boundary_check=(0,))

        b_dA = tl.zeros([BT, BT], dtype=tl.float32)
        b_db = tl.zeros([BT], dtype=tl.float32)
        for i_v in range(tl.cdiv(V, BV)):
            p_dv = tl.make_block_ptr(dv_ptr, (T, V), (v_stride_t, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            p_v = tl.make_block_ptr(v_ptr, (T, V), (v_stride_t, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            b_dv = tl.load(p_dv, boundary_check=(0, 1))
            b_v = tl.load(p_v, boundary_check=(0, 1))
            b_dA = tl.dot(b_dv, tl.trans(b_v), b_dA, allow_tf32=False)
            # Ascend tl.dot clobbers lhs; copy A before every V-slab use.
            b_A_c = b_A + 0.0
            b_dvb = tl.dot(b_A_c, b_dv, allow_tf32=False)
            b_db += tl.sum(b_dvb * b_v, 1)
            p_dv2 = tl.make_block_ptr(dv2_ptr, (T, V), (HV * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            tl.store(p_dv2, (b_dvb * b_beta[:, None]).to(p_dv2.dtype.element_ty), boundary_check=(0, 1))

        p_dA = tl.make_block_ptr(dA_ptr, (T, BT), (HV * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
        p_db = tl.make_block_ptr(db_ptr, (T,), (HV,), (i_t * BT,), (BT,), (0,))
        tl.store(p_dA, b_dA.to(p_dA.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_db, b_db.to(p_db.dtype.element_ty), boundary_check=(0,))


@triton.jit
def chunk_kda_bwd_wy_k_part_sub0(
    i_t,
    T,
    q_ptr,
    k_ptr,
    v_new_ptr,
    g_ptr,
    h_ptr,
    do_ptr,
    dh_ptr,
    dq_ptr,
    dk_ptr,
    dg_ptr,
    b_gn,
    scale,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    STATE_V_FIRST: tl.constexpr,
    ALIGNED: tl.constexpr,
    K_OFFSET: tl.constexpr,
):
    """Loop-free sub-block body for NSUB == 1 (BT == BC, chunk_size 32).

    bishengir (triton-ascend 3.2.2) SIGABRTs in ConvertLinalgRToBinary when the
    merged single-pass body sits in a trip-count-1 scf.for: the folded
    straight-line linalg region trips the pass (the same body compiles fine at
    NSUB >= 2, and the old two-sweep kernel compiles fine at trip 1). The caller
    branches here at trace time when BT // BC == 1. h/dh are loaded per use
    instead of kept resident - chunk_size 32 is a niche config and the state
    slabs stay L2-resident anyway. Returns sum(k * dk, 0) so the caller owns
    the loop-free b_kdk_sum accumulation.
    """
    i_k = K_OFFSET
    o_k = i_k * BK + tl.arange(0, BK)
    o_v = tl.arange(0, BV)
    o_i = tl.arange(0, BC)
    if K % BK:
        m_k = o_k < K

    i_tc_s = i_t * BT
    row_k = i_tc_s.to(tl.int64) * H * K + o_i[:, None] * (H * K)
    row_g = i_tc_s.to(tl.int64) * HV * K + o_i[:, None] * (HV * K)
    row_v = i_tc_s.to(tl.int64) * HV * V + o_i[:, None] * (HV * V)
    p_k = k_ptr + row_k + o_k[None, :]
    p_g = g_ptr + row_g + o_k[None, :]
    p_q = q_ptr + row_k + o_k[None, :]
    m_s = i_tc_s + o_i < T
    if K % BK:
        m_row = m_s[:, None] & m_k[None, :]

    b_dk = tl.zeros([BC, BK], dtype=tl.float32)
    for i_v in range(tl.cdiv(V, BV)):
        o_vv = i_v * BV + o_v
        p_vn = v_new_ptr + row_v + o_vv[None, :]
        if STATE_V_FIRST:
            p_dh = dh_ptr + o_vv[:, None] * K + o_k[None, :]
        else:
            p_dh = dh_ptr + o_vv[:, None] + o_k[None, :] * V
        if ALIGNED:
            b_vn = tl.load(p_vn)
            b_dhs = tl.load(p_dh)
        else:
            m_vn = m_s[:, None]
            if V % BV:
                m_vn = m_vn & (o_vv < V)[None, :]
            m_hvs = (o_vv < V)[:, None]
            if K % BK:
                m_hvs = m_hvs & m_k[None, :]
            b_vn = tl.load(p_vn, mask=m_vn, other=0.0)
            b_dhs = tl.load(p_dh, mask=m_hvs, other=0.0)
        b_dk = tl.dot(b_vn, b_dhs.to(b_vn.dtype), b_dk, allow_tf32=False)

    if ALIGNED:
        b_k = tl.load(p_k)
        b_g = tl.load(p_g).to(tl.float32)
    else:
        if K % BK:
            b_k = tl.load(p_k, mask=m_row, other=0.0)
            b_g = tl.load(p_g, mask=m_row, other=0.0).to(tl.float32)
        else:
            b_k = tl.load(p_k, mask=m_s[:, None], other=0.0)
            b_g = tl.load(p_g, mask=m_s[:, None], other=0.0).to(tl.float32)
    b_dk = b_dk * tl.where(m_s[:, None], exp2(b_gn[None, :] - b_g), 0)
    p_dk = dk_ptr + row_g + o_k[None, :]
    if K % BK:
        tl.store(p_dk, b_dk.to(dk_ptr.dtype.element_ty), mask=m_row)
    else:
        tl.store(p_dk, b_dk.to(dk_ptr.dtype.element_ty), mask=m_s[:, None])

    b_dq = tl.zeros([BC, BK], dtype=tl.float32)
    for i_v in range(tl.cdiv(V, BV)):
        o_vv = i_v * BV + o_v
        p_do = do_ptr + row_v + o_vv[None, :]
        if STATE_V_FIRST:
            p_h = h_ptr + o_vv[:, None] * K + o_k[None, :]
        else:
            p_h = h_ptr + o_vv[:, None] + o_k[None, :] * V
        if ALIGNED:
            b_do = tl.load(p_do)
            b_hs = tl.load(p_h)
        else:
            m_do = m_s[:, None]
            if V % BV:
                m_do = m_do & (o_vv < V)[None, :]
            m_hvs = (o_vv < V)[:, None]
            if K % BK:
                m_hvs = m_hvs & m_k[None, :]
            b_do = tl.load(p_do, mask=m_do, other=0.0)
            b_hs = tl.load(p_h, mask=m_hvs, other=0.0)
        b_dq = tl.dot(b_do, b_hs.to(b_do.dtype), b_dq, allow_tf32=False)

    if ALIGNED:
        b_q = tl.load(p_q)
    else:
        if K % BK:
            b_q = tl.load(p_q, mask=m_row, other=0.0)
        else:
            b_q = tl.load(p_q, mask=m_s[:, None], other=0.0)
    b_dq = b_dq * exp2(b_g) * scale
    b_dg = b_q * b_dq - b_k * b_dk

    p_dq = dq_ptr + row_g + o_k[None, :]
    p_dg = dg_ptr + row_g + o_k[None, :]
    if K % BK:
        tl.store(p_dq, b_dq.to(dq_ptr.dtype.element_ty), mask=m_row)
        tl.store(p_dg, b_dg.to(dg_ptr.dtype.element_ty), mask=m_row)
    else:
        tl.store(p_dq, b_dq.to(dq_ptr.dtype.element_ty), mask=m_s[:, None])
        tl.store(p_dg, b_dg.to(dg_ptr.dtype.element_ty), mask=m_s[:, None])
    return tl.sum(b_k * b_dk, axis=0)


@triton.jit(do_not_specialize=['T', 'task_num', 'num_core', 'BH'])
def chunk_kda_bwd_kernel_wy_k_part_npu(
    q,
    k,
    v_new,
    g,
    h,
    do,
    dh,
    dq,
    dk,
    dg,
    cu_seqlens,
    chunk_indices,
    chunk_offsets,
    scale,
    T,
    BH,
    task_num,
    num_core,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    STATE_V_FIRST: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    K_OFFSET: tl.constexpr,
    ALIGNED: tl.constexpr,
):
    """Single-pass k_part: dk and dq/dg computed in one sub-block sweep.

    vs the old two-sweep version: dk stays in registers (no GM round-trip),
    k/g are read once per sub-block, and when V fits one BV slab (SINGLE_SLAB)
    h/dh are loaded once per task and reused for the dgk reduction, the dk dot
    and the dq dot (previously 3 reads each). Bare pointers throughout.
    ALIGNED=1 (dense, T % BT == 0, K % BK == 0, V % BV == 0) drops all
    row/column-boundary masks.
    """
    i_k = K_OFFSET
    core_id = tl.program_id(0)
    SINGLE_SLAB: tl.constexpr = V <= BV

    for task_id in tl.range(core_id, task_num, num_core):
        i_t = task_id // BH
        i_bh = task_id % BH
        i_b, i_hv = i_bh // HV, i_bh % HV
        i_h = i_hv // (HV // H)

        if IS_VARLEN:
            i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
            bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
            T = (eos - bos).to(tl.int32)
            i_tg = tl.load(chunk_offsets + i_n).to(tl.int64) + i_t.to(tl.int64)
        else:
            i_tg = tl.cast(i_b, tl.int64) * tl.cdiv(T, BT) + i_t
            bos = tl.cast(i_b, tl.int64) * T

        q_ptr = q + (bos * H + i_h) * K
        k_ptr = k + (bos * H + i_h) * K
        v_new_ptr = v_new + (bos * HV + i_hv) * V
        g_ptr = g + (bos * HV + i_hv) * K
        h_ptr = h + (i_tg * HV + i_hv) * K * V
        do_ptr = do + (bos * HV + i_hv) * V
        dh_ptr = dh + (i_tg * HV + i_hv) * K * V
        dq_ptr = dq + (bos * HV + i_hv) * K
        dk_ptr = dk + (bos * HV + i_hv) * K
        dg_ptr = dg + (bos * HV + i_hv) * K

        o_k = i_k * BK + tl.arange(0, BK)
        o_v = tl.arange(0, BV)
        o_i = tl.arange(0, BC)
        NSUB: tl.constexpr = BT // BC
        if K % BK:
            m_k = o_k < K
        if V % BV:
            m_v = o_v < V

        p_gn = g_ptr + (min(T, i_t * BT + BT) - 1).to(tl.int64) * HV * K + o_k
        if K % BK:
            b_gn = tl.load(p_gn, mask=m_k, other=0).to(tl.float32)
        else:
            b_gn = tl.load(p_gn).to(tl.float32)

        # h/dh slab(s): loaded once when a single BV slab covers V, else streamed.
        b_dgk = tl.zeros([BK], dtype=tl.float32)
        if SINGLE_SLAB:
            # The row (V) and column (K) guards are independent: K % BK can
            # overflow the slab columns even when V is exactly covered (K=100).
            if V % BV or K % BK:
                m_hv = m_k[None, :]
                if V % BV:
                    m_hv = m_hv & m_v[:, None]
            else:
                m_hv = None
            if STATE_V_FIRST:
                p_h = h_ptr + o_v[:, None] * K + o_k[None, :]
                p_dh = dh_ptr + o_v[:, None] * K + o_k[None, :]
            else:
                p_h = h_ptr + o_v[:, None] + o_k[None, :] * V
                p_dh = dh_ptr + o_v[:, None] + o_k[None, :] * V
            if m_hv is None:
                b_h0 = tl.load(p_h)
                b_dh0 = tl.load(p_dh)
            else:
                b_h0 = tl.load(p_h, mask=m_hv, other=0.0)
                b_dh0 = tl.load(p_dh, mask=m_hv, other=0.0)
            b_dgk += tl.sum(b_h0 * b_dh0, axis=0)
        else:
            for i_v in range(tl.cdiv(V, BV)):
                if STATE_V_FIRST:
                    p_h = h_ptr + (i_v * BV + o_v[:, None]) * K + o_k[None, :]
                    p_dh = dh_ptr + (i_v * BV + o_v[:, None]) * K + o_k[None, :]
                else:
                    p_h = h_ptr + (i_v * BV + o_v[:, None]) + o_k[None, :] * V
                    p_dh = dh_ptr + (i_v * BV + o_v[:, None]) + o_k[None, :] * V
                if V % BV or K % BK:
                    m_hv = m_k[None, :]
                    if V % BV:
                        m_hv = m_hv & m_v[:, None]
                    b_h0 = tl.load(p_h, mask=m_hv, other=0.0)
                    b_dh0 = tl.load(p_dh, mask=m_hv, other=0.0)
                else:
                    b_h0 = tl.load(p_h)
                    b_dh0 = tl.load(p_dh)
                b_dgk += tl.sum(b_h0 * b_dh0, axis=0)
        b_dgk *= exp2(b_gn)

        # bishengir (triton-ascend 3.2.2) SIGABRTs in ConvertLinalgRToBinary on a
        # trip-count-1 scf.for carrying this merged body (BT == BC, chunk_size
        # 32): dispatch to the loop-free straight-line helper instead.
        if NSUB == 1:
            b_kdk_sum = chunk_kda_bwd_wy_k_part_sub0(
                i_t, T,
                q_ptr, k_ptr, v_new_ptr, g_ptr, h_ptr, do_ptr, dh_ptr, dq_ptr, dk_ptr, dg_ptr,
                b_gn, scale,
                H=H, HV=HV, K=K, V=V, BT=BT, BC=BC, BK=BK, BV=BV,
                STATE_V_FIRST=STATE_V_FIRST, ALIGNED=ALIGNED, K_OFFSET=i_k,
            )
        else:
            b_kdk_sum = tl.zeros([BK], dtype=tl.float32)
            for s in range(NSUB):
                i_tc_s = i_t * BT + s * BC
                row_k = i_tc_s.to(tl.int64) * H * K + o_i[:, None] * (H * K)
                row_g = i_tc_s.to(tl.int64) * HV * K + o_i[:, None] * (HV * K)
                row_v = i_tc_s.to(tl.int64) * HV * V + o_i[:, None] * (HV * V)
                p_k = k_ptr + row_k + o_k[None, :]
                p_g = g_ptr + row_g + o_k[None, :]
                p_q = q_ptr + row_k + o_k[None, :]
                m_s = i_tc_s + o_i < T
                if K % BK:
                    m_row = m_s[:, None] & m_k[None, :]

                # b_dk = v_new @ dh  (V slabs; SINGLE_SLAB reuses the resident b_dh0)
                b_dk = tl.zeros([BC, BK], dtype=tl.float32)
                if SINGLE_SLAB:
                    p_vn = v_new_ptr + row_v + o_v[None, :]
                    if ALIGNED:
                        b_vn = tl.load(p_vn)
                    else:
                        m_vn = m_s[:, None]
                        if V % BV:
                            m_vn = m_vn & m_v[None, :]
                        b_vn = tl.load(p_vn, mask=m_vn, other=0.0)
                    b_dk = tl.dot(b_vn, b_dh0.to(b_vn.dtype), b_dk, allow_tf32=False)
                else:
                    for i_v in range(tl.cdiv(V, BV)):
                        o_vv = i_v * BV + o_v
                        p_vn = v_new_ptr + row_v + o_vv[None, :]
                        if STATE_V_FIRST:
                            p_dh = dh_ptr + o_vv[:, None] * K + o_k[None, :]
                        else:
                            p_dh = dh_ptr + o_vv[:, None] + o_k[None, :] * V
                        if ALIGNED:
                            b_vn = tl.load(p_vn)
                            b_dhs = tl.load(p_dh)
                        else:
                            m_vn = m_s[:, None]
                            if V % BV:
                                m_vn = m_vn & (o_vv < V)[None, :]
                            m_hvs = (o_vv < V)[:, None]
                            if K % BK:
                                m_hvs = m_hvs & m_k[None, :]
                            b_vn = tl.load(p_vn, mask=m_vn, other=0.0)
                            b_dhs = tl.load(p_dh, mask=m_hvs, other=0.0)
                        b_dk = tl.dot(b_vn, b_dhs.to(b_vn.dtype), b_dk, allow_tf32=False)

                if ALIGNED:
                    b_k = tl.load(p_k)
                    b_g = tl.load(p_g).to(tl.float32)
                else:
                    if K % BK:
                        b_k = tl.load(p_k, mask=m_row, other=0.0)
                        b_g = tl.load(p_g, mask=m_row, other=0.0).to(tl.float32)
                    else:
                        b_k = tl.load(p_k, mask=m_s[:, None], other=0.0)
                        b_g = tl.load(p_g, mask=m_s[:, None], other=0.0).to(tl.float32)
                b_dk = b_dk * tl.where(m_s[:, None], exp2(b_gn[None, :] - b_g), 0)
                b_kdk_sum += tl.sum(b_k * b_dk, axis=0)
                p_dk = dk_ptr + row_g + o_k[None, :]
                if ALIGNED:
                    tl.store(p_dk, b_dk.to(dk.dtype.element_ty))
                else:
                    if K % BK:
                        tl.store(p_dk, b_dk.to(dk.dtype.element_ty), mask=m_row)
                    else:
                        tl.store(p_dk, b_dk.to(dk.dtype.element_ty), mask=m_s[:, None])

                # b_dq = do @ h (same slab structure)
                b_dq = tl.zeros([BC, BK], dtype=tl.float32)
                if SINGLE_SLAB:
                    p_do = do_ptr + row_v + o_v[None, :]
                    if ALIGNED:
                        b_do = tl.load(p_do)
                    else:
                        m_do = m_s[:, None]
                        if V % BV:
                            m_do = m_do & m_v[None, :]
                        b_do = tl.load(p_do, mask=m_do, other=0.0)
                    b_dq = tl.dot(b_do, b_h0.to(b_do.dtype), b_dq, allow_tf32=False)
                else:
                    for i_v in range(tl.cdiv(V, BV)):
                        o_vv = i_v * BV + o_v
                        p_do = do_ptr + row_v + o_vv[None, :]
                        if STATE_V_FIRST:
                            p_h = h_ptr + o_vv[:, None] * K + o_k[None, :]
                        else:
                            p_h = h_ptr + o_vv[:, None] + o_k[None, :] * V
                        if ALIGNED:
                            b_do = tl.load(p_do)
                            b_hs = tl.load(p_h)
                        else:
                            m_do = m_s[:, None]
                            if V % BV:
                                m_do = m_do & (o_vv < V)[None, :]
                            m_hvs = (o_vv < V)[:, None]
                            if K % BK:
                                m_hvs = m_hvs & m_k[None, :]
                            b_do = tl.load(p_do, mask=m_do, other=0.0)
                            b_hs = tl.load(p_h, mask=m_hvs, other=0.0)
                        b_dq = tl.dot(b_do, b_hs.to(b_do.dtype), b_dq, allow_tf32=False)

                if ALIGNED:
                    b_q = tl.load(p_q)
                else:
                    if K % BK:
                        b_q = tl.load(p_q, mask=m_row, other=0.0)
                    else:
                        b_q = tl.load(p_q, mask=m_s[:, None], other=0.0)
                b_dq = b_dq * exp2(b_g) * scale
                # The +dgk_total term is applied as a post-loop last-row fixup so no
                # loop-carried accumulator is read inside the sub loop (bishengir
                # SIGABRT otherwise). fp32 c + 0*dgk == c keeps this bitwise-exact.
                b_dg = b_q * b_dq - b_k * b_dk

                p_dq = dq_ptr + row_g + o_k[None, :]
                p_dg = dg_ptr + row_g + o_k[None, :]
                if ALIGNED:
                    tl.store(p_dq, b_dq.to(dq.dtype.element_ty))
                    tl.store(p_dg, b_dg.to(dg.dtype.element_ty))
                else:
                    if K % BK:
                        tl.store(p_dq, b_dq.to(dq.dtype.element_ty), mask=m_row)
                        tl.store(p_dg, b_dg.to(dg.dtype.element_ty), mask=m_row)
                    else:
                        tl.store(p_dq, b_dq.to(dq.dtype.element_ty), mask=m_s[:, None])
                        tl.store(p_dg, b_dg.to(dg.dtype.element_ty), mask=m_s[:, None])

        # last-row fixup: dg[last] += dgk + kdk_sum
        i_last = min(T, i_t * BT + BT) - 1
        p_dgl = dg_ptr + i_last.to(tl.int64) * HV * K + o_k
        if K % BK:
            b_dgl = tl.load(p_dgl, mask=m_k, other=0.0)
        else:
            b_dgl = tl.load(p_dgl)
        b_dgl = b_dgl + (b_dgk + b_kdk_sum)
        if K % BK:
            tl.store(p_dgl, b_dgl.to(dg.dtype.element_ty), mask=m_k)
        else:
            tl.store(p_dgl, b_dgl.to(dg.dtype.element_ty))


@triton.jit(do_not_specialize=['T', 'task_num', 'num_core', 'BH'])
def chunk_kda_bwd_kernel_wy_dw_part_npu(
    k,
    g,
    beta,
    A,
    h,
    dv,
    dA_acc,
    db_acc,
    dg,
    dk,
    cu_seqlens,
    chunk_indices,
    chunk_offsets,
    T,
    BH,
    task_num,
    num_core,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    STATE_V_FIRST: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    G_T_CONTIG: tl.constexpr,
    K_OFFSET: tl.constexpr,
):
    i_k = K_OFFSET
    core_id = tl.program_id(0)
    T_seq = T

    for task_id in tl.range(core_id, task_num, num_core):
        i_t = task_id // BH
        i_bh = task_id % BH
        i_b, i_hv = i_bh // HV, i_bh % HV
        i_h = i_hv // (HV // H)

        if IS_VARLEN:
            i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
            bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
            T = (eos - bos).to(tl.int32)
            i_tg = tl.load(chunk_offsets + i_n).to(tl.int64) + i_t.to(tl.int64)
        else:
            i_tg = tl.cast(i_b, tl.int64) * tl.cdiv(T, BT) + i_t
            bos, eos = tl.cast(i_b, tl.int64) * T, tl.cast(i_b, tl.int64) * T + T

        k_ptr = k + (bos * H + i_h) * K
        k_stride_t = H * K

        # G_T_CONTIG: whether beta/A/dv were host-transposed to [B, H*, T, *]
        # (k/g always arrive native).
        if G_T_CONTIG:
            if IS_VARLEN:
                beta_ptr = beta + tl.cast(i_hv, tl.int64) * T_seq + bos
                A_ptr = A + tl.cast(i_hv, tl.int64) * T_seq * BT + bos * BT
                dv_ptr = dv + tl.cast(i_hv, tl.int64) * T_seq * V + bos * V
            else:
                hv_off = tl.cast(i_b, tl.int64) * HV + i_hv
                beta_ptr = beta + hv_off * T_seq
                A_ptr = A + hv_off * T_seq * BT
                dv_ptr = dv + hv_off * T_seq * V
            a_stride_t = BT
            dv_stride_t = V
            beta_stride = 1
        else:
            beta_ptr = beta + bos * HV + i_hv
            A_ptr = A + (bos * HV + i_hv) * BT
            dv_ptr = dv + (bos * HV + i_hv) * V
            a_stride_t = HV * BT
            dv_stride_t = HV * V
            beta_stride = HV

        # g always arrives in its native [B,T,HV,K] layout (k_part reads it
        # there too); its rows are contiguous enough that no transpose pays.
        g_ptr = g + (bos * HV + i_hv) * K
        g_stride_t = HV * K

        h_ptr = h + (i_tg * HV + i_hv) * K * V
        dA_ptr = dA_acc + (bos * HV + i_hv) * BT
        db_ptr = db_acc + bos * HV + i_hv
        dg_ptr = dg + (bos * HV + i_hv) * K
        dk_ptr = dk + (bos * HV + i_hv) * K

        b_dw = tl.zeros([BT, BK], dtype=tl.float32)
        for i_v in range(tl.cdiv(V, BV)):
            p_dv = tl.make_block_ptr(dv_ptr, (T, V), (dv_stride_t, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            if STATE_V_FIRST:
                p_h = tl.make_block_ptr(h_ptr, (V, K), (K, 1), (i_v * BV, i_k * BK), (BV, BK), (1, 0))
            else:
                p_h = tl.make_block_ptr(h_ptr, (V, K), (1, V), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
            b_dv = tl.load(p_dv, boundary_check=(0, 1))
            b_h = tl.load(p_h, boundary_check=(0, 1))
            b_dw = tl.dot(b_dv, b_h.to(b_dv.dtype), b_dw, allow_tf32=False)
        # Match CUDA: downcast dw to A.dtype before the dA / dkgb GEMMs.
        # Hoisted above the k/g/beta/A loads (bitwise-neutral reorder of
        # independent ops) so the fp32->bf16 fixpipe pass overlaps their MTE
        # staging instead of serializing behind the kg/gb elementwise chain.
        b_dw = -b_dw.to(A.dtype.element_ty)

        p_k = tl.make_block_ptr(k_ptr, (T, K), (k_stride_t, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_g = tl.make_block_ptr(g_ptr, (T, K), (g_stride_t, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_beta = tl.make_block_ptr(beta_ptr, (T,), (beta_stride,), (i_t * BT,), (BT,), (0,))
        p_A = tl.make_block_ptr(A_ptr, (BT, T), (1, a_stride_t), (0, i_t * BT), (BT, BT), (0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_g = tl.load(p_g, boundary_check=(0, 1)).to(tl.float32)
        b_beta = tl.load(p_beta, boundary_check=(0,))
        b_A = tl.load(p_A, boundary_check=(0, 1))
        b_gk_exp = exp2(b_g)
        b_kg = b_k * b_gk_exp
        b_gb = b_gk_exp * b_beta[:, None]
        b_kg_a = b_kg.to(b_A.dtype)
        b_dkgb = tl.dot(b_A, b_dw, allow_tf32=False)

        p_dA_acc = tl.make_block_ptr(dA_ptr, (T, BT), (HV * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
        b_dA = tl.load(p_dA_acc, boundary_check=(0, 1)).to(tl.float32)
        # lhs is dead after this dot and dot1's rhs use does not clobber it
        # (probed), so no defensive copy.
        b_dA = tl.dot(b_dw, tl.trans(b_kg_a), b_dA, allow_tf32=False)
        tl.store(p_dA_acc, b_dA.to(p_dA_acc.dtype.element_ty), boundary_check=(0, 1))

        p_db_acc = tl.make_block_ptr(db_ptr, (T,), (HV,), (i_t * BT,), (BT,), (0,))
        b_db = tl.load(p_db_acc, boundary_check=(0,)).to(tl.float32)
        b_db += tl.sum(b_dkgb * b_kg, 1)
        tl.store(p_db_acc, b_db.to(p_db_acc.dtype.element_ty), boundary_check=(0,))

        p_dk = tl.make_block_ptr(dk_ptr, (T, K), (HV * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_dk = tl.load(p_dk, boundary_check=(0, 1)).to(tl.float32)
        b_dk = b_dk + b_dkgb * b_gb
        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))

        p_dg = tl.make_block_ptr(dg_ptr, (T, K), (HV * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_dg = tl.load(p_dg, boundary_check=(0, 1)).to(tl.float32)
        b_dg = b_dg + b_kg * b_dkgb * b_beta[:, None]
        tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), boundary_check=(0, 1))


@triton.jit(do_not_specialize=['T', 'task_num', 'num_core', 'BH', 'NT_OFFSET'])
def chunk_kda_bwd_kernel_wy_dA_finalize_npu(
    A,
    beta,
    dA_acc,
    db_acc,
    dA,
    db,
    cu_seqlens,
    chunk_indices,
    T,
    BH,
    task_num,
    num_core,
    NT_OFFSET,
    HV: tl.constexpr,
    BT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    G_T_CONTIG: tl.constexpr,
    TAIL_MODE: tl.constexpr,
):
    """dA = mask(-A @ ((mask * dA_acc * beta) @ A)); copy db_acc into db.

    Fuses the old mask kernel into mid+finalize so dA_acc stays in UB.
    TAIL_MODE 0 = aligned bulk (no boundary_check). TAIL_MODE 1 = tail/varlen.
    First tl.dot clobbers masked dA (dead). Second uses b_A as lhs (dead after store).
    """
    core_id = tl.program_id(0)
    T_seq = T

    for task_id in tl.range(core_id, task_num, num_core):
        i_t = NT_OFFSET + task_id // BH
        i_bh = task_id % BH
        i_b, i_hv = i_bh // HV, i_bh % HV

        if IS_VARLEN:
            i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
            bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
            T = (eos - bos).to(tl.int32)
        else:
            bos, eos = tl.cast(i_b, tl.int64) * T, tl.cast(i_b, tl.int64) * T + T

        if G_T_CONTIG:
            if IS_VARLEN:
                A_ptr = A + tl.cast(i_hv, tl.int64) * T_seq * BT + bos * BT
                beta_ptr = beta + tl.cast(i_hv, tl.int64) * T_seq + bos
            else:
                hv_off = tl.cast(i_b, tl.int64) * HV + i_hv
                A_ptr = A + hv_off * T_seq * BT
                beta_ptr = beta + hv_off * T_seq
            a_stride_t = BT
            beta_stride = 1
        else:
            A_ptr = A + (bos * HV + i_hv) * BT
            beta_ptr = beta + bos * HV + i_hv
            a_stride_t = HV * BT
            beta_stride = HV

        dA_acc_ptr = dA_acc + (bos * HV + i_hv) * BT
        db_acc_ptr = db_acc + bos * HV + i_hv
        dA_ptr = dA + (bos * HV + i_hv) * BT
        db_ptr = db + bos * HV + i_hv

        p_A = tl.make_block_ptr(A_ptr, (BT, T), (1, a_stride_t), (0, i_t * BT), (BT, BT), (0, 1))
        p_dA_acc = tl.make_block_ptr(dA_acc_ptr, (T, BT), (HV * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
        p_dA = tl.make_block_ptr(dA_ptr, (T, BT), (HV * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
        p_beta = tl.make_block_ptr(beta_ptr, (T,), (beta_stride,), (i_t * BT,), (BT,), (0,))
        p_db_acc = tl.make_block_ptr(db_acc_ptr, (T,), (HV,), (i_t * BT,), (BT,), (0,))
        p_db = tl.make_block_ptr(db_ptr, (T,), (HV,), (i_t * BT,), (BT,), (0,))

        o_t = i_t * BT + tl.arange(0, BT)
        if TAIL_MODE == 0:
            b_A = tl.load(p_A)
            b_dA = tl.load(p_dA_acc).to(tl.float32)
            b_beta = tl.load(p_beta)
            m_A = o_t[:, None] > o_t[None, :]
        else:
            b_A = tl.load(p_A, boundary_check=(0, 1))
            b_dA = tl.load(p_dA_acc, boundary_check=(0, 1)).to(tl.float32)
            b_beta = tl.load(p_beta, boundary_check=(0,))
            m_t = o_t < T
            m_A = (o_t[:, None] > o_t[None, :]) & (m_t[:, None] & m_t[None, :])

        b_dA = tl.where(m_A, b_dA * b_beta[None, :], 0)
        # mid: (mask * dA_acc * beta) @ A. lhs clobbers b_dA; A is rhs then lhs.
        b_mid = tl.dot(b_dA.to(b_A.dtype), b_A, allow_tf32=False)
        b_fin = tl.dot(b_A, b_mid.to(b_A.dtype), allow_tf32=False)
        b_fin = tl.where(m_A, -b_fin, 0)

        if TAIL_MODE == 0:
            tl.store(p_dA, b_fin.to(p_dA.dtype.element_ty))
            tl.store(p_db, tl.load(p_db_acc).to(p_db.dtype.element_ty))
        else:
            tl.store(p_dA, b_fin.to(p_dA.dtype.element_ty), boundary_check=(0, 1))
            tl.store(p_db, tl.load(p_db_acc, boundary_check=(0,)).to(p_db.dtype.element_ty), boundary_check=(0,))


@triton.jit(do_not_specialize=['T', 'task_num', 'task_num2', 'num_core', 'BH', 'NT_OFFSET'])
def chunk_kda_bwd_kernel_wy_dA_finalize_x2_npu(
    A,
    beta,
    dA_acc,
    db_acc,
    dA,
    db,
    T,
    BH,
    task_num,
    task_num2,
    num_core,
    NT_OFFSET,
    HV: tl.constexpr,
    BT: tl.constexpr,
    G_T_CONTIG: tl.constexpr,
):
    """Batched finalize: two independent (chunk, bh) tasks per loop iteration,
    stage-interleaved (both loads | both vec masks | dot1 x2 | dot2 x2 | stores)
    so the two dependent dot chains overlap and per-task AIV<->AIC segment
    transitions amortize. Dense aligned bulk only (TAIL_MODE=0 semantics);
    the wrapper routes tail/varlen launches to the original kernel.

    Odd task_num tail: p1 is clamped into range for loads/compute (harmless
    duplicate work in the last pair only) and its stores are guarded.
    """
    core_id = tl.program_id(0)
    T_seq = T

    for pair in tl.range(core_id, task_num2, num_core):
        p0 = pair * 2
        p1 = tl.minimum(pair * 2 + 1, task_num - 1)
        has1 = (pair * 2 + 1) < task_num

        i_t0 = NT_OFFSET + p0 // BH
        i_bh0 = p0 % BH
        i_b0, i_hv0 = i_bh0 // HV, i_bh0 % HV
        i_t1 = NT_OFFSET + p1 // BH
        i_bh1 = p1 % BH
        i_b1, i_hv1 = i_bh1 // HV, i_bh1 % HV

        bos0 = tl.cast(i_b0, tl.int64) * T
        bos1 = tl.cast(i_b1, tl.int64) * T

        if G_T_CONTIG:
            hv_off0 = tl.cast(i_b0, tl.int64) * HV + i_hv0
            A_ptr0 = A + hv_off0 * T_seq * BT
            beta_ptr0 = beta + hv_off0 * T_seq
            hv_off1 = tl.cast(i_b1, tl.int64) * HV + i_hv1
            A_ptr1 = A + hv_off1 * T_seq * BT
            beta_ptr1 = beta + hv_off1 * T_seq
            a_stride_t = BT
            beta_stride = 1
        else:
            A_ptr0 = A + (bos0 * HV + i_hv0) * BT
            beta_ptr0 = beta + bos0 * HV + i_hv0
            A_ptr1 = A + (bos1 * HV + i_hv1) * BT
            beta_ptr1 = beta + bos1 * HV + i_hv1
            a_stride_t = HV * BT
            beta_stride = HV

        dA_acc_ptr0 = dA_acc + (bos0 * HV + i_hv0) * BT
        db_acc_ptr0 = db_acc + bos0 * HV + i_hv0
        dA_ptr0 = dA + (bos0 * HV + i_hv0) * BT
        db_ptr0 = db + bos0 * HV + i_hv0
        dA_acc_ptr1 = dA_acc + (bos1 * HV + i_hv1) * BT
        db_acc_ptr1 = db_acc + bos1 * HV + i_hv1
        dA_ptr1 = dA + (bos1 * HV + i_hv1) * BT
        db_ptr1 = db + bos1 * HV + i_hv1

        p_A0 = tl.make_block_ptr(A_ptr0, (BT, T), (1, a_stride_t), (0, i_t0 * BT), (BT, BT), (0, 1))
        p_dA_acc0 = tl.make_block_ptr(dA_acc_ptr0, (T, BT), (HV * BT, 1), (i_t0 * BT, 0), (BT, BT), (1, 0))
        p_dA0 = tl.make_block_ptr(dA_ptr0, (T, BT), (HV * BT, 1), (i_t0 * BT, 0), (BT, BT), (1, 0))
        p_beta0 = tl.make_block_ptr(beta_ptr0, (T,), (beta_stride,), (i_t0 * BT,), (BT,), (0,))
        p_db_acc0 = tl.make_block_ptr(db_acc_ptr0, (T,), (HV,), (i_t0 * BT,), (BT,), (0,))
        p_db0 = tl.make_block_ptr(db_ptr0, (T,), (HV,), (i_t0 * BT,), (BT,), (0,))
        p_A1 = tl.make_block_ptr(A_ptr1, (BT, T), (1, a_stride_t), (0, i_t1 * BT), (BT, BT), (0, 1))
        p_dA_acc1 = tl.make_block_ptr(dA_acc_ptr1, (T, BT), (HV * BT, 1), (i_t1 * BT, 0), (BT, BT), (1, 0))
        p_dA1 = tl.make_block_ptr(dA_ptr1, (T, BT), (HV * BT, 1), (i_t1 * BT, 0), (BT, BT), (1, 0))
        p_beta1 = tl.make_block_ptr(beta_ptr1, (T,), (beta_stride,), (i_t1 * BT,), (BT,), (0,))
        p_db_acc1 = tl.make_block_ptr(db_acc_ptr1, (T,), (HV,), (i_t1 * BT,), (BT,), (0,))
        p_db1 = tl.make_block_ptr(db_ptr1, (T,), (HV,), (i_t1 * BT,), (BT,), (0,))

        o_t0 = i_t0 * BT + tl.arange(0, BT)
        o_t1 = i_t1 * BT + tl.arange(0, BT)
        m_A0 = o_t0[:, None] > o_t0[None, :]
        m_A1 = o_t1[:, None] > o_t1[None, :]

        # stage 1: both tasks' loads batched
        b_A0 = tl.load(p_A0)
        b_dA0 = tl.load(p_dA_acc0).to(tl.float32)
        b_beta0 = tl.load(p_beta0)
        b_db0 = tl.load(p_db_acc0)
        b_A1 = tl.load(p_A1)
        b_dA1 = tl.load(p_dA_acc1).to(tl.float32)
        b_beta1 = tl.load(p_beta1)
        b_db1 = tl.load(p_db_acc1)

        # stage 2: both vec masks
        b_dA0 = tl.where(m_A0, b_dA0 * b_beta0[None, :], 0)
        b_dA1 = tl.where(m_A1, b_dA1 * b_beta1[None, :], 0)

        # stage 3/4: both dot chains
        b_mid0 = tl.dot(b_dA0.to(b_A0.dtype), b_A0, allow_tf32=False)
        b_mid1 = tl.dot(b_dA1.to(b_A1.dtype), b_A1, allow_tf32=False)
        b_fin0 = tl.dot(b_A0, b_mid0.to(b_A0.dtype), allow_tf32=False)
        b_fin1 = tl.dot(b_A1, b_mid1.to(b_A1.dtype), allow_tf32=False)

        # stage 5: both stores
        b_fin0 = tl.where(m_A0, -b_fin0, 0)
        b_fin1 = tl.where(m_A1, -b_fin1, 0)
        tl.store(p_dA0, b_fin0.to(p_dA0.dtype.element_ty))
        tl.store(p_db0, b_db0.to(p_db0.dtype.element_ty))
        if has1:
            tl.store(p_dA1, b_fin1.to(p_dA1.dtype.element_ty))
            tl.store(p_db1, b_db1.to(p_db1.dtype.element_ty))


@input_guard
def chunk_kda_bwd_wy_dqkg_fused_npu(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    v_new: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    h: torch.Tensor,
    do: torch.Tensor,
    dh: torch.Tensor,
    dv: torch.Tensor,
    scale: float | None = None,
    state_v_first: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
    chunk_offsets: torch.LongTensor | None = None,
    use_graph: bool = False,
):
    if use_graph:
        raise NotImplementedError("use_graph is not supported on the Ascend NPU backend")
    B, T, H, K, HV, V = *k.shape, v.shape[2], v.shape[-1]
    BT = chunk_size
    # NSUB floors, so a non-multiple BT would leave the trailing rows unwritten.
    BC = 32 if BT >= 32 else _BC
    if BT % BC != 0:
        raise ValueError(f'KDA Ascend bwd requires chunk_size % {BC} == 0, got {BT}')

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    dq = g.new_empty(B, T, HV, K, dtype=torch.float)
    dk = g.new_empty(B, T, HV, K, dtype=torch.float)
    dv2 = torch.empty_like(v)
    dg = torch.empty_like(g, dtype=torch.float)
    db = torch.empty_like(beta, dtype=torch.float)
    dA = torch.empty_like(A, dtype=torch.float)

    bh_total = B * HV
    num_core = get_npu_properties()['num_vectorcore']
    task_num = NT * bh_total
    is_varlen = cu_seqlens is not None
    if chunk_offsets is None:
        chunk_offsets = prepare_chunk_offsets(cu_seqlens, BT) if is_varlen else g.new_zeros(1, dtype=torch.int64)

    BK = _get_bk(K)
    if BT // BC == 1:
        # BT == BC routes the k_part body through the loop-free straight-line
        # helper (NSUB == 1). bishengir 3.2.2 is unstable on the small-tile
        # dot family there ([BC, BV] x [BV, BK] with BK == 64: SIGABRT in
        # ConvertLinalgRToBinary or silently wrong results, depending on the
        # surrounding IR). Widening the dot N dim to 128 leaves every output
        # column's reduction unchanged and lands the dots
        # in the stable tile family.
        BK = max(BK, 128)
    BV = _get_bv(V)
    NK = triton.cdiv(K, BK)

    v_arg, g_t_contig = _t_contig_arg(v, HV)
    beta_arg = _t_contig_arg(beta, HV)[0]
    A_arg = _t_contig_arg(A, HV)[0]
    dv_arg = _t_contig_arg(dv, HV)[0]

    # v_part fully writes dA_acc/db_acc before any reader, so no zero-init.
    dA_acc = torch.empty(B, T, HV, BT, dtype=torch.float, device=A.device)
    db_acc = torch.empty(B, T, HV, dtype=torch.float, device=beta.device)

    chunk_kda_bwd_kernel_wy_v_part_npu[(num_core,)](
        v=v_arg,
        beta=beta_arg,
        A=A_arg,
        dv=dv_arg,
        dv2=dv2,
        dA_acc=dA_acc,
        db_acc=db_acc,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        BH=bh_total,
        task_num=task_num,
        num_core=num_core,
        HV=HV,
        V=V,
        BT=BT,
        BV=BV,
        IS_VARLEN=is_varlen,
        G_T_CONTIG=g_t_contig,
        **ascend_compile_kwargs(),
    )

    k_part_kwargs = dict(
        q=q,
        k=k,
        v_new=v_new,
        g=g,
        h=h,
        do=do,
        dh=dh,
        dq=dq,
        dk=dk,
        dg=dg,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
        scale=scale,
        T=T,
        BH=bh_total,
        task_num=task_num,
        num_core=num_core,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BT=BT,
        BC=BC,
        BK=BK,
        BV=BV,
        STATE_V_FIRST=state_v_first,
        IS_VARLEN=is_varlen,
        ALIGNED=(not is_varlen) and (T % BT == 0) and (K % BK == 0) and (V % BV == 0),
    )
    for k_off in range(NK):
        k_part_kwargs['K_OFFSET'] = k_off
        chunk_kda_bwd_kernel_wy_k_part_npu[(num_core,)](**k_part_kwargs)

    # g and k stay in their native [B, T, H*, K] layouts; g_t_contig only
    # describes the beta/A/dv + finalize layouts.
    dw_kwargs = dict(
        k=k,
        g=g,
        beta=beta_arg,
        A=A_arg,
        h=h,
        dv=dv_arg,
        dA_acc=dA_acc,
        db_acc=db_acc,
        dg=dg,
        dk=dk,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
        T=T,
        BH=bh_total,
        task_num=task_num,
        num_core=num_core,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
        STATE_V_FIRST=state_v_first,
        IS_VARLEN=is_varlen,
        G_T_CONTIG=g_t_contig,
    )
    for k_off in range(NK):
        dw_kwargs['K_OFFSET'] = k_off
        chunk_kda_bwd_kernel_wy_dw_part_npu[(num_core,)](**dw_kwargs)

    _launch_wy_dA_finalize(
        chunk_kda_bwd_kernel_wy_dA_finalize_npu,
        nt=NT,
        bh_total=bh_total,
        T=T,
        BT=BT,
        is_varlen=is_varlen,
        num_core=num_core,
        kernel_kwargs=dict(
            A=A_arg,
            beta=beta_arg,
            dA_acc=dA_acc,
            db_acc=db_acc,
            dA=dA,
            db=db,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            T=T,
            BH=bh_total,
            HV=HV,
            BT=BT,
            IS_VARLEN=is_varlen,
            G_T_CONTIG=g_t_contig,
        ),
    )

    dv = dv2
    return dq, dk, dv, db, dg, dA
