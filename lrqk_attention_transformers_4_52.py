import enum
import gc
import io
import logging
import math
import time
import warnings
from collections import defaultdict, namedtuple
from functools import partial
from types import MethodType
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as torchF
import transformers
import transformers.cache_utils as cache_utils
from flash_attn import flash_attn_with_kvcache
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.llama import modeling_llama
from transformers.models.mistral import modeling_mistral
from transformers.models.phi3 import modeling_phi3
from transformers.models.qwen2 import modeling_qwen2
from transformers.models.qwen3 import modeling_qwen3

import cpp_kernel

# Transformers 4.52 removed the per-model FlashAttention2 subclasses and
# _flash_attention_forward helpers. All models now use a single unified
# attention class (e.g. Qwen2Attention) that dispatches via
# ALL_ATTENTION_FUNCTIONS. The ATTENTION_CLASSES dicts are gone.
# GLM is also no longer packaged with transformers as of 4.52.

logger = logging.getLogger(__name__)

torch.set_float32_matmul_precision('high')


def repeat_kv(hidden_state: torch.Tensor, n_rep: int):
    # llama model and qwen2 share the same repeat KV
    return modeling_qwen2.repeat_kv(hidden_state, n_rep)


def _ensure_tuple2(x):
    if isinstance(x, (tuple, list)):
        assert len(x) == 2, "must be a tuple or list of length 2"
        return tuple(x)
    else:
        return (x, x)


def _v1_take_along_dim_with_mask_python(
    src: torch.Tensor,
    indices: torch.Tensor,
    indices_mask: torch.Tensor
) -> torch.Tensor:
    assert src.dim() == 4, "src must be 4D [bsz, kv_heads, seq_len, hdim]"
    assert indices.dim(
    ) == 4, "indices must be 4D [bsz, kv_heads, num_groups, k]"
    assert indices_mask.shape == indices.shape, "mask shape must match indices"
    assert indices_mask.dtype == torch.bool, "mask must be bool tensor"

    coords = indices_mask.nonzero(as_tuple=False)  # [num_valid, 4]

    idx_val = indices[indices_mask].to(src.device)  # [num_valid]
    i = coords[:, 0].to(src.device)
    j = coords[:, 1].to(src.device)
    dst = src[i, j, idx_val, :]  # [num_valid, hdim]

    return dst


@torch.compile(
    fullgraph=False,
    options={
        "epilogue_fusion": True,
        "max_autotune": True,
    }
)
def _take_along_dim_with_mask_python_indices(
    kv_heads: int,
    seq_len: int,
    indices: torch.Tensor,
    indices_mask: torch.Tensor
) -> torch.Tensor:
    coords = indices_mask.nonzero(as_tuple=False)  # [num_valid,4] on GPU
    i = coords[:, 0]
    j = coords[:, 1]
    idx_val = indices[indices_mask]

    linear_indices = (i * kv_heads * seq_len) + \
        (j * seq_len) + idx_val  # [num_valid]
    return linear_indices


def take_along_dim_with_mask_python(
    src: torch.Tensor,
    indices: torch.Tensor,
    indices_mask: torch.Tensor
) -> torch.Tensor:
    linear_indices = _take_along_dim_with_mask_python_indices(
        src.shape[1], src.shape[2], indices, indices_mask)
    linear_indices_cpu = linear_indices.to(src.device)
    flat_src = src.view(-1, src.size(-1))  # [bsz*kv_heads*seq_len, hdim]
    return cpp_kernel.d0_index_select(
        flat_src, linear_indices_cpu).to(indices.device)


def linalg_solve_f32(A: torch.Tensor, B: torch.Tensor, left: bool = True) -> torch.Tensor:
    """
    torch.linalg.solve(A, B, *, left=True, out=None) → Tensor
    Solve AX = B, returns X = A^{-1} B.
    if left=False, solve X A = B => X = B A^{-1}
    Converts to float32 for numerical stability.
    """
    return torch.linalg.solve(A.float(), B.float(), left=left).to(A.dtype)


class InitAQAK(enum.Enum):
    randn = enum.auto()
    top = enum.auto()
    topcol = enum.auto()


def _init_randn_AQ_AK(Q: torch.Tensor, K: torch.Tensor, r: int):
    bsz, num_heads, seq_len, hidden_size = Q.shape
    AQ = torch.randn((bsz, num_heads, seq_len, r), dtype=Q.dtype, device=Q.device)
    AK = torch.randn((bsz, num_heads, seq_len, r), dtype=K.dtype, device=K.device)
    return AQ, AK


def _init_topcol_AQ_AK(Q: torch.Tensor, K: torch.Tensor, r: int):
    bsz, num_heads, seq_len, hidden_size = Q.shape

    _Q_sum = torch.abs(Q).sum(dim=2, keepdim=True)  # [bsz, num_heads, 1, hidden_size]
    _K_sum = torch.abs(K).sum(dim=2, keepdim=True)  # [bsz, num_heads, 1, hidden_size]

    _QK_sum = _Q_sum + _K_sum  # [bsz, num_heads, 1, hidden_size]

    _, indices = torch.topk(_QK_sum, k=r, dim=-1)
    AQ = torch.gather(Q, dim=-1, index=indices.expand(-1, -1, seq_len, -1))
    AK = torch.gather(K, dim=-1, index=indices.expand(-1, -1, seq_len, -1))
    return AQ, AK


def _init_top_AQ_AK(Q: torch.Tensor, K: torch.Tensor, r: int):
    _Q_sum = torch.abs(Q).sum(dim=2, keepdim=True)
    _K_sum = torch.abs(K).sum(dim=2, keepdim=True)

    _, indices_Q = torch.topk(_Q_sum, k=r, dim=-1)
    _, indices_K = torch.topk(_K_sum, k=r, dim=-1)

    AQ = torch.gather(Q, dim=-1, index=indices_Q.expand(-1, -1, Q.shape[2], -1))
    AK = torch.gather(K, dim=-1, index=indices_K.expand(-1, -1, K.shape[2], -1))
    return AQ, AK


def _init_AQ_AK(
    Q: torch.Tensor,
    K: torch.Tensor,
    r: int,
    init_method: Union[InitAQAK, str] = InitAQAK.randn,
):
    if isinstance(init_method, str):
        if init_method not in InitAQAK.__members__:
            raise ValueError(f"Unknown init_method: {init_method}. Current support {list(InitAQAK.__members__.keys())}")
        init_method = InitAQAK[init_method]

    if init_method == InitAQAK.randn:
        return _init_randn_AQ_AK(Q, K, r)
    elif init_method == InitAQAK.top:
        return _init_top_AQ_AK(Q, K, r)
    elif init_method == InitAQAK.topcol:
        return _init_topcol_AQ_AK(Q, K, r)
    else:
        raise ValueError(f"Unknown init_method: {init_method}")


@torch.compile(
    fullgraph=False,
    options={
        "epilogue_fusion": True,
        "max_autotune": True,
    }
)
def _lrqk_prefill_inv(
    Q: torch.Tensor,
    K: torch.Tensor,
    A_Q: torch.Tensor,
    A_K: torch.Tensor,
    lambda_Q: float = 1.0,
    lambda_K: float = 1.0,
    max_iter: int = 8,
    tol: float = 1e-5,
):
    for _ in range(max_iter):
        AQTAQ = torch.matmul(A_Q.transpose(-1, -2), A_Q)
        AKTAK = torch.matmul(A_K.transpose(-1, -2), A_K)

        B_Q = torch.linalg.solve_ex(
            AQTAQ, torch.matmul(A_Q.transpose(-1, -2), Q))[0]
        B_K = torch.linalg.solve_ex(
            AKTAK, torch.matmul(A_K.transpose(-1, -2), K))[0]

        BQBQT = torch.matmul(B_Q, B_Q.transpose(-1, -2))
        RHS_Q = torch.matmul(K.transpose(-1, -2), A_K) + \
            lambda_Q * B_Q.transpose(-1, -2)
        new_A_Q = torch.matmul(
            Q,
            torch.linalg.solve_ex(
                AKTAK + lambda_Q * BQBQT,
                RHS_Q,
                left=False,
            )[0],
        )

        AQTAQ = torch.matmul(A_Q.transpose(-1, -2), A_Q)
        BKBKT = torch.matmul(B_K, B_K.transpose(-1, -2))
        RHS_K = torch.matmul(Q.transpose(-1, -2), A_Q) + \
            lambda_K * B_K.transpose(-1, -2)
        new_A_K = torch.matmul(
            K,
            torch.linalg.solve_ex(
                AQTAQ + lambda_K * BKBKT,
                RHS_K,
                left=False,
            )[0],
        )

        delta = max(
            torchF.mse_loss(A_Q, new_A_Q),
            torchF.mse_loss(A_K, new_A_K)
        )

        A_Q = new_A_Q
        A_K = new_A_K

        if delta < tol:
            break

    AQTAQ = torch.matmul(A_Q.transpose(-1, -2), A_Q)
    AKTAK = torch.matmul(A_K.transpose(-1, -2), A_K)
    B_Q = torch.linalg.solve_ex(
        AQTAQ, torch.matmul(A_Q.transpose(-1, -2), Q))[0]
    B_K = torch.linalg.solve_ex(
        AKTAK, torch.matmul(A_K.transpose(-1, -2), K))[0]

    return A_Q, B_Q, A_K, B_K


@torch.compile(
    fullgraph=False,
    options={
        "epilogue_fusion": True,
        "max_autotune": True,
    }
)
def _lrqk_prefill_inv_w1(
    Q: torch.Tensor,
    K: torch.Tensor,
    A_Q: torch.Tensor,
    A_K: torch.Tensor,
    max_iter: int = 8,
    tol: float = 1e-5,
):
    for _i in range(max_iter):
        AQTAQ = torch.matmul(A_Q.transpose(-1, -2), A_Q)
        AKTAK = torch.matmul(A_K.transpose(-1, -2), A_K)

        B_Q = torch.linalg.solve_ex(
            AQTAQ, torch.matmul(A_Q.transpose(-1, -2), Q))[0]
        B_K = torch.linalg.solve_ex(
            AKTAK, torch.matmul(A_K.transpose(-1, -2), K))[0]

        BQBQT = torch.matmul(B_Q, B_Q.transpose(-1, -2))
        RHS_Q = torch.matmul(K.transpose(-1, -2), A_K) + B_Q.transpose(-1, -2)
        new_A_Q = torch.matmul(
            Q,
            torch.linalg.solve_ex(
                AKTAK + BQBQT,
                RHS_Q,
                left=False,
            )[0],
        )

        AQTAQ = torch.matmul(new_A_Q.transpose(-1, -2), new_A_Q)
        BKBKT = torch.matmul(B_K, B_K.transpose(-1, -2))
        RHS_K = torch.matmul(Q.transpose(-1, -2), new_A_Q) + B_K.transpose(-1, -2)
        new_A_K = torch.matmul(
            K,
            torch.linalg.solve_ex(
                AQTAQ + BKBKT,
                RHS_K,
                left=False,
            )[0],
        )

        delta_AQ = torchF.mse_loss(A_Q, new_A_Q)
        delta_AK = torchF.mse_loss(A_K, new_A_K)
        delta = max(delta_AQ, delta_AK)

        A_Q = new_A_Q
        A_K = new_A_K

        if delta < tol:
            break

    AQTAQ = torch.matmul(A_Q.transpose(-1, -2), A_Q)
    AKTAK = torch.matmul(A_K.transpose(-1, -2), A_K)
    B_Q = torch.linalg.solve_ex(
        AQTAQ, torch.matmul(A_Q.transpose(-1, -2), Q))[0]
    B_K = torch.linalg.solve_ex(
        AKTAK, torch.matmul(A_K.transpose(-1, -2), K))[0]

    return A_Q, B_Q, A_K, B_K


@torch.compile(
    fullgraph=False,
    options={
        "epilogue_fusion": True,
        "max_autotune": True,
    }
)
def _lrqk_decode_inv(
    B_Q: torch.Tensor,
    A_K: torch.Tensor,
    B_K: torch.Tensor,
    K__: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    lambda1: float = 1.0,
    lambda2: float = 1.0,
    max_iter: int = 2,
    tol: float = 1e-3,
):
    max_iter = max(max_iter, 1)

    ktBKT = torch.matmul(k, B_K.transpose(-1, -2))
    BKBKT = torch.matmul(B_K, B_K.transpose(-1, -2))

    qtBQT_qtKTAK = torch.matmul(q, B_Q.transpose(-1, -2)) \
        + lambda2 * \
        torch.matmul(torch.matmul(q, K__.transpose(-1, -2)), A_K)

    BQBQT_AKTAK = torch.matmul(B_Q, B_Q.transpose(-1, -2)) \
        + lambda2 * torch.matmul(A_K.transpose(-1, -2), A_K)

    qtktT = lambda1 * torch.matmul(q, k.transpose(-1, -2))

    hat_k_t = torch.linalg.solve_ex(
        BKBKT, ktBKT, left=False)[0]  # type: torch.Tensor

    for _ in range(max_iter):
        hat_ktTkt = torch.matmul(hat_k_t.transpose(-1, -2), hat_k_t)
        hat_q_t = torch.linalg.solve_ex(
            BQBQT_AKTAK + lambda1 * hat_ktTkt,
            qtBQT_qtKTAK + torch.matmul(qtktT, hat_k_t),
            left=False,
        )[0]  # type: torch.Tensor

        hat_qtTqt = torch.matmul(hat_q_t.transpose(-1, -2), hat_q_t)
        new_hat_k_t = torch.linalg.solve_ex(
            BKBKT + lambda1 * hat_qtTqt,
            ktBKT + torch.matmul(qtktT, hat_q_t),
            left=False,
        )[0]  # type: torch.Tensor

        delta = torchF.mse_loss(hat_k_t, new_hat_k_t)

        hat_k_t = new_hat_k_t

        if delta < tol:
            break

    dBQ = torch.matmul(
        hat_q_t.transpose(-1, -2),
        torch.matmul(hat_q_t, B_Q) - q,
    )

    dBK = torch.matmul(
        hat_k_t.transpose(-1, -2),
        torch.matmul(hat_k_t, B_K) - k,
    )

    lr_q = _lrqk_decode_gd_B_lr(dBQ, hat_q_t)
    lr_k = _lrqk_decode_gd_B_lr(dBK, hat_k_t)

    B_Q = B_Q - lr_q * dBQ
    B_K = B_K - lr_k * dBK

    return B_Q, B_K, hat_q_t, hat_k_t


@torch.compile(
    fullgraph=False,
    options={
        "epilogue_fusion": True,
        "max_autotune": True,
    }
)
def _lrqk_decode_inv_w1(
    B_Q: torch.Tensor,
    A_K: torch.Tensor,
    B_K: torch.Tensor,
    K__: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    max_iter: int = 2,
    tol: float = 1e-3,
):
    max_iter = max(max_iter, 1)

    ktBKT = torch.matmul(k, B_K.transpose(-1, -2))
    BKBKT = torch.matmul(B_K, B_K.transpose(-1, -2))

    qtBQT_qtKTAK = torch.matmul(q, B_Q.transpose(-1, -2)) \
        + torch.matmul(torch.matmul(q, K__.transpose(-1, -2)), A_K)

    BQBQT = torch.matmul(B_Q, B_Q.transpose(-1, -2))
    AKTAK = torch.matmul(A_K.transpose(-1, -2), A_K)
    BQBQT_AKTAK = BQBQT + AKTAK

    qtktT = torch.matmul(q, k.transpose(-1, -2))

    hat_k_t = torch.linalg.solve_ex(
        BKBKT, ktBKT, left=False)[0]  # type: torch.Tensor

    for _i in range(max_iter):
        hat_ktTkt = torch.matmul(hat_k_t.transpose(-1, -2), hat_k_t)
        hat_q_t = torch.linalg.solve_ex(
            BQBQT_AKTAK + hat_ktTkt,
            qtBQT_qtKTAK + torch.matmul(qtktT, hat_k_t),
            left=False,
        )[0]  # type: torch.Tensor

        hat_qtTqt = torch.matmul(hat_q_t.transpose(-1, -2), hat_q_t)
        new_hat_k_t = torch.linalg.solve_ex(
            BKBKT + hat_qtTqt,
            ktBKT + torch.matmul(qtktT, hat_q_t),
            left=False,
        )[0]  # type: torch.Tensor

        delta = torchF.mse_loss(hat_k_t, new_hat_k_t)

        hat_k_t = new_hat_k_t

        if delta < tol:
            break

    dBQ = torch.matmul(
        hat_q_t.transpose(-1, -2),
        torch.matmul(hat_q_t, B_Q) - q,
    )

    dBK = torch.matmul(
        hat_k_t.transpose(-1, -2),
        torch.matmul(hat_k_t, B_K) - k,
    )

    lr_q = _lrqk_decode_gd_B_lr(dBQ, hat_q_t)
    lr_k = _lrqk_decode_gd_B_lr(dBK, hat_k_t)

    B_Q = B_Q - lr_q * dBQ
    B_K = B_K - lr_k * dBK

    return B_Q, B_K, hat_q_t, hat_k_t


def _lrqk_decode_gd_hat_lr(
    dhat: torch.Tensor,
    M: torch.Tensor,
    epsilon: float = 1e-6,
):
    bsz, nhead, _, r = dhat.shape

    dhatM = torch.matmul(dhat, M)

    dhat_norm = torch.square(dhat).sum(dim=(2, 3), keepdim=True)

    dhat_dhatM = torch.matmul(
        dhat.view(bsz, nhead, 1, -1),
        dhatM.view(bsz, nhead, -1, 1),
    )
    return dhat_norm / (dhat_dhatM + epsilon)


def _lrqk_decode_gd_B_lr(
    dB: torch.Tensor,
    hat: torch.Tensor,
    epsilon: float = 1e-6,
):
    dB_norm = torch.square(dB).sum(dim=(2, 3), keepdim=True)
    hat_dB = torch.matmul(hat, dB)
    hat_db_norm = torch.square(hat_dB).sum(dim=(2, 3), keepdim=True)
    return dB_norm / (hat_db_norm + epsilon)


def lrqk_decode_gd(
    B_Q: torch.Tensor,
    A_K: torch.Tensor,
    B_K: torch.Tensor,
    K__: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    lambda1: float = 1.0,
    lambda2: float = 1.0,
    max_iter: int = 8,
    tol: float = 1e-5,
):
    bsz, num_heads, seq_len, r = A_K.shape
    dtype = K__.dtype
    device = K__.device

    hat_q = torch.randn((bsz, num_heads, 1, r), dtype=dtype, device=device)
    hat_k = torch.randn((bsz, num_heads, 1, r), dtype=dtype, device=device)

    wqkT = lambda1 * torch.matmul(q, k.transpose(-1, -2))
    QKT = torch.matmul(q, K__.transpose(-1, -2))
    wqKTAK = lambda2 * torch.matmul(QKT, A_K)

    BQBQT = torch.matmul(B_Q, B_Q.transpose(-1, -2))
    qBQT = torch.matmul(q, B_Q.transpose(-1, -2))
    qBQT__wqKTAK = qBQT + wqKTAK

    kBKT = torch.matmul(k, B_K.transpose(-1, -2))
    BKBKT = torch.matmul(B_K, B_K.transpose(-1, -2))

    for _it in range(max_iter):
        hkthk = torch.matmul(hat_k.transpose(-1, -2), hat_k)
        AKTAK = torch.matmul(A_K.transpose(-1, -2), A_K)
        M = BQBQT + lambda1 * hkthk + lambda2 * AKTAK

        dhat_q = torch.matmul(hat_q, M) - wqkT * hat_k - qBQT__wqKTAK
        lr_dhat_q = _lrqk_decode_gd_hat_lr(dhat_q, M)

        hat_q -= lr_dhat_q * dhat_q

        M = BKBKT + lambda1 * torch.matmul(hat_q.transpose(-1, -2), hat_q)
        dhat_k = torch.matmul(hat_k, M) - wqkT * hat_q - kBKT
        lr_dhat_k = _lrqk_decode_gd_hat_lr(dhat_k, M)

        hat_k -= lr_dhat_k * dhat_k

        if all([
            lr_dhat_q.max() < tol,
            lr_dhat_k.max() < tol,
        ]):
            break

    dB_Q = torch.matmul(
        hat_q.transpose(-1, -2),
        torch.matmul(hat_q, B_Q) - q
    )
    lr_dB_Q = _lrqk_decode_gd_B_lr(dB_Q, hat_q)

    dB_K = torch.matmul(
        hat_k.transpose(-1, -2),
        torch.matmul(hat_k, B_K) - k
    )
    lr_dB_K = _lrqk_decode_gd_B_lr(dB_K, hat_k)

    B_Q -= lr_dB_Q * dB_Q
    B_K -= lr_dB_K * dB_K

    return B_Q, B_K, hat_q, hat_k

#################
# cast methods


def cast_lrqk_prefill(
    Q: torch.Tensor,
    K: torch.Tensor,
    r: int,
    lambda_Q: float = 1.0,
    lambda_K: float = 1.0,
    max_iter: int = 8,
    tol: float = 1e-5,
    init_method: Union[InitAQAK, str] = InitAQAK.randn,
):
    out_dtype = K.dtype
    with torch.autocast("cuda", dtype=torch.float32, enabled=True):
        Q32 = Q.to(dtype=torch.float32)
        K32 = K.to(dtype=torch.float32)
        A_Q, A_K = _init_AQ_AK(Q32, K32, r, init_method=init_method)
        if lambda_Q == 1.0 and lambda_K == 1.0:
            out = _lrqk_prefill_inv_w1(Q32, K32, A_Q, A_K, max_iter, tol)
        else:
            out = _lrqk_prefill_inv(Q32, K32, A_Q, A_K, lambda_Q, lambda_K, max_iter, tol)

    return (o.to(dtype=out_dtype) for o in out)


def cast_lrqk_decode(
    B_Q: torch.Tensor,
    A_K: torch.Tensor,
    B_K: torch.Tensor,
    K__: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    lambda_dec_q: float = 1.0,
    lambda_dec_k: float = 1.0,
    max_iter: int = 8,
    tol: float = 1e-5,
):
    out_dtype = k.dtype
    with torch.autocast("cuda", dtype=torch.float32, enabled=True):
        if lambda_dec_q == 1.0 and lambda_dec_k == 1.0:
            out = _lrqk_decode_inv_w1(
                B_Q.to(dtype=torch.float32),
                A_K.to(dtype=torch.float32),
                B_K.to(dtype=torch.float32),
                K__.to(dtype=torch.float32),
                q.to(dtype=torch.float32),
                k.to(dtype=torch.float32),
                max_iter, tol)
        else:
            out = _lrqk_decode_inv(
                B_Q.to(dtype=torch.float32),
                A_K.to(dtype=torch.float32),
                B_K.to(dtype=torch.float32),
                K__.to(dtype=torch.float32),
                q.to(dtype=torch.float32),
                k.to(dtype=torch.float32),
                lambda_dec_q, lambda_dec_k, max_iter, tol)

    return (o.to(dtype=out_dtype) for o in out)


# Dynamic cache CPU pool


class AutoIncreaseTensor:
    def __init__(
        self,
        capacity: int = 4096,
        dim: int = 0,
        scaling_ratio: float = 1.5,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        pin_memory: bool = False,
    ):
        self.capacity = capacity
        self.dim = dim
        self.scaling_ratio = scaling_ratio
        self.device = device
        self.dtype = dtype
        self.data: Optional[torch.Tensor] = None
        self.current_len = 0
        self.pin_memory = pin_memory

    @torch.inference_mode()
    def append(self, x: torch.Tensor):
        if self.device is None:
            self.device = x.device
        if self.dtype is None:
            self.dtype = x.dtype

        length = x.size(self.dim)
        target_length = self.current_len + length

        if self.data is None:
            init_cap = max(self.capacity, int(length * self.scaling_ratio))
            shape = list(x.shape)
            shape[self.dim] = init_cap
            self.data = torch.empty(
                shape,
                device=self.device,
                dtype=self.dtype,
                pin_memory=self.pin_memory,
            ).zero_()
            self.capacity = init_cap

        elif target_length >= self.capacity:
            new_cap = self.capacity
            while new_cap <= target_length:
                new_cap = int(new_cap * self.scaling_ratio)

            new_shape = list(self.data.shape)
            new_shape[self.dim] = new_cap
            if self.dim == 0:
                self.data = self.data.resize_(new_shape)
                self.data.narrow(self.dim, self.current_len, new_cap - self.current_len).zero_()
            else:
                new_data = torch.empty(
                    new_shape,
                    device=self.device,
                    dtype=self.dtype,
                    pin_memory=self.pin_memory,
                ).zero_()
                new_data.narrow(self.dim, 0, self.current_len).copy_(
                    self.data.narrow(self.dim, 0, self.current_len),
                )
                self.data = new_data
                torch.cuda.empty_cache()

            self.capacity = new_cap

        if self.pin_memory:
            self.data.narrow(self.dim, self.current_len, length).copy_(x)
        else:
            self.data.narrow(self.dim, self.current_len, length).copy_(
                x.to(self.device, self.dtype))
        self.current_len += length

    def has_data(self):
        return self.data is not None

    @property
    def shape(self) -> torch.Size:
        shape = list(self.data.shape)
        shape[self.dim] = self.current_len
        return torch.Size(shape)

    def tensor(self) -> torch.Tensor:
        """Return only the filled portion."""
        return self.data.narrow(self.dim, 0, self.current_len)

    def __len__(self):
        return self.current_len

    def __getitem__(self, idx):
        if self.data is None:
            raise IndexError("Cache is empty")
        return self.data.narrow(self.dim, 0, self.current_len)[idx]

    def take_along_dim(self, indices: torch.Tensor, dim: int = None, out=None) -> torch.Tensor:
        if self.data is None:
            raise IndexError("Cache is empty")
        dim = dim if dim is not None else self.dim
        return torch.take_along_dim(self.data, indices, dim, out=out)


class AutoIncreaseTensorSharedKV(AutoIncreaseTensor):

    def append(self, k: torch.Tensor, v: torch.Tensor):
        x = torch.cat([k, v], dim=-1)  # padding on the last dimension
        return super().append(x)


class PreAllocatedTensor:
    def __init__(self, capacity: int, dim: int, dtype: torch.dtype = None, device: torch.device = None):
        self.capacity = capacity
        self.dim = dim
        self.dtype = dtype
        self.device = device
        self.data = None
        self.current_len = 0

    @classmethod
    def from_tensor(cls, capacity: int, dim: int, data: torch.Tensor):
        ans = cls(capacity, dim, data.dtype, data.device)
        ans.append(data)
        return ans

    def append(self, x: torch.Tensor):
        if self.dtype is None:
            self.dtype = x.dtype

        if self.device is None:
            self.device = x.device

        xlen = x.shape[self.dim]
        if self.data is None:
            shape = list(x.shape)
            shape[self.dim] = self.capacity
            self.data = torch.empty(
                shape, dtype=self.dtype, device=self.device)

        if self.current_len + xlen > self.capacity:
            raise ValueError("Not enough capacity to store the tensor")

        self.data.narrow(self.dim, self.current_len, xlen).copy_(x)
        self.current_len += xlen

    def has_data(self):
        return self.data is not None

    def tensor(self) -> torch.Tensor:
        return self.data.narrow(self.dim, 0, self.current_len)

    def __len__(self):
        return self.current_len

    def __getitem__(self, idx):
        return self.data.narrow(self.dim, 0, self.current_len)[idx]


class _CacheQKV:
    def __init__(self, capacity: int, dim: int, dtype: torch.dtype = None, device: torch.device = None):
        self.Q = PreAllocatedTensor(capacity, dim, dtype, device)
        self.K = PreAllocatedTensor(capacity, dim, dtype, device)
        self.V = PreAllocatedTensor(capacity, dim, dtype, device)

    def append(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor):
        self.Q.append(Q)
        self.K.append(K)
        self.V.append(V)

        return self.Q.tensor(), self.K.tensor(), self.V.tensor()

    def __len__(self):
        return self.Q.current_len


class LightAttentionIndicesFactory:
    def __init__(
        self,
        num_lite_tokens: int = 1,
        attn_topk: int = 128,
        num_key_value_groups: int = 1,
        r: int = 16,
        max_iter: Union[int, tuple[int]] = 2,
        tol: Union[float, tuple[float]] = 1e-4,
        weights: tuple[float] = (1.0, 1.0, 1.0, 1.0),
        capacity: int = 4096,
        init_aq_ak_method: Union[InitAQAK, str] = InitAQAK.randn,
        scaling_ratio: float = 1.5,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        self.num_lite_tokens = num_lite_tokens
        self.attn_topk = attn_topk
        self.dim = 2
        self.num_key_value_groups = num_key_value_groups
        self.r = r
        self.max_iter = _ensure_tuple2(max_iter)
        self.tol = _ensure_tuple2(tol)
        self.init_aq_ak_method = init_aq_ak_method

        self.gpu_capacity = attn_topk + num_lite_tokens

        self.prefill_length = 0
        self.current_len = 0

        if isinstance(weights, float):
            self.weights = (weights, ) * 4
        elif isinstance(weights, tuple):
            assert len(weights) == 4, "weights must be a tuple of length 4"
            self.weights = weights
        else:
            raise ValueError("weights must be a float or a tuple of length 4")

        self.B_Q: Optional[torch.Tensor] = None
        self._A_K = AutoIncreaseTensor(
            capacity=capacity,
            dim=2,
            scaling_ratio=scaling_ratio,
            device=device,
            dtype=dtype,
        )

        self.B_K: Optional[torch.Tensor] = None

        self.KVcpu = AutoIncreaseTensorSharedKV(
            capacity=capacity,
            dim=2,
            scaling_ratio=scaling_ratio,
            device="cpu",
            pin_memory=True,
        )

        self.Kgpu: torch.Tensor = None
        self.Vgpu: torch.Tensor = None

        self.lite_len = 0
        self.lite_indices = [-1] * self.num_lite_tokens

        self.hit_indices: Optional[torch.Tensor] = None

        self.verbose = False

        self.copy_stream = torch.cuda.Stream()

    def __len__(self):
        return self.current_len

    @property
    def Klite(self):
        return self.__get_lite(self.Kgpu)

    @property
    def Vlite(self):
        return self.__get_lite(self.Vgpu)

    def __get_lite(self, src: torch.Tensor):
        nheads = src.shape[1]
        return src[
            :,
            slice(0, nheads, self.num_key_value_groups),
            slice(self.attn_topk, self.attn_topk + self.num_lite_tokens),
            :
        ]

    def __set_lite(self, dst: torch.Tensor, kv_states: torch.Tensor, index: int):
        _s = slice(self.attn_topk + index, self.attn_topk + index + 1)
        dst[:, :, _s, :] = repeat_kv(kv_states, self.num_key_value_groups)

    @property
    def Kgpu_topk_area(self):
        return self.Kgpu.narrow(self.dim, 0, self.attn_topk)

    @property
    def Vgpu_topk_area(self):
        return self.Vgpu.narrow(self.dim, 0, self.attn_topk)

    @property
    def A_K(self):
        return self._A_K.tensor()

    @property
    def A_K_hitted(self):
        if self.hit_indices is None:
            return self._A_K.tensor()

        return self._A_K.take_along_dim(self.hit_indices)

    def _get_A_K_hitted_with_lite(self):
        bsz, nhead, idx_len, _ = self.hit_indices.shape
        hit_indices = torch.cat([
            self.hit_indices,
            (
                torch.tensor(self.lite_indices, device=self.hit_indices.device)
                .reshape(1, 1, -1, 1)
                .expand(bsz, nhead, -1, 1)
            )
        ], dim=2)
        ans = self._A_K.take_along_dim(hit_indices)
        return ans

    def _append(self, key_states: torch.Tensor, value_states: torch.Tensor):
        self.__set_lite(self.Kgpu, key_states, self.lite_len)
        self.__set_lite(self.Vgpu, value_states, self.lite_len)

        self.lite_indices[self.lite_len] = self.current_len
        self.lite_len += 1
        self.current_len += 1

        if self.lite_len >= self.num_lite_tokens:
            self.KVcpu.append(self.Klite, self.Vlite)
            self.lite_len = 0

    @property
    def Kgpu_hitted(self):
        if self.hit_indices is None:
            return self.Kgpu
        hit_len = self.hit_indices.shape[2]
        if hit_len < self.attn_topk:
            return self.Kgpu[:, :, :hit_len, :]
        return self.Kgpu[:, :, 0:self.attn_topk, :]

    def sync_cache(
        self,
        B_Q: torch.Tensor,
        A_K: torch.Tensor,
        B_K: torch.Tensor,
    ):
        self.B_Q = B_Q
        self._A_K.append(A_K)
        self.B_K = B_K
        return self

    def compute_hit_indices(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
    ):
        bsz, nhead, seq_len, _ = key_states.shape
        if seq_len <= self.attn_topk:
            hit_indices = (
                torch.arange(
                    0, seq_len, device=query_states.device,
                    dtype=torch.long)
                .reshape(1, 1, -1, 1)
                .expand(bsz, nhead, -1, 1)
            )
            return hit_indices

        lw_attn = torch.matmul(
            key_states,
            query_states.transpose(-1, -2),
        )  # (bsz, num_heads, seq_len, 1)

        topk_attn = torch.topk(
            lw_attn,
            k=self.attn_topk,
            dim=self.dim,
        ).indices

        hit_indices = torch.sort(topk_attn, dim=self.dim).values
        return hit_indices

    def only_indices_in_cpu(self, now_indices: torch.Tensor, new_indices: torch.Tensor):
        bsz, num_heads, _, _ = now_indices.shape

        now_expanded = now_indices.view(bsz, num_heads, -1, 1)
        new_expanded = new_indices.view(bsz, num_heads, 1, -1)
        match_indices = (now_expanded == new_expanded)

        now_in_new = ~match_indices.any(dim=3)
        new_in_now = ~match_indices.any(dim=2)

        updated_indices = now_indices.clone()
        updated_indices[now_in_new] = new_indices[new_in_now]

        return updated_indices, now_in_new

    def fetch_by_hit(
        self,
        dst: torch.Tensor,
        src: torch.Tensor,
        hit_indices: torch.Tensor,
        dst_mask: Optional[torch.Tensor] = None,
    ):
        bsz, kvheads, seq_len, hdim = src.shape

        num_hitted_indices = hit_indices.shape[2]

        if num_hitted_indices < self.attn_topk:
            dst = dst.narrow(2, 0, num_hitted_indices)

        if src.device == dst.device:
            torch.take_along_dim(
                (
                    src.unsqueeze(2)
                    .expand(bsz, kvheads, self.num_key_value_groups, seq_len, hdim)
                ),
                hit_indices.view(
                    bsz, kvheads, self.num_key_value_groups, -1, 1),
                dim=3,
                out=dst.view(
                    bsz, kvheads, self.num_key_value_groups, -1, hdim),
            )
        else:
            if dst_mask is None:
                out = torch.take_along_dim(
                    (
                        src.unsqueeze(2)
                        .expand(bsz, kvheads, self.num_key_value_groups, seq_len, hdim)
                    ),
                    hit_indices.view(
                        bsz, kvheads, self.num_key_value_groups, -1, 1).cpu(),
                    dim=3,
                ).view(bsz, kvheads * self.num_key_value_groups, -1, hdim)
                dst.copy_(out)
            else:
                _indices = hit_indices.view(
                    bsz, kvheads, self.num_key_value_groups, -1)
                _mask = dst_mask.view(
                    bsz, kvheads, self.num_key_value_groups, -1)
                out = take_along_dim_with_mask_python(src, _indices, _mask)

                dst_mask = dst_mask.view(
                    bsz, kvheads * self.num_key_value_groups, -1)
                dst[dst_mask] = out.to(dst.device)

        return dst

    def prefill(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ):
        bsz, nheads, seq_len, hdim = query_states.shape

        assert seq_len >= self.attn_topk, "seq_len must be larger than attn_topk"

        rep_key_states = modeling_qwen2.repeat_kv(
            key_states, self.num_key_value_groups)

        self.Kgpu = torch.zeros(
            (bsz, nheads, self.gpu_capacity, hdim),
            dtype=query_states.dtype,
            device=query_states.device,
        )
        self.Vgpu = torch.zeros(
            (bsz, nheads, self.gpu_capacity, hdim),
            dtype=query_states.dtype,
            device=query_states.device,
        )

        self.prefill_length = seq_len
        self.current_len = seq_len

        A_Q, B_Q, A_K, B_K = cast_lrqk_prefill(
            query_states,
            rep_key_states,
            r=self.r,
            lambda_Q=self.weights[0],
            lambda_K=self.weights[1],
            max_iter=self.max_iter[0],
            tol=self.tol[0],
            init_method=self.init_aq_ak_method,
        )

        self.sync_cache(B_Q, A_K, B_K)

        if seq_len <= self.gpu_capacity:
            _key = key_states.narrow(2, 0, self.attn_topk)
            _value = value_states.narrow(2, 0, self.attn_topk)
            self.KVcpu.append(_key, _value)

            _remain = seq_len - self.attn_topk

            self.lite_len = _remain

            self.lite_indices = list(range(
                self.attn_topk, self.attn_topk + _remain)) + [-1]*(self.num_lite_tokens - _remain)

            self.Kgpu.narrow(self.dim, 0, seq_len).copy_(rep_key_states)
            self.Vgpu.narrow(self.dim, 0, seq_len).copy_(repeat_kv(
                value_states, self.num_key_value_groups))

            self.hit_indices = (
                torch.arange(0, self.attn_topk, device=query_states.device)
                .view(1, 1, self.attn_topk, 1)
                .expand(bsz, nheads, self.attn_topk, 1)
            )
        else:
            self.KVcpu.append(key_states, value_states)

            hit_indices = self.compute_hit_indices(
                query_states=A_Q[:, :, -1:, :],
                key_states=A_K[:, :, 0:seq_len - self.num_lite_tokens, :],
            )

            self.hit_indices = hit_indices

            self.fetch_by_hit(
                dst=self.Kgpu_topk_area,
                src=key_states,
                hit_indices=hit_indices,
            )

            self.fetch_by_hit(
                dst=self.Vgpu_topk_area,
                src=value_states,
                hit_indices=hit_indices,
            )

            _lite_begin = seq_len - self.num_lite_tokens
            self.lite_indices = list(range(_lite_begin, seq_len))
            self.lite_len = 0
            _s = slice(self.attn_topk, self.attn_topk + self.num_lite_tokens)
            self.Kgpu[:, :, _s, :] = repeat_kv(
                key_states.narrow(2, _lite_begin, self.num_lite_tokens),
                self.num_key_value_groups)

            self.Vgpu[:, :, _s, :] = repeat_kv(
                value_states.narrow(2, _lite_begin, self.num_lite_tokens),
                self.num_key_value_groups)

    def decode(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ):
        self._append(key_states, value_states)

        rep_key_states = modeling_qwen2.repeat_kv(
            key_states,
            self.num_key_value_groups,
        )

        if self.current_len <= self.gpu_capacity:
            B_Q, B_K, hat_q_t, hat_k_t = cast_lrqk_decode(
                B_Q=self.B_Q,
                A_K=self.A_K,
                B_K=self.B_K,
                K__=self.Kgpu.narrow(self.dim, 0, self.current_len-1),
                q=query_states,
                k=rep_key_states,
                lambda_dec_q=self.weights[2],
                lambda_dec_k=self.weights[3],
                max_iter=self.max_iter[1],
                tol=self.tol[1],
            )
            self.sync_cache(B_Q, hat_k_t, B_K)

            return (
                self.Kgpu.narrow(self.dim, 0, self.current_len),
                self.Vgpu.narrow(self.dim, 0, self.current_len),
            )
        else:
            B_Q, B_K, hat_q_t, hat_k_t = cast_lrqk_decode(
                B_Q=self.B_Q,
                A_K=self._get_A_K_hitted_with_lite(),
                B_K=self.B_K,
                K__=self.Kgpu,
                q=query_states,
                k=rep_key_states,
                lambda_dec_q=self.weights[2],
                lambda_dec_k=self.weights[3],
                max_iter=self.max_iter[1],
                tol=self.tol[1],
            )
            self.sync_cache(B_Q, hat_k_t, B_K)

        hit_indices = self.compute_hit_indices(
            query_states=hat_q_t,
            key_states=self.A_K.narrow(
                self.dim,
                0,
                self.lite_indices[self.lite_len] - 1,
            )
        )

        hit_indices, dst_mask = self.only_indices_in_cpu(
            self.hit_indices, hit_indices)
        self.hit_indices = hit_indices

        dstK = self.Kgpu.narrow(self.dim, 0, self.attn_topk)
        dstV = self.Vgpu.narrow(self.dim, 0, self.attn_topk)

        bsz, kvheads, _, hidden_size = key_states.shape

        _indices = hit_indices.view(
            bsz, kvheads, self.num_key_value_groups, -1)
        _mask = dst_mask.view(bsz, kvheads, self.num_key_value_groups, -1)

        out = take_along_dim_with_mask_python(self.KVcpu.data, _indices, _mask)

        if out.numel() > 0:
            _out_k = out.narrow(1, 0, hidden_size)
            _out_v = out.narrow(1, hidden_size, hidden_size)

            dstK[dst_mask] = _out_k
            dstV[dst_mask] = _out_v

        return self.Kgpu, self.Vgpu


class LightAttentionIndicesOffloadPrefill(LightAttentionIndicesFactory):
    """do not cache KV in gpu when prefill"""

    def prefill(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ):
        super().prefill(query_states, key_states, value_states)

        self.stream_temp = torch.cuda.Stream(query_states.device)
        with torch.cuda.stream(self.stream_temp):
            self.Kgpu_temp = self.Kgpu.to("cpu", non_blocking=True)
            self.Vgpu_temp = self.Vgpu.to("cpu", non_blocking=True)
            self._A_K_data_temp = self._A_K.data.to("cpu", non_blocking=True)
            self._A_K.data = None
            self.Kgpu = None
            self.Vgpu = None

    def decode(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ):
        if self.Kgpu_temp is not None:
            with torch.cuda.stream(self.stream_temp):
                self.stream_temp.synchronize()
                self.Kgpu = self.Kgpu_temp.to("cuda", non_blocking=True)
                self.Vgpu = self.Vgpu_temp.to("cuda", non_blocking=True)
                self._A_K.data = self._A_K_data_temp.to(
                    "cuda", non_blocking=True)
                self._A_K_data_temp = None
                self.Kgpu_temp = None
                self.Vgpu_temp = None
                self.stream_temp.synchronize()

        return super().decode(query_states, key_states, value_states)


class LightAttentionIndicesNoHitMiss(LightAttentionIndicesFactory):
    """do not apply hit/miss system"""

    def decode(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ):
        self._append(key_states, value_states)

        rep_key_states = modeling_qwen2.repeat_kv(
            key_states,
            self.num_key_value_groups,
        )

        if self.current_len <= self.gpu_capacity:
            B_Q, B_K, hat_q_t, hat_k_t = cast_lrqk_decode(
                B_Q=self.B_Q,
                A_K=self.A_K,
                B_K=self.B_K,
                K__=self.Kgpu.narrow(self.dim, 0, self.current_len-1),
                q=query_states,
                k=rep_key_states,
                lambda_dec_q=self.weights[2],
                lambda_dec_k=self.weights[3],
                max_iter=self.max_iter[1],
                tol=self.tol[1],
            )
            self.sync_cache(B_Q, hat_k_t, B_K)

            return (
                self.Kgpu.narrow(self.dim, 0, self.current_len),
                self.Vgpu.narrow(self.dim, 0, self.current_len),
            )
        else:
            B_Q, B_K, hat_q_t, hat_k_t = cast_lrqk_decode(
                B_Q=self.B_Q,
                A_K=self._get_A_K_hitted_with_lite(),
                B_K=self.B_K,
                K__=self.Kgpu,
                q=query_states,
                k=rep_key_states,
                lambda_dec_q=self.weights[2],
                lambda_dec_k=self.weights[3],
                max_iter=self.max_iter[1],
                tol=self.tol[1],
            )
            self.sync_cache(B_Q, hat_k_t, B_K)

        hit_indices = self.compute_hit_indices(
            query_states=hat_q_t,
            key_states=self.A_K.narrow(
                self.dim,
                0,
                self.lite_indices[self.lite_len] - 1,
            )
        )
        self.hit_indices = hit_indices

        bsz, kvheads, _, ndim = key_states.shape

        dstK = self.Kgpu.narrow(self.dim, 0, self.attn_topk)
        dstV = self.Vgpu.narrow(self.dim, 0, self.attn_topk)

        dndim = key_states.shape[-1] * 2

        out = torch.take_along_dim(
            (
                self.KVcpu.data.unsqueeze(2)
                .expand(-1, -1, self.num_key_value_groups, -1, -1)
            ),
            hit_indices.view(
                bsz, kvheads, self.num_key_value_groups, -1, 1).cpu(),
            dim=3,
        ).view(bsz, kvheads * self.num_key_value_groups, -1, dndim)

        dstK.copy_(out.narrow(-1, 0, ndim))
        dstV.copy_(out.narrow(-1, ndim, ndim))

        return self.Kgpu, self.Vgpu


class DynamicLRQKCache(cache_utils.Cache):

    class State(enum.Enum):
        pending = 0
        collecting = 1
        decoding = 2

    def __init__(
        self,
        r: int = 16,
        num_active_tokens: int = 512,
        lambda_Q: float = 1.0,
        lambda_K: float = 1.0,
        max_iter: Union[int, tuple[int]] = 2,
        lite_tokens: int = 16,
        tol: Union[float, tuple[float]] = 1e-2,
        max_sequence_length: int = 4096,
        num_key_value_groups: int = 1,
        lwattn_factory=LightAttentionIndicesFactory,
        init_aq_ak_method: Union[InitAQAK, str] = InitAQAK.randn,
    ):
        super().__init__()

        self.num_key_value_groups = num_key_value_groups
        self.r = r
        self.num_active_tokens = num_active_tokens

        self.lambda_Q = lambda_Q
        self.lambda_K = lambda_K
        self.max_iter = _ensure_tuple2(max_iter)
        self.tol = _ensure_tuple2(tol)
        self.lite_tokens = lite_tokens

        self.lwattn = defaultdict(lambda: lwattn_factory(
            num_lite_tokens=lite_tokens,
            attn_topk=num_active_tokens,
            num_key_value_groups=num_key_value_groups,
            r=r,
            max_iter=max_iter,
            tol=tol,
            capacity=max_sequence_length,
            init_aq_ak_method=init_aq_ak_method,
        ))

        self.temp_buff_cache_qkv = defaultdict(
            lambda: _CacheQKV(num_active_tokens, 2))  # type: Dict[int, _CacheQKV]

        self.sequence_state = defaultdict(
            lambda: DynamicLRQKCache.State.pending)

    def __getitem__(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("DynamicLRQKCache does not support indexing")

    def __iter__(self):
        raise NotImplementedError(
            "DynamicLRQKCache does not support iteration")

    def __len__(self):
        """Return the number of layers in the cache."""
        return len(self.sequence_state)

    def get_seq_length(self, layer_idx=0):
        return len(self.lwattn[layer_idx])

    def get_max_cache_shape(self) -> Optional[int]:
        """Returns the maximum sequence length of the cache object."""
        return None

    @property
    def seen_tokens(self):
        """Deprecated compatibility property."""
        return self.get_seq_length()

    def update(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Custom update that takes query_states in addition to key/value states.

        In transformers 4.52, the standard Cache.update signature is
        update(key, value, layer_idx, cache_kwargs). This custom implementation
        uses a different signature intentionally; it is only called from the
        LRQK attention forward methods which pass the right arguments.

        When prefill:
            - query_states: (bsz, num_heads, seq_len, hidden_size)
            - key_states: (bsz, num_key_value_heads, seq_len, hidden_size)
            - value_states: (bsz, num_key_value_heads, seq_len, hidden_size)
        When decode:
            - query_states: (bsz, num_heads, 1, hidden_size)
            - key_states: (bsz, num_key_value_heads, 1, hidden_size)
            - value_states: (bsz, num_key_value_heads, 1, hidden_size)
        """

        if any(c is None for c in (query_states, key_states, value_states)):
            raise ValueError(
                "None of query_states, key_states, and value_states can be None"
            )

        state = self.sequence_state[layer_idx]

        kcache, vcache = None, None
        bsz, _, seq_len, _ = key_states.shape

        # attention_mask: in 4.52 with flash_attention_2, this is typically None
        # or a 4D causal mask. Only apply manual masking for 2D padding masks.

        if state == self.State.pending:
            if attention_mask is not None and attention_mask.dim() == 2:
                key_states = key_states.clone()
                value_states = value_states.clone()
                key_states.mul_(attention_mask.view(bsz, 1, seq_len, 1))
                value_states.mul_(attention_mask.view(bsz, 1, seq_len, 1))

            kcache = key_states
            vcache = value_states

            if seq_len < self.num_active_tokens:
                self.temp_buff_cache_qkv[layer_idx].append(
                    query_states, key_states, value_states)
                self.sequence_state[layer_idx] = self.State.collecting
            else:
                self.lwattn[layer_idx].prefill(
                    query_states=query_states,
                    key_states=key_states,
                    value_states=value_states,
                )
                self.sequence_state[layer_idx] = self.State.decoding

        elif state == self.State.collecting:
            tbqkv = self.temp_buff_cache_qkv[layer_idx]
            cache_len = len(tbqkv)

            if cache_len + seq_len < self.num_active_tokens:
                _, kcache, vcache = self.temp_buff_cache_qkv[layer_idx].append(
                    query_states, key_states, value_states)
            else:
                Q, K, V = self.temp_buff_cache_qkv[layer_idx].append(
                    query_states, key_states, value_states)

                self.lwattn[layer_idx].prefill(
                    query_states=Q,
                    key_states=K,
                    value_states=V,
                )

                kcache = K
                vcache = V

                self.sequence_state[layer_idx] = self.State.decoding

                del self.temp_buff_cache_qkv[layer_idx]

        elif state == self.State.decoding:
            kcache, vcache = self.lwattn[layer_idx].decode(
                query_states,
                key_states,
                value_states,
            )
            torch.cuda.empty_cache()

        return kcache, vcache


class FullOffloadCache(cache_utils.Cache):
    def __init__(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("OffloadedCache can only be used with a GPU")
        super().__init__()

        self.KVcpu = defaultdict(lambda: AutoIncreaseTensorSharedKV(
            capacity=4096,
            dim=2,
            device="cpu",
            pin_memory=True,
        ))

    def __len__(self):
        return len(self.KVcpu)

    def get_seq_length(self, layer_idx=0):
        return len(self.KVcpu[layer_idx])

    def get_max_cache_shape(self) -> Optional[int]:
        return None

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        _kvcpu = self.KVcpu[layer_idx]
        _kvcpu.append(key_states, value_states)

        hdim = key_states.shape[-1]
        kvcuda = _kvcpu.tensor()

        kstate = kvcuda.narrow(3, 0, hdim).to(key_states.device)
        vstate = kvcuda.narrow(3, hdim, hdim).to(value_states.device)
        return kstate, vstate


# wrapper for time estimate


class WrapForwardTimer:
    def __init__(self, module: nn.Module):
        self.module = module
        self.prefill_begin = None
        self.prefill_end = None
        self.decode_begin = None
        self.decode_time_list = []
        self.seq_len = 0
        self.original_method = None

    def prefill_tps(self):
        if self.prefill_begin is None:
            return 0.0
        else:
            return self.seq_len / (self.prefill_end - self.prefill_begin)

    def decode_tps(self):
        if len(self.decode_time_list) == 0:
            return 0.0
        else:
            return len(self.decode_time_list) / sum(self.decode_time_list)

    def remove_wrapper(self):
        setattr(self.module, "forward", self.original_method)
        del self.original_method
        setattr(self.module, "manual_time_wrapped", False)

    def reset(self):
        self.prefill_begin = None
        self.prefill_end = None
        self.decode_begin = None
        self.decode_time_list.clear()
        self.seq_len = 0

    @classmethod
    def wrap(cls, module: nn.Module):
        wraper = cls(module)
        is_manual_time_wrapped = getattr(module, "manual_time_wrapped", False)

        if is_manual_time_wrapped:
            return wraper, wraper.module

        original_method = getattr(module, "forward")
        wraper.original_method = original_method
        setattr(wraper.module, "manual_time_wrapped", True)

        def __new_method(self, *args, **kwargs):
            if wraper.prefill_begin is None:
                input_ids = kwargs.get("input_ids", None)
                if input_ids is None:
                    raise ValueError("input_ids is None")
                wraper.seq_len = input_ids.shape[1]

                input_ids = None
                wraper.prefill_begin = time.time()
            else:
                wraper.decode_begin = time.time()

            result = original_method(*args, **kwargs)

            if wraper.prefill_end is None:
                wraper.prefill_end = time.time()
            else:
                wraper.decode_time_list.append(
                    time.time() - wraper.decode_begin)

            return result

        setattr(wraper.module, "forward", __new_method.__get__(wraper.module))
        return wraper, wraper.module


# ---------------------------------------------------------------------------
# Transformers 4.52 compatible LRQK attention classes
#
# In 4.52, per-model FlashAttention2 subclasses (Qwen2FlashAttention2 etc.)
# and the ATTENTION_CLASSES dicts were removed. Each model now has a single
# unified attention class that dispatches via ALL_ATTENTION_FUNCTIONS.
#
# Our LRQK classes subclass the unified attention class for each model and
# override forward() with the new 4.52 signature:
#   (hidden_states, position_embeddings, attention_mask, past_key_value,
#    cache_position, **kwargs) -> (attn_output, attn_weights)   # 2-tuple!
#
# position_embeddings is always provided in 4.52 (computed once per layer
# stack in the model's forward), so the fallback position_ids path is gone.
# ---------------------------------------------------------------------------


class LRQK_Qwen2Attention(modeling_qwen2.Qwen2Attention):

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[cache_utils.Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(
            bsz, q_len, -1, self.head_dim).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, -1, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = modeling_qwen2.apply_rotary_pos_emb(
            query_states, key_states, cos, sin)

        key_states, value_states = past_key_value.update(
            query_states, key_states, value_states, self.layer_idx,
            attention_mask=attention_mask)

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            modeling_qwen2.logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        # flash_attn_with_kvcache expects (batch, seqlen, nheads, headdim)
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        attn_output = flash_attn_with_kvcache(
            query_states, key_states, value_states, causal=True)

        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        return attn_output, None


class LRQK_Qwen3Attention(modeling_qwen3.Qwen3Attention):

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[cache_utils.Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(
            bsz, q_len, -1, self.head_dim).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, -1, self.head_dim).transpose(1, 2)

        # Qwen3-specific: per-head QK normalization applied before RoPE
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        cos, sin = position_embeddings
        query_states, key_states = modeling_qwen3.apply_rotary_pos_emb(
            query_states, key_states, cos, sin)

        key_states, value_states = past_key_value.update(
            query_states, key_states, value_states, self.layer_idx,
            attention_mask=attention_mask)

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            modeling_qwen3.logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        # flash_attn_with_kvcache expects (batch, seqlen, nheads, headdim)
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        attn_output = flash_attn_with_kvcache(
            query_states, key_states, value_states, causal=True)

        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        return attn_output, None


class LRQK_LlamaAttention(modeling_llama.LlamaAttention):

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[cache_utils.Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(
            bsz, q_len, self.config.num_attention_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.config.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.config.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = modeling_llama.apply_rotary_pos_emb(
            query_states, key_states, cos, sin)

        key_states, value_states = past_key_value.update(
            query_states, key_states, value_states, self.layer_idx,
            attention_mask=attention_mask)

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        # flash_attn_with_kvcache expects (batch, seqlen, nheads, headdim)
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        attn_output = flash_attn_with_kvcache(
            query_states, key_states, value_states, causal=True)

        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        return attn_output, None


class LRQK_MistralAttention(modeling_mistral.MistralAttention):

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[cache_utils.Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(
            bsz, q_len, -1, self.head_dim).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, -1, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = modeling_mistral.apply_rotary_pos_emb(
            query_states, key_states, cos, sin)

        key_states, value_states = past_key_value.update(
            query_states, key_states, value_states, self.layer_idx,
            attention_mask=attention_mask,
        )

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        # flash_attn_with_kvcache expects (batch, seqlen, nheads, headdim)
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        attn_output = flash_attn_with_kvcache(
            query_states, key_states, value_states, causal=True)

        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        return attn_output, None


class LRQK_Phi3Attention(modeling_phi3.Phi3Attention):

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[cache_utils.Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, q_len, _ = hidden_states.size()

        qkv = self.qkv_proj(hidden_states)
        query_pos = self.config.num_attention_heads * self.head_dim
        query_states = qkv[..., :query_pos]
        key_states = qkv[..., query_pos: query_pos +
                         self.num_key_value_heads * self.head_dim]
        value_states = qkv[..., query_pos +
                           self.num_key_value_heads * self.head_dim:]

        query_states = query_states.view(
            bsz, q_len, self.config.num_attention_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # position_embeddings are always pre-computed in 4.52
        cos, sin = position_embeddings
        query_states, key_states = modeling_phi3.apply_rotary_pos_emb(
            query_states, key_states, cos, sin)

        key_states, value_states = past_key_value.update(
            query_states, key_states, value_states, self.layer_idx,
            attention_mask=attention_mask,
        )

        if query_states.dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.qkv_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        # flash_attn_with_kvcache expects (batch, seqlen, nheads, headdim)
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        attn_output = flash_attn_with_kvcache(
            query_states, key_states, value_states, causal=True)

        attn_output = attn_output.reshape(
            bsz, q_len, self.config.hidden_size).contiguous()
        attn_output = self.o_proj(attn_output)

        return attn_output, None


def load_model_hack(
    model_name: str,
    device=None,
    yarn=None,
    base_model: Optional[transformers.PreTrainedModel] = None,
) -> transformers.PreTrainedModel:
    """
    Load the model with self_attn.forward replaced for LRQK custom caching.

    In transformers 4.52, per-model ATTENTION_CLASSES dicts are gone. We load
    the model normally and then patch each layer's self_attn.forward directly.

    Args:
    - model_name: the model name
    - device: the device to load the model
    - yarn: additional arguments passed to from_pretrained
    - base_model: if provided, use this model instead of loading from hub
    """
    mn = model_name.lower()
    _flash_new = None

    if "qwen3" in mn:
        _flash_new = LRQK_Qwen3Attention
    elif "qwen" in mn:
        _flash_new = LRQK_Qwen2Attention
    elif "llama" in mn:
        _flash_new = LRQK_LlamaAttention
    elif "mistral" in mn:
        _flash_new = LRQK_MistralAttention
    elif "phi-3" in mn or "phi3" in mn:
        _flash_new = LRQK_Phi3Attention
    elif "phi" in mn:
        _flash_new = None
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    if yarn is None:
        yarn = {}

    if base_model is not None:
        model = base_model
    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_name,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto" if device is None else device,
            **yarn,
        )

    if _flash_new is not None:
        # Patch each layer's self_attn.forward with the LRQK custom forward.
        # The bound method uses only attributes present on the base class
        # (e.g. Qwen2Attention), so this works even though self_attn is not
        # an instance of our LRQK subclass.
        for layer in model.model.layers:
            if not hasattr(layer, "self_attn"):
                continue
            layer.self_attn.forward = _flash_new.forward.__get__(
                layer.self_attn)

    model = model.eval()

    return model


def load_tokenizer(model_name: str, max_length=None) -> transformers.PreTrainedTokenizer:
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name,
        padding_side="left",
        max_length=max_length,
    )  # type: transformers.PreTrainedTokenizer

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def load_lrqk_model(
    model_name: str,
    max_length=None,
    device=None,
    yarn=None,
    base_model: Optional[transformers.PreTrainedModel] = None,
) -> Tuple[transformers.PreTrainedModel, transformers.PreTrainedTokenizer]:
    """Not thread safe"""
    model = load_model_hack(model_name, device, yarn=yarn, base_model=base_model)
    tokenizer = load_tokenizer(model_name, max_length)
    return model, tokenizer


def load_model(
    model_name: str,
    lrqk=True,
    max_length=None,
    device=None,
    yarn=None,
    base_model: Optional[transformers.PreTrainedModel] = None,
) -> Tuple[transformers.PreTrainedModel, transformers.PreTrainedTokenizer]:
    """Not thread safe"""
    if yarn is None:
        yarn = {}

    if device is None:
        warnings.warn(
            "When device is None, we will use device_map=auto."
            "This behaviour is not tested under our setting, and there could be some unexpected behaviours."
            "We only test on single GPU."
            "It is recommended to specify one device explicitly."
        )

    if lrqk:
        model = load_model_hack(model_name, device, yarn=yarn, base_model=base_model)
    elif base_model is not None:
        model = base_model
    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
            device_map="auto" if device is None else device,
            **yarn,
        )

    model = model.eval()

    tokenizer = load_tokenizer(model_name, max_length)
    return model, tokenizer
