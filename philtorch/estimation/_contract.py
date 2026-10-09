"""A weighted contraction with exponential weights, in Triton, differentiable in reverse mode.

For a of shape (P, I, J), b of shape (P, J, K) and c of shape (P, I, K),
:func:`weighted_contract` sums

    R = sum over one of i, j, k of W[i, j, k] x[i, j] y[i, k] z[j, k],

with W = exp(a[i, j] + b[j, k] - c[i, k]), computed on the fly so the
(P, I, J, K) weights are never stored. Its derivatives are contractions of
the same form, so its backward calls itself, to any order; it has no
forward-mode or vmap rules.

The HMM chains use it for the gradient of shared transition matrices, where
W is a step's weight and the sum runs over the batch and time.

Every kernel tiles a 4-D index space [p, i, j, k], or a reordering of it. The
tile sizes and warps, by the matrix size, come from tuning these kernels on
an RTX 5060 Ti.
"""

import torch
import triton
import triton.language as tl
from torch import Tensor

_NEG_INF = float("-inf")
# Triton kernels can only read module constants made with tl.constexpr.
_KERNEL_NEG_INF = tl.constexpr(_NEG_INF)


@triton.jit
def _weights(terms, c):
    """W from the terms a + b and c; 0 wherever c is -inf."""
    return tl.where(c == _KERNEL_NEG_INF, 0.0, tl.exp(terms - c))


# In the contraction kernels, block sizes are named by index; the loop runs
# over the summed one. Out-of-range a, b and c load as -inf, which gives W = 0.


@triton.jit
def _contract_over_k_kernel(
    a_ptr, b_ptr, c_ptr, x_ptr, y_ptr, z_ptr, out_ptr, n_p, n_i, n_j, n_k,
    BP: tl.constexpr, BI: tl.constexpr, BJ: tl.constexpr, BK: tl.constexpr,
):  # fmt: skip
    """out[p, i, j] = x[p, i, j] sum_k W y[p, i, k] z[p, j, k]."""
    pid = tl.program_id(0)
    t_i, t_j = tl.cdiv(n_i, BI), tl.cdiv(n_j, BJ)
    p = (pid // (t_j * t_i)).to(tl.int64) * BP + tl.arange(0, BP)[:, None, None]
    i = (pid // t_j % t_i) * BI + tl.arange(0, BI)[None, :, None]
    j_col = (pid % t_j) * BJ + tl.arange(0, BJ)[None, None, :]
    j_row = (pid % t_j) * BJ + tl.arange(0, BJ)[None, :, None]
    ij = (p * n_i + i) * n_j + j_col
    ij_mask = (p < n_p) & (i < n_i) & (j_col < n_j)
    a = tl.load(a_ptr + ij, mask=ij_mask, other=_KERNEL_NEG_INF)
    acc = tl.zeros([BP, BI, BJ], a_ptr.dtype.element_ty)
    for k0 in range(0, n_k, BK):
        k = k0 + tl.arange(0, BK)[None, None, :]
        ik, ik_mask = (p * n_i + i) * n_k + k, (p < n_p) & (i < n_i) & (k < n_k)
        jk, jk_mask = (p * n_j + j_row) * n_k + k, (p < n_p) & (j_row < n_j) & (k < n_k)
        b = tl.load(b_ptr + jk, mask=jk_mask, other=_KERNEL_NEG_INF)
        c = tl.load(c_ptr + ik, mask=ik_mask, other=_KERNEL_NEG_INF)
        y = tl.load(y_ptr + ik, mask=ik_mask, other=0.0)
        z = tl.load(z_ptr + jk, mask=jk_mask, other=0.0)
        w = _weights(a[:, :, :, None] + b[:, None, :, :], c[:, :, None, :])
        acc += tl.sum(w * y[:, :, None, :] * z[:, None, :, :], axis=3)
    x = tl.load(x_ptr + ij, mask=ij_mask, other=0.0)
    tl.store(out_ptr + ij, acc * x, mask=ij_mask)


@triton.jit
def _contract_over_i_kernel(
    a_ptr, b_ptr, c_ptr, x_ptr, y_ptr, z_ptr, out_ptr, n_p, n_i, n_j, n_k,
    BP: tl.constexpr, BJ: tl.constexpr, BK: tl.constexpr, BI: tl.constexpr,
):  # fmt: skip
    """out[p, j, k] = z[p, j, k] sum_i W x[p, i, j] y[p, i, k]."""
    pid = tl.program_id(0)
    t_j, t_k = tl.cdiv(n_j, BJ), tl.cdiv(n_k, BK)
    p = (pid // (t_k * t_j)).to(tl.int64) * BP + tl.arange(0, BP)[:, None, None]
    j_row = (pid // t_k % t_j) * BJ + tl.arange(0, BJ)[None, :, None]
    j_col = (pid // t_k % t_j) * BJ + tl.arange(0, BJ)[None, None, :]
    k = (pid % t_k) * BK + tl.arange(0, BK)[None, None, :]
    jk = (p * n_j + j_row) * n_k + k
    jk_mask = (p < n_p) & (j_row < n_j) & (k < n_k)
    b = tl.load(b_ptr + jk, mask=jk_mask, other=_KERNEL_NEG_INF)
    acc = tl.zeros([BP, BJ, BK], a_ptr.dtype.element_ty)
    for i0 in range(0, n_i, BI):
        i = i0 + tl.arange(0, BI)[None, :, None]
        ij, ij_mask = (p * n_i + i) * n_j + j_col, (p < n_p) & (i < n_i) & (j_col < n_j)
        ik, ik_mask = (p * n_i + i) * n_k + k, (p < n_p) & (i < n_i) & (k < n_k)
        a = tl.load(a_ptr + ij, mask=ij_mask, other=_KERNEL_NEG_INF)
        c = tl.load(c_ptr + ik, mask=ik_mask, other=_KERNEL_NEG_INF)
        x = tl.load(x_ptr + ij, mask=ij_mask, other=0.0)
        y = tl.load(y_ptr + ik, mask=ik_mask, other=0.0)
        w = _weights(a[:, :, :, None] + b[:, None, :, :], c[:, :, None, :])
        acc += tl.sum(w * x[:, :, :, None] * y[:, :, None, :], axis=1)
    z = tl.load(z_ptr + jk, mask=jk_mask, other=0.0)
    tl.store(out_ptr + jk, acc * z, mask=jk_mask)


@triton.jit
def _contract_over_j_kernel(
    a_ptr, b_ptr, c_ptr, x_ptr, y_ptr, z_ptr, out_ptr, n_p, n_i, n_j, n_k,
    BP: tl.constexpr, BI: tl.constexpr, BK: tl.constexpr, BJ: tl.constexpr,
):  # fmt: skip
    """out[p, i, k] = y[p, i, k] sum_j W x[p, i, j] z[p, j, k]."""
    pid = tl.program_id(0)
    t_i, t_k = tl.cdiv(n_i, BI), tl.cdiv(n_k, BK)
    p = (pid // (t_k * t_i)).to(tl.int64) * BP + tl.arange(0, BP)[:, None, None]
    i = (pid // t_k % t_i) * BI + tl.arange(0, BI)[None, :, None]
    k = (pid % t_k) * BK + tl.arange(0, BK)[None, None, :]
    ik = (p * n_i + i) * n_k + k
    ik_mask = (p < n_p) & (i < n_i) & (k < n_k)
    c = tl.load(c_ptr + ik, mask=ik_mask, other=_KERNEL_NEG_INF)
    acc = tl.zeros([BP, BI, BK], a_ptr.dtype.element_ty)
    for j0 in range(0, n_j, BJ):
        j_col = j0 + tl.arange(0, BJ)[None, None, :]
        j_row = j0 + tl.arange(0, BJ)[None, :, None]
        ij, ij_mask = (p * n_i + i) * n_j + j_col, (p < n_p) & (i < n_i) & (j_col < n_j)
        jk, jk_mask = (p * n_j + j_row) * n_k + k, (p < n_p) & (j_row < n_j) & (k < n_k)
        a = tl.load(a_ptr + ij, mask=ij_mask, other=_KERNEL_NEG_INF)
        b = tl.load(b_ptr + jk, mask=jk_mask, other=_KERNEL_NEG_INF)
        x = tl.load(x_ptr + ij, mask=ij_mask, other=0.0)
        z = tl.load(z_ptr + jk, mask=jk_mask, other=0.0)
        w = _weights(a[:, :, :, None] + b[:, None, :, :], c[:, :, None, :])
        acc += tl.sum(w * x[:, :, :, None] * z[:, None, :, :], axis=2)
    y = tl.load(y_ptr + ik, mask=ik_mask, other=0.0)
    tl.store(out_ptr + ik, acc * y, mask=ik_mask)


# Tuned configurations: (block sizes, in the kernel's order, num_warps,
# num_stages), chosen by the matrix size K. The tuning ran on square matrices
# with K in 2, 4, ..., 128; some of its trees split on P, which was a function
# of K there, and are written here in terms of K.


def _contract_config(over: str, size: int):
    """Blocks in the kernel's order: the output's indices, then the summed one."""
    if over == "i":  # blocks (p, j, k, i)
        if size <= 8:
            return (16, 2, 4, 4), 2, 1
        return (1, 16, 16, 16), 4, 1
    if over == "j":  # blocks (p, i, k, j)
        if size < 8:
            return (16, 4, 2, 1), 4, 1
        return ((16, 4, 4, 4), 4, 1) if size < 16 else ((1, 8, 32, 16), 1, 1)
    # over == "k": blocks (p, i, j, k)
    if size > 32:
        return (1, 64, 16, 1), 2, 1
    if size >= 32:
        return (1, 16, 16, 4), 2, 1
    return (16, 2, 4, 1), 4, 1


def _contract(over: str, a, b, c, x, y, z) -> Tensor:
    n_p, n_i, n_j = a.shape
    n_k = b.size(-1)
    blocks, num_warps, num_stages = _contract_config(over, n_i)
    out_shape = {"i": (n_p, n_j, n_k), "j": (n_p, n_i, n_k), "k": (n_p, n_i, n_j)}[over]
    out = a.new_empty(out_shape)
    kernel = {
        "i": _contract_over_i_kernel,
        "j": _contract_over_j_kernel,
        "k": _contract_over_k_kernel,
    }[over]
    # The output's tile counts: (p, then the output's two matrix indices).
    sizes = {"i": (n_j, n_k), "j": (n_i, n_k), "k": (n_i, n_j)}[over]
    grid = (
        triton.cdiv(n_p, blocks[0])
        * triton.cdiv(sizes[0], blocks[1])
        * triton.cdiv(sizes[1], blocks[2]),
    )
    kernel[grid](
        a, b, c, x, y, z, out, n_p, n_i, n_j, n_k, *blocks,
        num_warps=num_warps, num_stages=num_stages,
    )  # fmt: skip
    return out


# Each factor and its index pair; a contraction's output is the pair without
# the summed index.
_PAIRS = {"x": "ij", "y": "ik", "z": "jk"}


def _output_factor(over: str) -> str:
    return next(name for name, pair in _PAIRS.items() if over not in pair)


@torch.library.custom_op("philtorch::weighted_contract", mutates_args=())
def weighted_contract(
    a: Tensor, b: Tensor, c: Tensor, x: Tensor, y: Tensor, z: Tensor, over: str
) -> Tensor:
    """Sum W x y z over index ``over`` (one of "i", "j", "k"); see the module docstring."""
    out_like = {"x": a, "y": c, "z": b}[_output_factor(over)]
    if out_like.numel() == 0 or a.size(-1) == 0:
        return torch.zeros_like(out_like)
    args = [t.contiguous() for t in (a, b, c, x, y, z)]
    return _contract(over, *args)


@weighted_contract.register_fake
def _(a, b, c, x, y, z, over):
    return torch.empty_like({"x": a, "y": c, "z": b}[_output_factor(over)])


def _contract_setup(ctx, inputs, output):
    a, b, c, x, y, z, over = inputs
    ctx.over = over
    ctx.save_for_backward(a, b, c, x, y, z)


def _contract_backward(ctx, grad):
    a, b, c, x, y, z = ctx.saved_tensors
    factors = {"x": x, "y": y, "z": z}
    out_name = _output_factor(ctx.over)
    grads = {}
    for name, pair in _PAIRS.items():
        replaced = dict(factors)
        if name == out_name:
            # The output's own factor: same sum, with the factor replaced by grad.
            replaced[name] = grad
            over = ctx.over
        else:
            # Sum instead over the output index this factor lacks.
            replaced[out_name] = grad * factors[out_name]
            replaced[name] = torch.ones_like(factors[name])
            over = next(index for index in _PAIRS[out_name] if index not in pair)
        grads[name] = weighted_contract(a, b, c, replaced["x"], replaced["y"], replaced["z"], over)
    # d W / d a = W and d W / d b = W, d W / d c = -W.
    grad_a, grad_b, grad_c = grads["x"] * x, grads["z"] * z, -grads["y"] * y
    return grad_a, grad_b, grad_c, grads["x"], grads["y"], grads["z"], None


weighted_contract.register_autograd(_contract_backward, setup_context=_contract_setup)
