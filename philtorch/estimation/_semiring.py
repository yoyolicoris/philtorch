"""Triton kernels for batched matrix products in the log and max-plus semirings.

For a of shape (P, I, J) and b of shape (P, J, K):

* ``log_bmm``: c[p, i, k] = logsumexp_j a[p, i, j] + b[p, j, k]
* ``max_bmm``: c[p, i, k] = max_j a[p, i, j] + b[p, j, k]

Each op has a vmap rule that folds the vmapped dimension into P, so torch's
generic associative_scan, which vmaps its combine function, still launches
one kernel per combine.

The forward kernels tile the output and loop over j in tiles, so they need no
(P, I, J, K) temporary. The log product keeps, for every output entry, a
running maximum over j and a sum rescaled by it, as attention kernels do for
the softmax: it is the exact logsumexp, which no shift by row or column
maxima can underflow.

Both products are differentiable to any order. Their derivatives are
weighted contractions

    R = sum over one of i, j, k of W[i, j, k] x[i, j] y[i, k] z[j, k],

with W = exp(a[i, j] + b[j, k] - c[i, k]) for the log product, which lies in
[0, 1], or W = [a[i, j] + b[j, k] = c[i, k]] for the max product, whose
gradient splits each output's among its tied terms.
:func:`weighted_contract` computes one, and its own derivatives are
contractions of the same form, so its backward calls itself. For the max
product, W is piecewise constant, so derivatives through it vanish.

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
def _semiring_bmm_kernel(
    a_ptr, b_ptr, out_ptr, n_p, n_i, n_j, n_k,
    IS_MAX: tl.constexpr, BP: tl.constexpr, BI: tl.constexpr, BK: tl.constexpr, BJ: tl.constexpr,
):  # fmt: skip
    """out[p, i, k] = max or logsumexp over j of a[p, i, j] + b[p, j, k]; contiguous."""
    pid = tl.program_id(0)
    t_i, t_k = tl.cdiv(n_i, BI), tl.cdiv(n_k, BK)
    p = (pid // (t_k * t_i)).to(tl.int64) * BP + tl.arange(0, BP)[:, None, None]
    i = (pid // t_k % t_i) * BI + tl.arange(0, BI)[None, :, None]
    k = (pid % t_k) * BK + tl.arange(0, BK)[None, None, :]
    dtype = a_ptr.dtype.element_ty
    running_max = tl.full([BP, BI, BK], _KERNEL_NEG_INF, dtype)
    running_sum = tl.zeros([BP, BI, BK], dtype)
    for j0 in range(0, n_j, BJ):
        j_col = j0 + tl.arange(0, BJ)[None, None, :]
        j_row = j0 + tl.arange(0, BJ)[None, :, None]
        ij_mask = (p < n_p) & (i < n_i) & (j_col < n_j)
        jk_mask = (p < n_p) & (j_row < n_j) & (k < n_k)
        a = tl.load(a_ptr + (p * n_i + i) * n_j + j_col, mask=ij_mask, other=_KERNEL_NEG_INF)
        b = tl.load(b_ptr + (p * n_j + j_row) * n_k + k, mask=jk_mask, other=_KERNEL_NEG_INF)
        terms = a[:, :, :, None] + b[:, None, :, :]  # [p, i, j, k]
        new_max = tl.maximum(running_max, tl.max(terms, axis=2))
        if not IS_MAX:
            # Shift by 0 while everything so far is -inf, to avoid -inf - -inf.
            shift = tl.where(new_max == _KERNEL_NEG_INF, 0.0, new_max)
            running_sum = running_sum * tl.exp(running_max - shift) + tl.sum(
                tl.exp(terms - shift[:, :, None, :]), axis=2
            )
        running_max = new_max
    out = running_max if IS_MAX else running_max + tl.log(running_sum)
    tl.store(out_ptr + (p * n_i + i) * n_k + k, out, mask=(p < n_p) & (i < n_i) & (k < n_k))


@triton.jit
def _weights(terms, c, IS_MAX: tl.constexpr):
    """W from the terms a + b and the product c; 0 wherever c is -inf."""
    if IS_MAX:
        w = tl.where((terms == c) & (c != _KERNEL_NEG_INF), 1.0, 0.0)
    else:
        w = tl.where(c == _KERNEL_NEG_INF, 0.0, tl.exp(terms - c))
    return w


# In the contraction kernels, block sizes are named by index; the loop runs
# over the summed one. Out-of-range a, b and c load as -inf, which gives W = 0.


@triton.jit
def _contract_over_k_kernel(
    a_ptr, b_ptr, c_ptr, x_ptr, y_ptr, z_ptr, out_ptr, n_p, n_i, n_j, n_k,
    IS_MAX: tl.constexpr, BP: tl.constexpr, BI: tl.constexpr, BJ: tl.constexpr, BK: tl.constexpr,
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
        w = _weights(a[:, :, :, None] + b[:, None, :, :], c[:, :, None, :], IS_MAX)
        acc += tl.sum(w * y[:, :, None, :] * z[:, None, :, :], axis=3)
    x = tl.load(x_ptr + ij, mask=ij_mask, other=0.0)
    tl.store(out_ptr + ij, acc * x, mask=ij_mask)


@triton.jit
def _contract_over_i_kernel(
    a_ptr, b_ptr, c_ptr, x_ptr, y_ptr, z_ptr, out_ptr, n_p, n_i, n_j, n_k,
    IS_MAX: tl.constexpr, BP: tl.constexpr, BJ: tl.constexpr, BK: tl.constexpr, BI: tl.constexpr,
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
        w = _weights(a[:, :, :, None] + b[:, None, :, :], c[:, :, None, :], IS_MAX)
        acc += tl.sum(w * x[:, :, :, None] * y[:, :, None, :], axis=1)
    z = tl.load(z_ptr + jk, mask=jk_mask, other=0.0)
    tl.store(out_ptr + jk, acc * z, mask=jk_mask)


@triton.jit
def _contract_over_j_kernel(
    a_ptr, b_ptr, c_ptr, x_ptr, y_ptr, z_ptr, out_ptr, n_p, n_i, n_j, n_k,
    IS_MAX: tl.constexpr, BP: tl.constexpr, BI: tl.constexpr, BK: tl.constexpr, BJ: tl.constexpr,
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
        w = _weights(a[:, :, :, None] + b[:, None, :, :], c[:, :, None, :], IS_MAX)
        acc += tl.sum(w * x[:, :, :, None] * z[:, None, :, :], axis=2)
    y = tl.load(y_ptr + ik, mask=ik_mask, other=0.0)
    tl.store(out_ptr + ik, acc * y, mask=ik_mask)


# Tuned configurations: (block sizes, in the kernel's order, num_warps,
# num_stages), chosen by the matrix size K. The tuning ran on square matrices
# with K in 2, 4, ..., 128; some of its trees split on P, which was a function
# of K there, and are written here in terms of K.


def _product_config(size: int, is_max: bool):
    """Blocks (p, i, k, j)."""
    if not is_max:
        return ((16, 4, 4, 4), 4, 1) if size <= 32 else ((8, 16, 4, 2), 2, 1)
    if size <= 8:
        return (16, 4, 4, 4), 4, 1
    return ((16, 4, 4, 16), 1, 1) if size <= 32 else ((4, 8, 32, 4), 1, 1)


def _contract_config(over: str, size: int, is_max: bool):
    """Blocks in the kernel's order: the output's indices, then the summed one."""
    if over == "i":  # blocks (p, j, k, i)
        if size <= 8:
            return (16, 2, 4, 4), 2, 1
        if size <= 32 or not is_max:
            return (1, 16, 16, 16), 4, 1
        return (1, 16, 16, 16), 1, 3
    if over == "j":  # blocks (p, i, k, j)
        if size < 8:
            return (16, 4, 2, 1), 4, 1
        return ((16, 4, 4, 4), 4, 1) if size < 16 else ((1, 8, 32, 16), 1, 1)
    # over == "k": blocks (p, i, j, k)
    if size > 32:
        return (1, 64, 16, 1), 2, 1
    if size >= 32 or (size >= 16 and is_max):
        return (1, 16, 16, 4), 2, 1
    return (16, 2, 4, 1), 4, 1


def _product(a: Tensor, b: Tensor, is_max: bool) -> Tensor:
    n_p, n_i, n_j = a.shape
    n_k = b.size(-1)
    out = a.new_empty(n_p, n_i, n_k)
    blocks, num_warps, num_stages = _product_config(n_i, is_max)
    BP, BI, BK, BJ = blocks
    grid = (triton.cdiv(n_p, BP) * triton.cdiv(n_i, BI) * triton.cdiv(n_k, BK),)
    _semiring_bmm_kernel[grid](
        a, b, out, n_p, n_i, n_j, n_k, is_max, BP, BI, BK, BJ,
        num_warps=num_warps, num_stages=num_stages,
    )  # fmt: skip
    return out


def _log_bmm(a: Tensor, b: Tensor) -> Tensor:
    return _product(a, b, False)


def _max_bmm(a: Tensor, b: Tensor) -> Tensor:
    return _product(a, b, True)


def _contract(over: str, a, b, c, x, y, z, is_max: bool) -> Tensor:
    n_p, n_i, n_j = a.shape
    n_k = b.size(-1)
    blocks, num_warps, num_stages = _contract_config(over, n_i, is_max)
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
        a, b, c, x, y, z, out, n_p, n_i, n_j, n_k, is_max, *blocks,
        num_warps=num_warps, num_stages=num_stages,
    )  # fmt: skip
    return out


# Each factor and its index pair; a contraction's output is the pair without
# the summed index.
_PAIRS = {"x": "ij", "y": "ik", "z": "jk"}


def _output_factor(over: str) -> str:
    return next(name for name, pair in _PAIRS.items() if over not in pair)


@torch.library.custom_op("philtorch::semiring_weighted_contract", mutates_args=())
def weighted_contract(
    a: Tensor, b: Tensor, c: Tensor, x: Tensor, y: Tensor, z: Tensor, over: str, is_max: bool
) -> Tensor:
    """Sum W x y z over index ``over`` (one of "i", "j", "k"); see the module docstring."""
    out_like = {"x": a, "y": c, "z": b}[_output_factor(over)]
    if out_like.numel() == 0 or a.size(-1) == 0:
        return torch.zeros_like(out_like)
    args = [t.contiguous() for t in (a, b, c, x, y, z)]
    return _contract(over, *args, is_max)


@weighted_contract.register_fake
def _(a, b, c, x, y, z, over, is_max):
    return torch.empty_like({"x": a, "y": c, "z": b}[_output_factor(over)])


def _contract_setup(ctx, inputs, output):
    a, b, c, x, y, z, over, is_max = inputs
    ctx.over, ctx.is_max = over, is_max
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
        grads[name] = weighted_contract(
            a, b, c, replaced["x"], replaced["y"], replaced["z"], over, ctx.is_max
        )
    if ctx.is_max:
        grad_a, grad_b, grad_c = torch.zeros_like(a), torch.zeros_like(b), torch.zeros_like(c)
    else:
        # d W / d a = W and d W / d b = W, d W / d c = -W.
        grad_a, grad_b, grad_c = grads["x"] * x, grads["z"] * z, -grads["y"] * y
    return grad_a, grad_b, grad_c, grads["x"], grads["y"], grads["z"], None, None


weighted_contract.register_autograd(_contract_backward, setup_context=_contract_setup)


def _fold_vmap_dim(info, in_dims, tensors):
    """Fold each tensor's vmapped dimension into its leading batch dimension P.

    The ops are batched over P already, so a vmapped call is one call on a
    larger batch. An input without a vmapped dimension is repeated. Returns
    the folded tensors and P, which unfolding needs when the vmapped batch is
    empty.
    """
    folded, n_p = [], None
    for t, dim in zip(tensors, in_dims):
        t = t.movedim(dim, 0) if dim is not None else t.expand(info.batch_size, *t.shape)
        n_p = t.size(1)
        folded.append(t.reshape(-1, *t.shape[2:]))
    return folded, n_p


def _unfold_vmap_dim(info, out, n_p):
    return out.reshape(info.batch_size, n_p, *out.shape[1:]), 0


@weighted_contract.register_vmap
def _(info, in_dims, a, b, c, x, y, z, over, is_max):
    folded, n_p = _fold_vmap_dim(info, in_dims[:6], (a, b, c, x, y, z))
    return _unfold_vmap_dim(info, weighted_contract(*folded, over, is_max), n_p)


def _register_product(name: str, kernel, is_max: bool):
    @torch.library.custom_op(f"philtorch::semiring_{name}", mutates_args=())
    def op(a: Tensor, b: Tensor) -> Tensor:
        if a.numel() == 0 or b.numel() == 0:
            fill = _NEG_INF if a.size(-1) == 0 else 0.0
            return a.new_full((a.size(0), a.size(1), b.size(2)), fill)
        return kernel(a.contiguous(), b.contiguous())

    @op.register_fake
    def _(a, b):
        return a.new_empty(a.size(0), a.size(1), b.size(2))

    def setup_context(ctx, inputs, output):
        ctx.save_for_backward(*inputs, output)

    def backward(ctx, grad):
        a, b, c = ctx.saved_tensors
        ones_a, ones_b = torch.ones_like(a), torch.ones_like(b)
        if is_max:
            # Split each output's gradient evenly among the terms that tie for
            # its maximum, as torch.amax does, rather than giving each all of it.
            ties = weighted_contract(a, b, c, ones_a, torch.ones_like(c), ones_b, "j", True)
            grad = torch.where(ties > 0, grad / ties.clamp(min=1), 0.0)
        grad_a = weighted_contract(a, b, c, ones_a, grad, ones_b, "k", is_max)
        grad_b = weighted_contract(a, b, c, ones_a, grad, ones_b, "i", is_max)
        return grad_a, grad_b

    op.register_autograd(backward, setup_context=setup_context)

    @op.register_vmap
    def _(info, in_dims, a, b):
        folded, n_p = _fold_vmap_dim(info, in_dims, (a, b))
        return _unfold_vmap_dim(info, op(*folded), n_p)

    return op


log_bmm = _register_product("log_bmm", _log_bmm, is_max=False)
max_bmm = _register_product("max_bmm", _max_bmm, is_max=True)
