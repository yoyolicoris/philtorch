"""A weighted contraction with exponential weights, in Triton, differentiable in reverse mode.

For a of shape (P, I, J), b of shape (P, J, K) and c of shape (P, I, K),
:func:`weighted_contract` sums

    R = sum over one of i, j, k of W[i, j, k] x[i, j] y[i, k] z[j, k],

with W = exp(a[i, j] + b[j, k] + c[i, k]), computed on the fly so the
(P, I, J, K) weights are never stored; any of them may be -inf, for W = 0.
Its derivatives are contractions of the same form, so its backward
calls itself, to any order; it has no forward-mode or vmap rules.

The HMM chains use it for the gradient of shared transition matrices, where
W is a step's weight and the sum runs over the batch and time.

One kernel sums over j, tiling the index space [p, i, j, k]; a sum over i
or k is one over j of permuted inputs.
"""

import torch
import triton
import triton.language as tl
from torch import Tensor

# Triton kernels can only read module constants made with tl.constexpr.
_KERNEL_NEG_INF = tl.constexpr(float("-inf"))


@triton.jit
def _contract_kernel(
    a_ptr, b_ptr, c_ptr, x_ptr, y_ptr, z_ptr, out_ptr, n_p, n_i, n_j, n_k,
    s_ai, s_aj, s_bj, s_bk, s_ci, s_ck,
    BP: tl.constexpr, BI: tl.constexpr, BK: tl.constexpr, BJ: tl.constexpr,
):  # fmt: skip
    """out[p, i, k] = y[p, i, k] sum_j W x[p, i, j] z[p, j, k].

    Each pair of inputs on the same indices, (a, x), (b, z) and (c, y),
    shares the row and column strides given, with its matrices contiguous
    blocks; the output is contiguous. Block sizes are named by index; the
    loop runs over j. Out-of-range a, b and c load as -inf, which gives W = 0.
    """
    pid = tl.program_id(0)
    t_i, t_k = tl.cdiv(n_i, BI), tl.cdiv(n_k, BK)
    p = (pid // (t_k * t_i)).to(tl.int64) * BP + tl.arange(0, BP)[:, None, None]
    i = (pid // t_k % t_i) * BI + tl.arange(0, BI)[None, :, None]
    k = (pid % t_k) * BK + tl.arange(0, BK)[None, None, :]
    ik = p * n_i * n_k + i * s_ci + k * s_ck
    ik_mask = (p < n_p) & (i < n_i) & (k < n_k)
    c = tl.load(c_ptr + ik, mask=ik_mask, other=_KERNEL_NEG_INF)
    acc = tl.zeros([BP, BI, BK], a_ptr.dtype.element_ty)
    for j0 in range(0, n_j, BJ):
        j_col = j0 + tl.arange(0, BJ)[None, None, :]
        j_row = j0 + tl.arange(0, BJ)[None, :, None]
        ij, ij_mask = p * n_i * n_j + i * s_ai + j_col * s_aj, (p < n_p) & (i < n_i) & (j_col < n_j)
        jk, jk_mask = p * n_j * n_k + j_row * s_bj + k * s_bk, (p < n_p) & (j_row < n_j) & (k < n_k)
        a = tl.load(a_ptr + ij, mask=ij_mask, other=_KERNEL_NEG_INF)
        b = tl.load(b_ptr + jk, mask=jk_mask, other=_KERNEL_NEG_INF)
        x = tl.load(x_ptr + ij, mask=ij_mask, other=0.0)
        z = tl.load(z_ptr + jk, mask=jk_mask, other=0.0)
        w = tl.exp(a[:, :, :, None] + b[:, None, :, :] + c[:, :, None, :])
        acc += tl.sum(w * x[:, :, :, None] * z[:, None, :, :], axis=2)
    y = tl.load(y_ptr + ik, mask=ik_mask, other=0.0)
    tl.store(out_ptr + (p * n_i + i) * n_k + k, acc * y, mask=ik_mask)


# Terms per program. With BK <= 32 and BI <= 4, this matched the best blocks of
# a search over the HMM's three shapes, K = 8 to 64, on an RTX 5060 Ti.
_PROGRAM_TERMS = 1024


def _contract_config(n_i: int, n_j: int, n_k: int) -> tuple[int, int, int, int]:
    """Blocks (p, i, k, j) by the problem's shape, for one warp."""
    BK = min(triton.next_power_of_2(n_k), 32)
    BI = min(triton.next_power_of_2(n_i), 4)
    BJ = max(min(triton.next_power_of_2(n_j), _PROGRAM_TERMS // (BI * BK)), 1)
    return max(_PROGRAM_TERMS // (BI * BK * BJ), 1), BI, BK, BJ


def _matrices(t: Tensor, u: Tensor) -> tuple[Tensor, Tensor]:
    """t and u with one layout of contiguous matrices, each row- or column-major."""
    t, u = (v if v.is_contiguous() or v.mT.is_contiguous() else v.contiguous() for v in (t, u))
    if t.stride()[1:] != u.stride()[1:]:
        t, u = t.contiguous(), u.contiguous()
    return t, u


def _contract(a, b, c, x, y, z) -> Tensor:
    """The sum over j, out[p, i, k], reading transposed views in place."""
    (a, x), (b, z), (c, y) = _matrices(a, x), _matrices(b, z), _matrices(c, y)
    n_p, n_i, n_j = a.shape
    n_k = b.size(-1)
    BP, BI, BK, BJ = _contract_config(n_i, n_j, n_k)
    out = a.new_empty(n_p, n_i, n_k)
    grid = (triton.cdiv(n_p, BP) * triton.cdiv(n_i, BI) * triton.cdiv(n_k, BK),)
    strides = (*a.stride()[1:], *b.stride()[1:], *c.stride()[1:])
    _contract_kernel[grid](
        a, b, c, x, y, z, out, n_p, n_i, n_j, n_k, *strides, BP, BI, BK, BJ,
        num_warps=1, num_stages=1,
    )  # fmt: skip
    return out


def _over_j(over: str, a, b, c, x, y, z):
    """The arguments that make a sum over ``over`` one over j.

    The exponent a[i, j] + b[j, k] + c[i, k] keeps its form with the summed
    index in the middle: over k, with a' = c, b' = b^T and c' = a; over i,
    with a' = a^T, b' = c and c' = b, the factors moving with their pairs.
    """
    if over == "k":
        return c, b.mT, a, y, x, z.mT
    if over == "i":
        return a.mT, c, b, x.mT, z, y
    return a, b, c, x, y, z


# Each factor and its index pair; a contraction's output is the pair without
# the summed index.
_PAIRS = {"x": "ij", "y": "ik", "z": "jk"}


def _output_factor(over: str) -> str:
    return next(name for name, pair in _PAIRS.items() if over not in pair)


@torch.library.custom_op("philtorch::weighted_contract", mutates_args=())
def weighted_contract(
    a: Tensor, b: Tensor, c: Tensor, x: Tensor, y: Tensor, z: Tensor, over: str
) -> Tensor:
    """Sum W x y z over index ``over`` (one of "i", "j", "k"), with a nonempty output;
    see the module docstring."""
    return _contract(*_over_j(over, a, b, c, x, y, z))


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
    # d W / d a = d W / d b = d W / d c = W.
    grad_a, grad_b, grad_c = grads["x"] * x, grads["z"] * z, grads["y"] * y
    return grad_a, grad_b, grad_c, grads["x"], grads["y"], grads["z"], None


weighted_contract.register_autograd(_contract_backward, setup_context=_contract_setup)
