"""Helion kernels for batched matrix products in the log and max-plus semirings.

For a of shape (P, I, J) and b of shape (P, J, K):

* ``log_bmm``: c[p, i, k] = logsumexp_j a[p, i, j] + b[p, j, k]
* ``max_bmm``: c[p, i, k] = max_j a[p, i, j] + b[p, j, k]

The forward kernels tile the output and loop over j in tiles, so they need no
(P, I, J, K) temporary. The log product keeps, for every output entry, a
running maximum over j and a sum rescaled by it, as attention kernels do for
the softmax: it is the exact logsumexp, which no shift by row or column
maxima can underflow.

Both products are differentiable to any order. Their derivatives are
weighted contractions

    R = sum over one of i, j, k of W[i, j, k] x[i, j] y[i, k] z[j, k],

with W = exp(a[i, j] + b[j, k] - c[i, k]) for the log product, which lies in
[0, 1], or W = [a[i, j] + b[j, k] = c[i, k]] for the max product.
:func:`weighted_contract` computes one, and its own derivatives are
contractions of the same form, so its backward calls itself. For the max
product, W is piecewise constant, so derivatives through it vanish.
"""

import os

import helion
import helion.language as hl
import torch
from torch import Tensor

_SETTINGS = {} if os.environ.get("HELION_AUTOTUNE_EFFORT") else {"autotune_effort": "quick"}
_NEG_INF = float("-inf")


@helion.kernel(**_SETTINGS, static_shapes=False)
def _log_bmm_kernel(a: Tensor, b: Tensor) -> Tensor:
    n_p, n_i, n_j = a.shape
    n_k = b.size(-1)
    out = torch.empty([n_p, n_i, n_k], dtype=a.dtype, device=a.device)
    for tile_p, tile_i, tile_k in hl.tile([n_p, n_i, n_k]):
        running_max = hl.full([tile_p, tile_i, tile_k], _NEG_INF, dtype=a.dtype)
        running_sum = hl.zeros([tile_p, tile_i, tile_k], dtype=a.dtype)
        for tile_j in hl.tile(n_j):
            terms = (
                a[tile_p, tile_i, tile_j][:, :, :, None] + b[tile_p, tile_j, tile_k][:, None, :, :]
            )
            new_max = torch.maximum(running_max, torch.amax(terms, dim=2))
            # Shift by 0 while everything so far is -inf, to avoid -inf - -inf.
            shift = torch.where(new_max == _NEG_INF, torch.zeros_like(new_max), new_max)
            running_sum = running_sum * torch.exp(running_max - shift) + torch.sum(
                torch.exp(terms - shift[:, :, None, :]), dim=2
            )
            running_max = new_max
        out[tile_p, tile_i, tile_k] = running_max + torch.log(running_sum)
    return out


@helion.kernel(**_SETTINGS, static_shapes=False)
def _max_bmm_kernel(a: Tensor, b: Tensor) -> Tensor:
    n_p, n_i, n_j = a.shape
    n_k = b.size(-1)
    out = torch.empty([n_p, n_i, n_k], dtype=a.dtype, device=a.device)
    for tile_p, tile_i, tile_k in hl.tile([n_p, n_i, n_k]):
        running_max = hl.full([tile_p, tile_i, tile_k], _NEG_INF, dtype=a.dtype)
        for tile_j in hl.tile(n_j):
            terms = (
                a[tile_p, tile_i, tile_j][:, :, :, None] + b[tile_p, tile_j, tile_k][:, None, :, :]
            )
            running_max = torch.maximum(running_max, torch.amax(terms, dim=2))
        out[tile_p, tile_i, tile_k] = running_max
    return out


# The weights below take tiles indexed [p, i, j, k]. Where c is -inf, every
# term is -inf too and contributes nothing.


@helion.kernel(**_SETTINGS, static_shapes=False)
def _contract_over_k(
    a: Tensor, b: Tensor, c: Tensor, x: Tensor, y: Tensor, z: Tensor, is_max: hl.constexpr
) -> Tensor:
    """R[p, i, j] = x[p, i, j] sum_k W y[p, i, k] z[p, j, k]."""
    n_p, n_i, n_j = a.shape
    n_k = b.size(-1)
    out = torch.empty_like(a)
    for tile_p, tile_i, tile_j in hl.tile([n_p, n_i, n_j]):
        acc = hl.zeros([tile_p, tile_i, tile_j], dtype=a.dtype)
        for tile_k in hl.tile(n_k):
            cc = c[tile_p, tile_i, tile_k][:, :, None, :]
            terms = (
                a[tile_p, tile_i, tile_j][:, :, :, None] + b[tile_p, tile_j, tile_k][:, None, :, :]
            )
            if is_max:
                w = torch.where((terms == cc) & (cc != _NEG_INF), 1.0, 0.0)
            else:
                w = torch.where(cc == _NEG_INF, 0.0, torch.exp(terms - cc))
            acc = acc + torch.sum(
                w
                * y[tile_p, tile_i, tile_k][:, :, None, :]
                * z[tile_p, tile_j, tile_k][:, None, :, :],
                dim=3,
            )
        out[tile_p, tile_i, tile_j] = acc * x[tile_p, tile_i, tile_j]
    return out


@helion.kernel(**_SETTINGS, static_shapes=False)
def _contract_over_i(
    a: Tensor, b: Tensor, c: Tensor, x: Tensor, y: Tensor, z: Tensor, is_max: hl.constexpr
) -> Tensor:
    """R[p, j, k] = z[p, j, k] sum_i W x[p, i, j] y[p, i, k]."""
    n_p, n_i, n_j = a.shape
    n_k = b.size(-1)
    out = torch.empty_like(b)
    for tile_p, tile_j, tile_k in hl.tile([n_p, n_j, n_k]):
        acc = hl.zeros([tile_p, tile_j, tile_k], dtype=b.dtype)
        for tile_i in hl.tile(n_i):
            cc = c[tile_p, tile_i, tile_k][:, :, None, :]
            terms = (
                a[tile_p, tile_i, tile_j][:, :, :, None] + b[tile_p, tile_j, tile_k][:, None, :, :]
            )
            if is_max:
                w = torch.where((terms == cc) & (cc != _NEG_INF), 1.0, 0.0)
            else:
                w = torch.where(cc == _NEG_INF, 0.0, torch.exp(terms - cc))
            acc = acc + torch.sum(
                w
                * x[tile_p, tile_i, tile_j][:, :, :, None]
                * y[tile_p, tile_i, tile_k][:, :, None, :],
                dim=1,
            )
        out[tile_p, tile_j, tile_k] = acc * z[tile_p, tile_j, tile_k]
    return out


@helion.kernel(**_SETTINGS, static_shapes=False)
def _contract_over_j(
    a: Tensor, b: Tensor, c: Tensor, x: Tensor, y: Tensor, z: Tensor, is_max: hl.constexpr
) -> Tensor:
    """R[p, i, k] = y[p, i, k] sum_j W x[p, i, j] z[p, j, k]."""
    n_p, n_i, n_j = a.shape
    n_k = b.size(-1)
    out = torch.empty_like(c)
    for tile_p, tile_i, tile_k in hl.tile([n_p, n_i, n_k]):
        acc = hl.zeros([tile_p, tile_i, tile_k], dtype=c.dtype)
        for tile_j in hl.tile(n_j):
            cc = c[tile_p, tile_i, tile_k][:, :, None, :]
            terms = (
                a[tile_p, tile_i, tile_j][:, :, :, None] + b[tile_p, tile_j, tile_k][:, None, :, :]
            )
            if is_max:
                w = torch.where((terms == cc) & (cc != _NEG_INF), 1.0, 0.0)
            else:
                w = torch.where(cc == _NEG_INF, 0.0, torch.exp(terms - cc))
            acc = acc + torch.sum(
                w
                * x[tile_p, tile_i, tile_j][:, :, :, None]
                * z[tile_p, tile_j, tile_k][:, None, :, :],
                dim=2,
            )
        out[tile_p, tile_i, tile_k] = acc * y[tile_p, tile_i, tile_k]
    return out


_CONTRACT_KERNELS = {"i": _contract_over_i, "j": _contract_over_j, "k": _contract_over_k}
# Each factor and its index pair; a contraction's output is the pair without
# the summed index.
_PAIRS = {"x": "ij", "y": "ik", "z": "jk"}


def _output_factor(over: str) -> str:
    return next(name for name, pair in _PAIRS.items() if over not in pair)


@torch.library.custom_op("philtorch_prototype::weighted_contract", mutates_args=())
def weighted_contract(
    a: Tensor, b: Tensor, c: Tensor, x: Tensor, y: Tensor, z: Tensor, over: str, is_max: bool
) -> Tensor:
    """Sum W x y z over index ``over`` (one of "i", "j", "k"); see the module docstring."""
    out_like = {"x": a, "y": c, "z": b}[_output_factor(over)]
    if out_like.numel() == 0 or a.size(-1) == 0:
        return torch.zeros_like(out_like)
    args = [t.contiguous() for t in (a, b, c, x, y, z)]
    return _CONTRACT_KERNELS[over](*args, hl.constexpr(is_max))


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


def _register_product(name: str, kernel, is_max: bool):
    @torch.library.custom_op(f"philtorch_prototype::{name}", mutates_args=())
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
        grad_a = weighted_contract(a, b, c, ones_a, grad, ones_b, "k", is_max)
        grad_b = weighted_contract(a, b, c, ones_a, grad, ones_b, "i", is_max)
        return grad_a, grad_b

    op.register_autograd(backward, setup_context=setup_context)
    return op


log_bmm = _register_product("log_bmm", _log_bmm_kernel, is_max=False)
max_bmm = _register_product("max_bmm", _max_bmm_kernel, is_max=True)
