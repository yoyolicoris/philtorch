"""Helion versions of philtorch/prototype/_semiring_triton.py's kernels (dev only).

The production kernels are hand-written Triton. These are their Helion
originals, kept to explore and tune new variants: scripts/helion/aot.py tunes
them, and the configurations it finds go into the Triton module by hand. See
that module for what the kernels compute.

Inside a kernel, constants are spelled out (``float("-inf")``): a module
constant would compile to an attribute of this module.
"""

import os

import helion
import helion.language as hl
import torch
from helion.experimental import aot_kernel
from torch import Tensor

# Settings for the tuning (scripts/helion/aot.py). Forked precompile processes
# can fail in a multithreaded parent, so spawn them. The adaptive compile
# timeout carries over from one shape's tuning to the next, so after a small
# shape every configuration of a long one times out: fix the timeout instead.
_SETTINGS = {
    "autotune_precompile": "spawn",
    "autotune_adaptive_timeout": False,
    "autotune_compile_timeout": 120,
} | ({} if os.environ.get("HELION_AUTOTUNE_EFFORT") else {"autotune_effort": "quick"})
# The leading dimension P is a batch: it doesn't choose the configuration.
_PRODUCT_BATCHED = [[0, None, None]] * 2
_CONTRACT_BATCHED = [[0, None, None]] * 6 + [None]
# Seeds for the tuning: Helion's default tiles P by 16 too, and its
# 16^4-element temporaries can take minutes to compile, so a search that
# starts from it alone can find nothing.
_PRODUCT_FALLBACK = helion.Config(block_sizes=[1, 8, 8, 8], num_warps=4)
_CONTRACT_FALLBACK = helion.Config(block_sizes=[1, 16, 16, 16], num_warps=4)


@aot_kernel(batched=_PRODUCT_BATCHED, autotune_seed_configs=_PRODUCT_FALLBACK, **_SETTINGS)
def _log_bmm_kernel(a: Tensor, b: Tensor) -> Tensor:
    n_p, n_i, n_j = a.shape
    n_k = b.size(-1)
    out = torch.empty([n_p, n_i, n_k], dtype=a.dtype, device=a.device)
    for tile_p, tile_i, tile_k in hl.tile([n_p, n_i, n_k]):
        running_max = hl.full([tile_p, tile_i, tile_k], float("-inf"), dtype=a.dtype)
        running_sum = hl.zeros([tile_p, tile_i, tile_k], dtype=a.dtype)
        for tile_j in hl.tile(n_j):
            terms = (
                a[tile_p, tile_i, tile_j][:, :, :, None] + b[tile_p, tile_j, tile_k][:, None, :, :]
            )
            new_max = torch.maximum(running_max, torch.amax(terms, dim=2))
            # Shift by 0 while everything so far is -inf, to avoid -inf - -inf.
            shift = torch.where(new_max == float("-inf"), torch.zeros_like(new_max), new_max)
            running_sum = running_sum * torch.exp(running_max - shift) + torch.sum(
                torch.exp(terms - shift[:, :, None, :]), dim=2
            )
            running_max = new_max
        out[tile_p, tile_i, tile_k] = running_max + torch.log(running_sum)
    return out


@aot_kernel(batched=_PRODUCT_BATCHED, autotune_seed_configs=_PRODUCT_FALLBACK, **_SETTINGS)
def _max_bmm_kernel(a: Tensor, b: Tensor) -> Tensor:
    n_p, n_i, n_j = a.shape
    n_k = b.size(-1)
    out = torch.empty([n_p, n_i, n_k], dtype=a.dtype, device=a.device)
    for tile_p, tile_i, tile_k in hl.tile([n_p, n_i, n_k]):
        running_max = hl.full([tile_p, tile_i, tile_k], float("-inf"), dtype=a.dtype)
        for tile_j in hl.tile(n_j):
            terms = (
                a[tile_p, tile_i, tile_j][:, :, :, None] + b[tile_p, tile_j, tile_k][:, None, :, :]
            )
            running_max = torch.maximum(running_max, torch.amax(terms, dim=2))
        out[tile_p, tile_i, tile_k] = running_max
    return out


# The weights below take tiles indexed [p, i, j, k]. Where c is -inf, every
# term is -inf too and contributes nothing.


@aot_kernel(batched=_CONTRACT_BATCHED, autotune_seed_configs=_CONTRACT_FALLBACK, **_SETTINGS)
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
                w = torch.where((terms == cc) & (cc != float("-inf")), 1.0, 0.0)
            else:
                w = torch.where(cc == float("-inf"), 0.0, torch.exp(terms - cc))
            acc = acc + torch.sum(
                w
                * y[tile_p, tile_i, tile_k][:, :, None, :]
                * z[tile_p, tile_j, tile_k][:, None, :, :],
                dim=3,
            )
        out[tile_p, tile_i, tile_j] = acc * x[tile_p, tile_i, tile_j]
    return out


@aot_kernel(batched=_CONTRACT_BATCHED, autotune_seed_configs=_CONTRACT_FALLBACK, **_SETTINGS)
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
                w = torch.where((terms == cc) & (cc != float("-inf")), 1.0, 0.0)
            else:
                w = torch.where(cc == float("-inf"), 0.0, torch.exp(terms - cc))
            acc = acc + torch.sum(
                w
                * x[tile_p, tile_i, tile_j][:, :, :, None]
                * y[tile_p, tile_i, tile_k][:, :, None, :],
                dim=1,
            )
        out[tile_p, tile_j, tile_k] = acc * z[tile_p, tile_j, tile_k]
    return out


@aot_kernel(batched=_CONTRACT_BATCHED, autotune_seed_configs=_CONTRACT_FALLBACK, **_SETTINGS)
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
                w = torch.where((terms == cc) & (cc != float("-inf")), 1.0, 0.0)
            else:
                w = torch.where(cc == float("-inf"), 0.0, torch.exp(terms - cc))
            acc = acc + torch.sum(
                w
                * x[tile_p, tile_i, tile_j][:, :, :, None]
                * z[tile_p, tile_j, tile_k][:, None, :, :],
                dim=2,
            )
        out[tile_p, tile_i, tile_k] = acc * y[tile_p, tile_i, tile_k]
    return out
