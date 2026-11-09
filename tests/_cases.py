"""Test-case generators shared by the test modules."""

import numpy as np
import torch


def each_choice(*factors):
    """Cases in which every option of every factor appears at least once.

    Case i takes option i of each factor, cycling the shorter factors, so there
    are as many cases as the longest factor has options. It covers every branch
    that depends on one factor alone, as a shape check of independent inputs
    does, without the factors' full product.
    """
    n = max(len(factor) for factor in factors)
    return [tuple(factor[i % len(factor)] for factor in factors) for i in range(n)]


def stable_denominators(order: int, B: int) -> np.ndarray:
    """B random stable denominators of the given order, a (B, order) array.

    Each row has an absolute sum of 1.
    """
    a = np.random.randn(B, order)
    return a / np.abs(a).sum(axis=-1, keepdims=True)


def time_varying_coeffs(B: int, T: int, num_order: int, den_order: int):
    """Random stable time-varying coefficients b and a.

    b is (B, T, num_order + 1) and a (B, T, den_order), each step's a of
    absolute sum at most 1.
    """
    b = torch.randn(B, T, num_order + 1)
    a = torch.randn(B, T, den_order)
    a = a / a.abs().sum(dim=-1, keepdim=True).clamp(min=1e-6)
    return b, a
