"""Sequence alignment: dynamic time warping (DTW) and soft-DTW.

The distances are differentiable to any order, and their gradients with
respect to the cost matrix are the alignments. They run Triton kernels, so
their inputs must be CUDA tensors.
"""

from .dtw import dtw, soft_dtw_divergence

__all__ = ["dtw", "soft_dtw_divergence"]
