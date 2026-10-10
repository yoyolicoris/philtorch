"""Sequence alignment: dynamic time warping (DTW), soft-DTW and CTC.

The distances are differentiable to any order, and their gradients with
respect to the cost matrix are the alignments; the CTC loss is soft-DTW
between frames and a target's states. They run only on CUDA GPUs, as Triton
kernels.
"""

from .ctc import ctc_loss, forced_align
from .dtw import dtw, dtw_path, soft_dtw_divergence

__all__ = ["ctc_loss", "dtw", "dtw_path", "forced_align", "soft_dtw_divergence"]
