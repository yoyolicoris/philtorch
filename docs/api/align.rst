philtorch.align
===============

.. automodule:: philtorch.align

.. currentmodule:: philtorch.align

Dynamic time warping
--------------------

.. note::
   These run only on CUDA GPUs, as Triton kernels: their inputs must be CUDA
   tensors, and Triton must be installed, as it is with PyTorch's CUDA builds
   for Linux.

.. autosummary::
   :toctree: generated
   :nosignatures:

   dtw
   soft_dtw_divergence
