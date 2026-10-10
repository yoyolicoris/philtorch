philtorch.align
===============

.. automodule:: philtorch.align

.. currentmodule:: philtorch.align

.. note::
   These run only on CUDA GPUs, as Triton kernels: their inputs must be CUDA
   tensors, and Triton must be installed, as it is with PyTorch's CUDA builds
   for Linux.

Dynamic time warping
--------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   dtw
   dtw_path
   soft_dtw_divergence

Connectionist temporal classification
-------------------------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   ctc_loss
   forced_align
