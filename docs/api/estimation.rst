philtorch.estimation
====================

.. automodule:: philtorch.estimation

.. currentmodule:: philtorch.estimation

Kalman filtering
----------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   kalman_filter
   kalman_smoother

.. autoclass:: KalmanFilterResult
   :members: means, covs, log_likelihood
   :no-inherited-members:

.. autoclass:: KalmanSmootherResult
   :members: means, covs, cross_covs, log_likelihood
   :no-inherited-members:

Hidden Markov models
--------------------

.. note::
   These run only on CUDA GPUs, as Triton kernels: their inputs must be CUDA
   tensors, and Triton must be installed, as it is with PyTorch's CUDA builds
   for Linux.

.. autosummary::
   :toctree: generated
   :nosignatures:

   hmm_filter
   hmm_smoother
   hmm_viterbi
