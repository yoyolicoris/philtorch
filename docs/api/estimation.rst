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

.. autosummary::
   :toctree: generated
   :nosignatures:

   hmm_filter
   hmm_smoother
   hmm_viterbi
