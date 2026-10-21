"""Expectation-maximization over the steps of any model."""

from collections.abc import Callable

import torch
from torch import Tensor


def em(
    step: Callable[..., tuple[Tensor, tuple]],
    params: tuple,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> tuple[tuple, Tensor]:
    r"""Fit a model by expectation-maximization, repeating one of its steps.

    Each call ``step(*params)`` runs an E-step and an M-step and returns the
    log-likelihood of :attr:`params`, of shape :math:`(B)`, and the new
    parameters, as :func:`kalman_em_step` does and the first two results of
    :func:`hmm_em_step` are. The iterations stop after :attr:`max_iter` steps,
    or once the log-likelihood summed over the batch rises by less than
    :attr:`tol`.

    Args:
        step (Callable): one step of expectation-maximization.
        params (tuple): the initial parameters.
        max_iter (int, optional): the most steps to run. Default: 100.
        tol (float, optional): the least rise of the log-likelihood to go on.
            Default: 1e-6.

    Returns:
        tuple: the last parameters, and the summed log-likelihood of the
        parameters each step started from, of shape :math:`(\text{steps})`.

    Raises:
        ValueError: if :attr:`max_iter` is less than 1.

    Example::

        >>> from functools import partial
        >>> from philtorch.estimation import em, kalman_em_step
        >>> y = torch.randn(4, 50, 1, dtype=torch.float64)
        >>> one = torch.ones(1, 1, dtype=torch.float64)
        >>> m0 = torch.zeros(1, dtype=torch.float64)
        >>> params, history = em(
        ...     partial(kalman_em_step, y), (one, one, one, one, m0, one)
        ... )
        >>> bool(history[-1] > history[0])
        True
    """
    if max_iter < 1:
        raise ValueError(f"max_iter must be at least 1, got {max_iter}")
    history = []
    for _ in range(max_iter):
        log_likelihood, params = step(*params)
        history.append(log_likelihood.sum())
        if len(history) > 1 and history[-1] - history[-2] < tol:
            break
    return params, torch.stack(history)
