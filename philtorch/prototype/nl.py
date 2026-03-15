import torch
from torch import Tensor
import torch.nn.functional as F
from typing import Any, Optional, Callable, Union
from functools import partial

from ..lpv.ssm import _ext_ss_recur, state_space_recursion, extension_backend_indicator


def newton_solve(
    func: Callable[[Tensor, Tensor], Tensor],
    x: Tensor,
    y0: Tensor,
    init: Tensor,
    max_iter: int = 3,
    atol: float = 1e-6,
    rtol: float = 1e-6,
    fprime: Optional[Callable[[Tensor, Tensor], Tensor]] = None,
    unroll_factor: int = 1,
    return_intermediate: bool = False,
) -> Union[Tensor, tuple[Tensor, list[tuple[Tensor, Tensor]]]]:
    """Solve for the root of a function using Newton's method.

    Args:
        func (Callable[[Tensor], Tensor]): A given function that given an input tensor and previous state, returns the next state. The function should be differentiable with respect to its input.
        x (Tensor): Input tensor to be passed to the function with shape (B, L, N).
        y0 (Tensor): Initial state to be passed to the function with shape (B, L, D).
        init (Tensor): Initial state to be passed to the function with shape (B, D).
        max_iter (int, optional): Maximum number of iterations. Defaults to 3.
        atol (float, optional): Tolerance for convergence. Defaults to 1e-6.
        rtol (float, optional): Relative tolerance for convergence. Defaults to 1e-6.
        fprime (Optional[Callable[[Tensor], Tensor]], optional): Optional function to compute the Jacobian. If None, the Jacobian will be computed using autograd. Defaults to None.

    Returns:
        Tensor: Approximation of the root of the function.
    """
    assert max_iter < x.size(1), "max_iter must be less than the sequence length of x"
    M = init.size(-1)
    recur_runner = (
        _ext_ss_recur
        if extension_backend_indicator(x, M) and unroll_factor in (None, 1)
        else partial(state_space_recursion, unroll_factor=unroll_factor)
    )
    y = torch.cat([init.unsqueeze(1), y0], dim=1)
    computed = []
    intermediate = []
    for i in range(max_iter):
        next_y = func(y[:, :-1], x[:, i:])
        # computed = next_y[:, :1]
        res = next_y - y[:, 1:]

        if fprime is not None:
            Jac = fprime(y[:, 1:-1], x[:, i + 1 :])
        else:
            # Compute the Jacobian using autograd
            Jac = torch.autograd.functional.jacobian(
                lambda y: func(y, x[:, i + 1 :]).sum((0, 1)),
                y[:, 1:-1],
                create_graph=True,
                vectorize=True,
            ).permute(1, 2, 0, 3)
        # Solve for the update step
        delta = recur_runner(Jac, res[:, 0], res[:, 1:])
        new_y = y[:, 1:] + torch.cat(
            [torch.zeros_like(init).unsqueeze(1), delta], dim=1
        )
        if torch.allclose(new_y, y[:, 1:], atol=atol, rtol=rtol):
            break
        computed.append(next_y[:, :1])
        y = new_y

        if return_intermediate:
            intermediate.append(
                (
                    torch.cat(computed + [new_y[:, 1:]], dim=1),
                    res.square().sum(dim=(-1, -2)),
                )
            )
    result = torch.cat(computed + [new_y[:, 1:]], dim=1)
    if return_intermediate:
        return result, intermediate
    return result
