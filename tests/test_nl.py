import torch
import pytest
from torch._higher_order_ops import scan
from philtorch.prototype.nl import newton_solve


def feedforward_comp_func(gprev, x, at, rt):
    attack_mask = x < gprev
    coeff = torch.where(attack_mask, at, rt)
    return coeff * x + (1 - coeff) * gprev


def fprime(gprev, x, at, rt):
    attack_mask = x < gprev
    coeff = torch.where(attack_mask, at, rt)
    return (1 - coeff).unsqueeze(-2)


@pytest.mark.parametrize("use_fprime", [False, True])
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA not available"
            ),
        ),
    ],
)
def test_newton_solve(use_fprime: bool, device: str):
    batch_size = 3
    g = torch.rand(batch_size, 100, 1).to(device)
    y0 = torch.ones(batch_size, 100, 1).to(device)
    at = torch.rand(batch_size, 1, 1).to(device)
    rt = torch.rand(batch_size, 1, 1).to(device)
    init = torch.ones(batch_size, 1).to(device)
    y, iters = newton_solve(
        lambda gprev, x: feedforward_comp_func(gprev, x, at=at, rt=rt),
        g,
        y0,
        max_iter=10,
        init=init,
        atol=1e-8,
        rtol=1e-6,
        fprime=(
            (lambda gprev, x: fprime(gprev, x, at=at, rt=rt)) if use_fprime else None
        ),
    )

    # bug: scan will move the specified dim to the first dim when returning the output, so we need to transpose it back
    target = scan(
        lambda gprev, x: (lambda x: (x, x.clone()))(
            feedforward_comp_func(gprev, x, at=at.squeeze(-2), rt=rt.squeeze(-2))
        ),
        init=init,
        xs=g,
        dim=1,
    )[1].transpose(0, 1)

    assert torch.allclose(y, target), (y - target).abs().max()
