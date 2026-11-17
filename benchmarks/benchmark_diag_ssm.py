"""Benchmark diag_state_space's two paths for one input and one output (#203).

A real model with complex poles, x of shape (B, N), runs three ways from the
same A, C and D:

- ``modal``: ``diag_state_space`` with x as one feature, shape (B, N, 1),
  which keeps it on the modal path: one complex recursion per mode, with the
  input and output projected at every step;
- ``pfe``: ``diag_state_space`` with x of shape (B, N), which folds the
  projections into residues and runs ``pfefilt`` in its half form, one pole
  of each conjugate pair, real poles in the real dtype;
- ``ssm``: ``state_space(A, x, C=C, D=D)``, the dense recursion.

Each time includes the eigendecomposition. Run from the repository root:

    pixi run python benchmarks/benchmark_diag_ssm.py
"""

import torch
from _timing import bench

from philtorch.lti import diag_state_space, state_space


def model(M, dtype, device):
    """A real, stable, well-conditioned model of order M (even): rotation
    blocks of radius 0.8 to 0.95 in a random orthonormal basis."""
    radius = torch.rand(M // 2, dtype=torch.float64) * 0.15 + 0.8
    angle = torch.rand(M // 2, dtype=torch.float64) * torch.pi
    c, s = radius * angle.cos(), radius * angle.sin()
    blocks = torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2)
    Q, _ = torch.linalg.qr(torch.randn(M, M, dtype=torch.float64))
    A = Q @ torch.block_diag(*blocks) @ Q.T
    C = torch.randn(M, dtype=torch.float64)
    D = torch.randn((), dtype=torch.float64)
    return (t.to(dtype=dtype, device=device) for t in (A, C, D))


def main():
    torch.manual_seed(0)
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    names = ("modal", "pfe", "ssm")
    header = f"{'device dtype M B N':<34s}" + "".join(
        f"{f'{n} ({m})':>14s}" for m in ("fwd", "f+b") for n in names
    )
    for device in devices:
        print(f"\nDevice: {torch.cuda.get_device_name() if device == 'cuda' else 'CPU'}")
        print(header)
        print("-" * len(header))
        for dtype in (torch.float32, torch.float64):
            for M in (2, 8, 32):
                A, C, D = model(M, dtype, device)
                for B in (1, 16):
                    for N in (1_000, 10_000, 100_000):
                        x = torch.randn(B, N, dtype=dtype, device=device)
                        e0 = A.new_zeros(M, 1)
                        e0[0] = 1
                        y_modal = diag_state_space(x[..., None], A=A, B=e0, C=C, D=D[None])
                        err = (diag_state_space(x, A=A, C=C, D=D) - y_modal).abs().max().item()
                        scale = y_modal.abs().max().item()
                        assert err <= (1e-3 if dtype == torch.float32 else 1e-8) * scale, err

                        Ag, Cg, Dg = (t.clone().requires_grad_() for t in (A, C, D))
                        fns = {
                            "modal": lambda A, C, D: diag_state_space(
                                x[..., None], A=A, B=e0, C=C, D=D[None]
                            ),
                            "pfe": lambda A, C, D: diag_state_space(x, A=A, C=C, D=D),
                            "ssm": lambda A, C, D: state_space(A, x, C=C, D=D),
                        }
                        row = f"{f'{device} {str(dtype)[6:]} {M} {B} {N}':<34s}"
                        row += "".join(
                            f"{bench(lambda n=n: fns[n](A, C, D), width=11)}" for n in names
                        )
                        row += "".join(
                            f"{bench(lambda n=n: fns[n](Ag, Cg, Dg).sum().backward(), width=11)}"
                            for n in names
                        )
                        print(row, flush=True)


if __name__ == "__main__":
    main()
