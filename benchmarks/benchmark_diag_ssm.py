"""Benchmark diag_state_space against pfefilt for one input and one output (#203).

A real model with complex poles, x of shape (B, N), runs three ways from the
same A, C and D:

- ``diag``: ``diag_state_space(x, A=A, C=C, D=D)``, as today;
- ``pfe``: the same eigendecomposition folded into residues,
  r = (C V) * (V^-1 B), and ``pfefilt`` in its half form, one pole of each
  conjugate pair, real poles in the real dtype;
- ``ssm``: ``state_space(A, x, C=C, D=D)``, the dense recursion.

Each time includes the eigendecomposition. Run from the repository root:

    pixi run python benchmarks/benchmark_diag_ssm.py
"""

import torch
from _timing import bench

from philtorch.lti import diag_state_space, pfefilt, state_space
from philtorch.mat import find_eigenvectors


def pfe_route(x, A, C, D):
    """diag_state_space(x, A=A, C=C, D=D) through pfefilt, B = e_0."""
    L = torch.linalg.eigvals(A)
    V = find_eigenvectors(A, L)
    e0 = torch.zeros(V.size(-1), 1, dtype=V.dtype, device=V.device)
    e0[0] = 1
    r = (C.to(V.dtype) @ V) * torch.linalg.solve(V, e0)[:, 0]
    # one pole of each conjugate pair; a real pole's residue is real
    keep = L.imag >= 0
    real_pole = L.imag == 0
    r = torch.where(real_pole, r.real.to(r.dtype), r)
    return pfefilt(r[keep], L[keep], D, x)


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
    names = ("diag", "pfe", "ssm")
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
                        y_ref = diag_state_space(x, A=A, C=C, D=D)
                        err = (pfe_route(x, A, C, D) - y_ref).abs().max().item()
                        scale = y_ref.abs().max().item()
                        assert err <= (1e-3 if dtype == torch.float32 else 1e-8) * scale, err

                        fns = {
                            "diag": lambda: diag_state_space(x, A=A, C=C, D=D),
                            "pfe": lambda: pfe_route(x, A, C, D),
                            "ssm": lambda: state_space(A, x, C=C, D=D),
                        }
                        Ag, Cg, Dg = (t.clone().requires_grad_() for t in (A, C, D))
                        grads = {
                            "diag": lambda: diag_state_space(x, A=Ag, C=Cg, D=Dg),
                            "pfe": lambda: pfe_route(x, Ag, Cg, Dg),
                            "ssm": lambda: state_space(Ag, x, C=Cg, D=Dg),
                        }
                        row = f"{f'{device} {str(dtype)[6:]} {M} {B} {N}':<34s}"
                        row += "".join(f"{bench(fns[n], width=11)}" for n in names)
                        row += "".join(
                            f"{bench(lambda n=n: grads[n]().sum().backward(), width=11)}"
                            for n in names
                        )
                        print(row, flush=True)


if __name__ == "__main__":
    main()
