import torch
import torch.nn.functional as F
from typing import Optional, Union, Tuple
from torch import Tensor
from torch.autograd import Function
from typing import Any, List, Optional, Tuple, Union
from torchlpc import sample_wise_lpc

from ..mat import matrix_power_accumulate, find_eigenvectors
from .recur import linear_recurrence
from .. import EXTENSION_LOADED


def extension_backend_indicator(x: Tensor, M: int) -> bool:
    """
    Check if the extension backend should be used based on the input tensor and its last dimension.
    """
    return EXTENSION_LOADED and (x.is_cpu or (M <= 2))


class LTIMatrixRecurrence(Function):
    @staticmethod
    def forward(A: Tensor, zi: Tensor, x: Tensor) -> Tensor:
        if x.size(-1) == 2:
            return torch.ops.philtorch.lti_recur2(A, zi, x)
        return torch.ops.philtorch.lti_recurN(A, zi, x)

    @staticmethod
    def setup_context(ctx: Any, inputs: List[Any], output: Any) -> Any:
        A, zi, _ = inputs
        y = output
        ctx.save_for_backward(A, zi, y)
        ctx.save_for_forward(A, zi, y)

    @staticmethod
    def backward(
        ctx: Any, grad_y: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        A, zi, y = ctx.saved_tensors
        grad_x = grad_A = grad_zi = None

        AmT = A.mT.conj_physical()

        flipped_grad_x = LTIMatrixRecurrence.apply(
            AmT, torch.zeros_like(zi), grad_y.flip(1)
        )

        if ctx.needs_input_grad[1]:
            grad_zi = (AmT @ flipped_grad_x[:, -1, :, None]).squeeze(-1)

        if ctx.needs_input_grad[2]:
            grad_x = flipped_grad_x.flip(1)

        if ctx.needs_input_grad[0]:
            valid_y = y[:, :-1]
            padded_y = torch.cat([zi.unsqueeze(1), valid_y], dim=1)
            if A.dim() == 2:
                grad_A = flipped_grad_x.flip(1).flatten(
                    0, 1
                ).T @ padded_y.conj_physical().flatten(0, 1)
            else:
                grad_A = flipped_grad_x.flip(1).mT @ padded_y.conj_physical()

        return grad_A, grad_zi, grad_x

    @staticmethod
    def jvp(
        ctx: Any, grad_A: torch.Tensor, grad_zi: torch.Tensor, grad_x: torch.Tensor
    ) -> torch.Tensor:
        A, zi, y = ctx.saved_tensors

        fwd_zi = grad_zi if grad_zi is not None else torch.zeros_like(zi)
        fwd_x = grad_x if grad_x is not None else torch.zeros_like(y)

        if grad_A is not None:
            padded_y = torch.cat([zi.unsqueeze(1), y[:, :-1]], dim=1)
            fwd_A = (
                (grad_A if grad_A.dim() == 2 else grad_A.unsqueeze(-3))
                @ padded_y.unsqueeze(-1)
            ).squeeze(-1)
            fwd_x = fwd_x + fwd_A

        return LTIMatrixRecurrence.apply(A, fwd_zi, fwd_x)


def _recursion_loop(
    A: Tensor,
    zi: Tensor,
    x: Tensor,
    out_idx: Optional[int] = None,
) -> Tensor:
    results = []
    AT = A.mT
    if x.dim() == 2:
        M = A.size(-1)
        x = torch.cat(
            [x.unsqueeze(-1), x.new_zeros(*x.shape, M - 1)], dim=-1
        )  # (batch, time, M)
    if A.dim() == 2:
        h = zi
        for xn in x.unbind(1):
            h = torch.addmm(xn, h, AT)
            results.append(h if out_idx is None else h[:, out_idx])
        output = torch.stack(results, dim=1)
    else:
        h = zi.unsqueeze(1)
        for xn in x.unbind(1):
            h = torch.baddbmm(xn.unsqueeze(1), h, AT)
            results.append(h if out_idx is None else h[:, :, out_idx])
        output = torch.cat(results, dim=1)

    return output


def state_space_recursion(
    A: Tensor,
    zi: Tensor,
    x: Tensor,
    *,
    unroll_factor: int = 1,
    out_idx: Optional[int] = None,
) -> Tensor:
    assert x.dim() in (
        2,
        3,
    ), f"Input signal must be 2D or 3D (batch, time, [features]), got {x.shape}"
    assert A.dim() in (2, 3), f"State matrix A must be 2D or 3D, got {A.shape}"
    assert A.size(-2) == A.size(-1), f"State matrix A must be square, got {A.shape}"
    if A.dim() == 3:
        assert x.size(0) == A.size(
            0
        ), f"Batch size of A must match batch size of x, got A: {A.size(0)}, x: {x.size(0)}"

    if x.dim() == 3:
        assert A.size(-1) == x.size(
            -1
        ), f"Last dimension of A must match last dimension of x, got A: {A.size(-1)}, x: {x.size(-1)}"

    batch_size, N = x.size(0), x.size(1)
    M = A.size(-1)

    assert zi.dim() == 2, f"Initial conditions zi must be 2D, got {zi.shape}"
    assert (
        zi.size(0) == batch_size
    ), f"Batch size of zi must match batch size of x, got zi: {zi.size(0)}, x: {batch_size}"
    assert (
        zi.size(1) == M
    ), f"Last dimension of zi must match last dimension of A, got zi: {zi.size(1)}, A: {M}"

    if unroll_factor < 1:
        raise ValueError("Unroll factor must be >= 1")
    else:
        block_size = unroll_factor

    # boundary condition
    if block_size == 1 or block_size >= N:
        return _recursion_loop(A, zi, x, out_idx=out_idx)

    remainder = N % block_size
    if remainder != 0:
        x = F.pad(x, (0, 0) * (x.dim() - 2) + (0, block_size - remainder))
        N = x.size(1)  # Update T after padding

    unrolled_x = x.unflatten(1, (-1, block_size)).flatten(2, -1)

    A_powers = matrix_power_accumulate(A, block_size)
    A_powered = A_powers[..., -1, :, :]
    A_powers_plus_I = torch.cat(
        [
            A_powers[..., :-1, :, :].flip(-3),
            torch.eye(M, device=A.device, dtype=A.dtype)
            .broadcast_to(A.shape)
            .unsqueeze(-3),
        ],
        dim=-3,
    )

    mat1 = (
        A_powers_plus_I.transpose(-2, -1).flatten(-3, -2)
        if x.dim() == 3
        else A_powers_plus_I[..., 0]  # assume x -> x * [1, 0, 0, ...] in the 2D case
    )
    z = unrolled_x @ mat1

    initials = torch.cat(
        [
            zi.unsqueeze(1),
            state_space_recursion(A_powered, zi, z, unroll_factor=unroll_factor),
        ],
        dim=1,
    )

    # prepare the augmented matrix and input for all the remaining steps
    aug_x = torch.cat(
        [initials[:, :-1], unrolled_x[..., : -(1 if x.dim() == 2 else M)]], dim=2
    )

    if out_idx is None:
        mat2 = A_powers[..., :-1, :, :].flatten(-3, -2)
        if x.dim() == 3:
            mat3 = (
                torch.cat(
                    [
                        A_powers_plus_I[..., 1:, :, :],
                        A_powers_plus_I.new_zeros(
                            A_powers_plus_I.shape[:-3] + (block_size - 2, M, M)
                        ),
                    ],
                    dim=-3,
                )
                .unfold(-3, block_size - 1, 1)
                .transpose(-2, -1)
                .flip(-4)
                .flatten(-4, -3)
                .flatten(-2, -1)
            )
        else:
            mat3 = (
                torch.cat(
                    [
                        A_powers_plus_I[..., 1:, :, 0],
                        A_powers_plus_I.new_zeros(
                            A_powers_plus_I.shape[:-3] + (block_size - 2, M)
                        ),
                    ],
                    dim=-2,
                )
                .unfold(-2, block_size - 1, 1)
                .flip(-3)
                .flatten(-3, -2)
            )
    else:
        mat2 = A_powers[..., :-1, out_idx, :]
        if x.dim() == 3:
            mat3 = (
                torch.cat(
                    [
                        A_powers_plus_I[..., 1:, out_idx, :],
                        A_powers_plus_I.new_zeros(
                            A_powers_plus_I.shape[:-3] + (block_size - 2, M)
                        ),
                    ],
                    dim=-2,
                )
                .unfold(-2, block_size - 1, 1)
                .transpose(-2, -1)
                .flip(-3)
                .flatten(-2, -1)
            )
        else:
            mat3 = (
                torch.cat(
                    [
                        A_powers_plus_I[..., 1:, out_idx, 0],
                        A_powers_plus_I.new_zeros(
                            A_powers_plus_I.shape[:-3] + (block_size - 2,)
                        ),
                    ],
                    dim=-1,
                )
                .unfold(-1, block_size - 1, 1)
                .flip(-2)
            )

    aug_A = torch.cat([mat2.mT, mat3.mT], dim=-2)
    output = aug_x @ aug_A

    # concat the first M - 1 outputs with the last one
    if out_idx is None:
        output = (
            torch.cat([output, initials[:, 1:, :]], dim=2)
            .unflatten(2, (-1, M))
            .flatten(1, 2)
        )
    else:
        output = torch.cat([output, initials[:, 1:, out_idx, None]], dim=2).flatten(
            1, 2
        )
    if remainder != 0:
        # if we padded the input, we need to remove the padding from the output
        output = output[:, : -(block_size - remainder)]
    return output


def state_space(
    A: Tensor,
    x: Tensor,
    B: Optional[Tensor] = None,
    C: Optional[Tensor] = None,
    D: Optional[Tensor] = None,
    zi: Optional[Tensor] = None,
    unroll_factor: Optional[int] = None,
    out_idx: Optional[int] = None,
    # **kwargs,
):
    assert x.dim() in (
        2,
        3,
    ), f"Input signal must be 2D or 3D (batch, time, [features]), got {x.shape}"

    assert A.dim() in (2, 3), f"State matrix A must be 2D or 3D, got {A.shape}"
    assert A.size(-2) == A.size(-1), f"State matrix A must be square, got {A.shape}"
    if A.dim() == 3:
        assert x.size(0) == A.size(
            0
        ), f"Batch size of A must match batch size of x, got A: {A.size(0)}, x: {x.size(0)}"

    if not (C is None or out_idx is None):
        raise ValueError(
            "C and out_idx cannot be used together. Use either C or out_idx."
        )

    batch_size, N, *_ = x.shape
    M = A.size(-1)

    return_zf = True
    if zi is None:
        return_zf = False
        zi = x.new_zeros(batch_size, M)
    elif zi.dim() == 1:
        zi = zi.unsqueeze(0).expand(batch_size, -1)

    if unroll_factor is None:
        unroll_factor = round(N**0.5)

    if x.dim() == 2:
        features = -1
    else:
        features = x.size(-1)

    if B is not None:
        match B.shape:
            case (BM,) if BM == M:
                assert (
                    x.dim() == 2
                ), f"Input signal x must be 2D when B is of shape {M,}, got {x.shape}"
                Bx = x.unsqueeze(-1) * B
            case (B_batch, BM) if B_batch == batch_size and BM == M:
                assert (
                    x.dim() == 2
                ), f"Input signal x must be 2D when B is of shape {batch_size, M}, got {x.shape}"
                Bx = x.unsqueeze(-1) * B.unsqueeze(1)
            case (BM, F) if BM == M and F == features:
                Bx = x @ B.T
            case (B_batch, BM, F) if (
                B_batch == batch_size and BM == M and F == features
            ):
                Bx = torch.linalg.vecdot(
                    B.unsqueeze(1).conj(), x.unsqueeze(-2)
                )  # (batch_size, N, M)
            case _:
                raise ValueError(
                    f"Input matrix B must be of shape ({M,}), ({batch_size, M}), ({M, features}), or ({batch_size, M, features}), got {B.shape}"
                )
    else:
        Bx = x

    if return_zf or out_idx is None:
        h = state_space_recursion(A, zi, Bx, unroll_factor=unroll_factor, out_idx=None)
        zf = h[:, -1, :] if return_zf else None
        h = (
            torch.cat([zi.unsqueeze(1), h[:, :-1]], dim=1)
            if out_idx is None
            else torch.cat([zi[:, None, out_idx], h[:, :-1, out_idx]], dim=1)
        )
    else:
        zf = None
        h = state_space_recursion(
            A, zi, Bx, unroll_factor=unroll_factor, out_idx=out_idx
        )
        h = torch.cat([zi[:, None, out_idx], h[:, :-1]], dim=1)

    y = _ssm_C_D(h, x, C, D, batch_size, M)

    if return_zf:
        return y, zf
    return y


def _ssm_C_D(h, x, C, D, batch_size, M):
    if x.dim() == 2:
        features = -1
    else:
        features = x.size(-1)

    if D is not None:
        match D.shape:
            case (F,) if F == features:
                Dx = x @ D
            case (D_batch,) if D_batch == batch_size:
                assert (
                    x.dim() == 2
                ), f"Input signal x must be 2D when D is of shape {batch_size,}, got {x.shape}"
                Dx = D.unsqueeze(1) * x
            case (1,) | ():
                Dx = x * D
            case (_,):
                assert (
                    x.dim() == 2
                ), f"Input signal x must be 2D when D is of shape (_,), got {x.shape}"
                Dx = D * x.unsqueeze(-1)
            case (D_batch, F) if D_batch == batch_size and F == features:
                Dx = torch.linalg.vecdot(D.unsqueeze(1).conj(), x)
            case (D_batch, _) if D_batch == batch_size:
                assert (
                    x.dim() == 2
                ), f"Input signal x must be 2D when D is of shape ({batch_size}, _), got {x.shape}"
                Dx = D.unsqueeze(1) * x.unsqueeze(-1)
            case (_, F) if F == features:
                Dx = x @ D.T
            case (D_batch, _, F) if D_batch == batch_size and F == features:
                Dx = torch.linalg.vecdot(D.unsqueeze(1).conj(), x.unsqueeze(-2))
            case _:
                raise ValueError(
                    f"Input matrix D must be of shape ({batch_size,}), (), ({features},), (_, ), ({batch_size}, {features}), ({batch_size}, _), (_, {features}), or ({batch_size}, _, {features}), got {D.shape}"
                )
    else:
        Dx = None

    if C is not None:
        match C.shape:
            case (CM,) if CM == M:
                Ch = h @ C
            case (C_batch, CM) if C_batch == batch_size and CM == M:
                Ch = torch.linalg.vecdot(C.unsqueeze(1).conj(), h)
            case (_, CM) if CM == M:
                Ch = h @ C.T
            case (C_batch, _, CM) if C_batch == batch_size and CM == M:
                Ch = h @ C.mT
            case _:
                raise ValueError(
                    f"Output matrix C must be of shape ({M,}), ({batch_size, M}), (_, {M}), or ({batch_size}, _, {M}), got {C.shape}"
                )
    else:
        Ch = h

    if Dx is not None:
        y = Ch + Dx
    else:
        y = Ch
    return y


def diag_state_space(
    x: Tensor,
    L: Optional[Tensor] = None,
    V: Optional[Tensor] = None,
    Vinv: Optional[Tensor] = None,
    A: Optional[Tensor] = None,
    B: Optional[Tensor] = None,
    C: Optional[Tensor] = None,
    D: Optional[Tensor] = None,
    zi: Optional[Tensor] = None,
    out_idx: Optional[int] = None,
    unroll_factor: Optional[int] = None,
):
    assert x.dim() in (
        2,
        3,
    ), f"Input signal must be 2D or 3D (batch, time, [features]), got {x.shape}"
    if not (C is None or out_idx is None):
        raise ValueError(
            "C and out_idx cannot be used together. Use either C or out_idx."
        )

    batch_size, N = x.size(0), x.size(1)
    if unroll_factor is None:
        unroll_factor = round(N**0.5)

    if L is None:
        assert A is not None, "Either L or A must be provided"
        assert A.dim() in (2, 3), f"State matrix A must be 2D or 3D, got {A.shape}"
        assert A.size(-2) == A.size(-1), f"State matrix A must be square, got {A.shape}"
        if A.dim() == 3:
            assert x.size(0) == A.size(
                0
            ), f"Batch size of A must match batch size of x, got A: {A.size(0)}, x: {x.size(0)}"
        L = torch.linalg.eigvals(A)
        V = Vinv = None
        M = A.size(-1)
    else:
        match L.shape:
            case (_,):
                M = L.size(0)
                if V is not None:
                    assert V.dim() == 2, f"P must be 2D, got {V.shape}"
                    assert (
                        V.size(0) == V.size(1) == M
                    ), f"P must be square with size {M}, got {V.shape}"
                if Vinv is not None:
                    assert Vinv.dim() == 2, f"Vinv must be 2D, got {Vinv.shape}"
                    assert (
                        Vinv.size(0) == Vinv.size(1) == M
                    ), f"Vinv must be square with size {M}, got {Vinv.shape}"

                if A is not None:
                    assert A.dim() == 2, f"A must be 2D, got {A.shape}"
                    assert (
                        A.size(0) == A.size(1) == M
                    ), f"A must be square with size {M}, got {A.shape}"

            case (L_batch, _) if L_batch == batch_size:
                M = L.size(1)
                assert not (
                    V is None and Vinv is None
                ), "P and Vinv cannot both be None when L is a batch of vectors"
                if V is not None:
                    assert V.dim() == 3, f"P must be 3D, got {V.shape}"
                    assert (
                        V.size(0) == batch_size and V.size(1) == V.size(2) == M
                    ), f"P must be a batch of square matrices with size {M}, got {V.shape}"
                if Vinv is not None:
                    assert Vinv.dim() == 3, f"Vinv must be 3D, got {Vinv.shape}"
                    assert (
                        Vinv.size(0) == batch_size and Vinv.size(1) == Vinv.size(2) == M
                    ), f"Vinv must be a batch of square matrices with size {M}, got {Vinv.shape}"

                if A is not None:
                    assert A.dim() == 3, f"A must be 3D, got {A.shape}"
                    assert (
                        A.size(0) == batch_size and A.size(1) == A.size(2) == M
                    ), f"A must be a batch of square matrices with size {M}, got {A.shape}"

            case _:
                raise ValueError(
                    f"L must be a vector or a batch of vectors, got {L.shape}"
                )

    match (V, Vinv, A):
        case (None, None, None):
            # scalar case
            V = Vinv = torch.eye(M, device=x.device, dtype=x.dtype)
        case (None, None, _):
            V = find_eigenvectors(A, L)
            Vinv = torch.linalg.inv(V)
        case (None, _, _):
            V = torch.linalg.inv(Vinv)
        case (_, None, _):
            Vinv = torch.linalg.inv(V)
        case (_, _, _):
            pass
        case _:
            raise ValueError("Only one of V, Vinv, or A can be provided at a time.")
    assert Vinv is not None, "Vinv must be provided or computed from A or V"
    assert V is not None, "V must be provided or computed from A or Vinv"

    return_zf = True
    if zi is None:
        return_zf = False
        zi = x.new_zeros(batch_size, M)
    elif zi.dim() == 1:
        zi = zi.unsqueeze(0).expand(batch_size, -1)

    x_orig = x
    if Vinv.is_complex():
        if not zi.is_complex():
            zi = zi + 0j  # Ensure zi is complex if Vinv is complex
        if not x.is_complex():
            x = x + 0j  # Ensure x is complex if Vinv is complex
        if B is not None and not B.is_complex():
            B = B + 0j

    match Vinv.dim():
        case 2:
            Vinvzi = zi @ Vinv.T
        case 3:
            Vinvzi = torch.linalg.vecdot(Vinv.conj(), zi.unsqueeze(1))
        case _:
            assert False, f"Vinv must be 2D or 3D, got {Vinv.shape}"

    if x.dim() == 2:
        features = -1
    else:
        features = x.size(-1)

    if B is not None:
        match B.shape:
            case (BM,) if BM == M:
                assert (
                    x.dim() == 2
                ), f"Input signal x must be 2D when B is of shape {M,}, got {x.shape}"
                VinvB = Vinv @ B
                if VinvB.dim() == 2:
                    VinvB = VinvB.unsqueeze(1)
                VinvBx = x.unsqueeze(-1) * VinvB
            case (B_batch, BM) if B_batch == batch_size and BM == M:
                assert (
                    x.dim() == 2
                ), f"Input signal x must be 2D when B is of shape {batch_size, M}, got {x.shape}"
                VinvB = (
                    B @ Vinv.T
                    if Vinv.dim() == 2
                    else torch.linalg.vecdot(Vinv.conj(), B.unsqueeze(1))
                )
                VinvBx = x.unsqueeze(-1) * VinvB.unsqueeze(1)
            case (BM, F) if BM == M and F == features:
                VinvB = Vinv @ B
                VinvBx = x @ VinvB.mT
            case (B_batch, BM, F) if (
                B_batch == batch_size and BM == M and F == features
            ):
                VinvB = Vinv @ B
                VinvBx = torch.linalg.vecdot(VinvB.unsqueeze(1).conj(), x.unsqueeze(-2))
            case _:
                raise ValueError(
                    f"Input matrix B must be of shape ({M,}), ({batch_size, M}), ({M, features}), or ({batch_size, M, features}), got {B.shape}"
                )
    elif x.dim() == 2 and Vinv.dim() == 2:
        VinvBx = x.unsqueeze(-1) * Vinv[:, 0]
    elif x.dim() == 2 and Vinv.dim() == 3:
        VinvBx = x.unsqueeze(-1) * Vinv[:, None, :, 0]
    elif x.dim() == 3:
        VinvBx = x @ Vinv.mT
    else:
        assert False, f"Input signal x must be 2D or 3D, got {x.shape}"

    Vinvh = linear_recurrence(
        L.broadcast_to((batch_size, M)).flatten(0, 1),
        Vinvzi.flatten(),
        VinvBx.mT.flatten(0, 1),
        unroll_factor=unroll_factor,
    ).unflatten(0, (batch_size, M))
    if not return_zf and out_idx is not None:
        h = torch.cat(
            [
                zi[:, None, out_idx],
                (V[..., out_idx, None, :] @ Vinvh[..., :-1]).squeeze(-2),
            ],
            dim=1,
        )
        zf = None
    else:
        h = (V @ Vinvh).mT
        zf = h[:, -1, :] if return_zf else None
        h = torch.cat(
            (
                [zi.unsqueeze(1), h[:, :-1]]
                if out_idx is None
                else [zi[:, None, out_idx], h[:, :-1, out_idx]]
            ),
            dim=1,
        )

    if C is not None and not C.is_complex():
        h = h.real
        zf = zf.real if return_zf else None

    y = _ssm_C_D(h, x_orig, C, D, batch_size, M)
    if return_zf:
        return y, zf
    return y
