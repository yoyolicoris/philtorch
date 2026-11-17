from itertools import chain, product

import numpy as np
import pytest
import torch
from scipy import signal
from torch.utils import _pytree as pytree
from torch.utils._python_dispatch import TorchDispatchMode

from philtorch.lti import diag_state_space, state_space, state_space_recursion
from philtorch.lti.ssm import _ssm_B, _ssm_C_D
from philtorch.mat import companion

from ._cases import each_choice, stable_denominators
from .test_lti_lfilter import _generate_random_signal


def _generate_diagonalizable_matrix(shape: tuple[int, ...]) -> torch.Tensor:
    order = shape[-1]
    values = torch.arange(1, order + 1, dtype=torch.get_default_dtype())
    basis = torch.vander(values, N=order, increasing=True)
    Q, _ = torch.linalg.qr(basis)
    eigenvalues = torch.linspace(0.2, 0.8, order, dtype=Q.dtype)
    A = Q @ torch.diag(eigenvalues) @ Q.mT
    return A.expand(*shape[:-2], order, order).clone()


@pytest.mark.parametrize("B", [1, 8])
@pytest.mark.parametrize("order", [1, 3])
@pytest.mark.parametrize("unroll_factor", [None, 5])
def test_time_invariant_ssm(
    B: int,
    order: int,
    unroll_factor: int,
):
    """Test time-invariant filters against scipy.signal.lfilter"""
    # 29 is not a multiple of 5 and exceeds it: the padded last block.
    T = 29

    # Generate test data
    a = stable_denominators(order, B)
    x = _generate_random_signal(B, T)

    # Convert to torch tensors
    a_torch = torch.from_numpy(a)
    x_torch = torch.from_numpy(x)
    A = companion(a_torch).squeeze(0)
    zi = x_torch.new_zeros(B, A.size(-1))

    # Apply philtorch filter
    y_torch = state_space_recursion(A, zi, x_torch, out_idx=0, unroll_factor=unroll_factor)

    # Apply scipy filter
    y_scipy = np.stack(
        [signal.lfilter([1.0], [1.0] + a[i].tolist(), x[i]) for i in range(B)],
    )

    # Compare outputs
    assert np.allclose(y_torch.numpy(), y_scipy)


@pytest.mark.parametrize("order", [8])
@pytest.mark.parametrize("out_idx", [0, 1, 3, 7])
def test_out_idx(order: int, out_idx: int):
    """Test the out_idx functionality of state_space_recursion"""

    # Generate test data
    B, T = 2, 10
    a = stable_denominators(order, B)
    x = _generate_random_signal(B, T)

    # Convert to torch tensors
    a_torch = torch.from_numpy(a)
    x_torch = torch.from_numpy(x)
    A = companion(a_torch)
    zi = x_torch.new_zeros(B, A.size(-1))

    # Apply philtorch filter with out_idx
    y_torch = state_space_recursion(A, zi, x_torch, out_idx=out_idx)[:, out_idx:]

    y_scipy = np.stack(
        [signal.lfilter([1.0], [1.0] + a[i].tolist(), x[i]) for i in range(B)],
    )
    if out_idx > 0:
        y_scipy = y_scipy[:, :-out_idx]

    # Compare outputs
    assert np.allclose(y_torch.numpy(), y_scipy)


# diag_state_space folds A's eigenvectors into B or into x, so B's and x's
# shapes meet A's: those go together; C and zi are independent of them.
@pytest.mark.parametrize(
    ("x_B_A_shape", "C_shape", "zi_shape"),
    each_choice(
        list(
            product(
                [
                    ((5, 97), (3,)),
                    ((5, 97), (5, 3)),
                    ((5, 97), None),
                    ((5, 97, 3), None),
                    ((5, 97, 2), (3, 2)),
                    ((5, 97, 2), (5, 3, 2)),
                ],
                [(3, 3), (5, 3, 3)],
            )
        ),
        [None, (3,), (5, 3), (2, 3), (5, 2, 3)],
        [None, (3,), (5, 3)],
    ),
)
@pytest.mark.parametrize("ssm", [state_space, diag_state_space])
def test_ssm_shape_handling(x_B_A_shape, C_shape, zi_shape, ssm):
    unroll_factor = 4
    ((x_shape, B_shape), A_shape), D_shape = x_B_A_shape, None

    x = torch.randn(*x_shape)
    A = _generate_diagonalizable_matrix(A_shape)
    B = torch.randn(*B_shape) if B_shape is not None else None
    C = torch.randn(*C_shape) if C_shape is not None else None
    if D_shape is None:
        D = None
    else:
        D = torch.randn(*D_shape) if len(D_shape) > 0 else torch.randn(1)
    zi = torch.randn(*zi_shape) if zi_shape is not None else None

    result = ssm(A=A, x=x, B=B, C=C, D=D, zi=zi, unroll_factor=unroll_factor)

    if zi is not None:
        y, zf = result
        assert zf.shape[-1] == zi_shape[-1]
    else:
        y = result

    assert y.shape[:2] == x.shape[:2]

    if y.dim() == 3:
        if C_shape is None:
            assert y.shape[2] == A.shape[-1]
        elif len(C_shape) == 2:
            assert y.shape[2] == C_shape[0]
        elif len(C_shape) == 3:
            assert y.shape[2] == C_shape[1]
        else:
            assert False, f"Unexpected C_shape: {C_shape}"


# Each group fixes x's dimensions and the D, B and C shapes that go with them.
_A_SHAPES = [(3, 3), (5, 3, 3)]
_ZI_SHAPES = [None, (3,), (5, 3)]


@pytest.mark.parametrize(
    ("D_shape", "x_shape", "B_shape", "C_shape", "A_shape", "zi_shape"),
    list(
        chain(
            each_choice(
                [(5,), (1,), ()],
                [(5, 97)],
                [(3,), (5, 3)],
                [(3,), (5, 3)],
                _A_SHAPES,
                _ZI_SHAPES,
            ),
            each_choice(
                [(1,), (), (2, 2), (5, 2, 2)],
                [(5, 97, 2)],
                [(3, 2), (5, 3, 2)],
                [(2, 3), (5, 2, 3)],
                _A_SHAPES,
                _ZI_SHAPES,
            ),
            each_choice(
                [(4,), (5, 4)],
                [(5, 97)],
                [(3,), (5, 3)],
                [(4, 3), (5, 4, 3)],
                _A_SHAPES,
                _ZI_SHAPES,
            ),
            each_choice(
                [(7, 2), (5, 7, 2)],
                [(5, 97, 2)],
                [(3, 2), (5, 3, 2)],
                [(7, 3), (5, 7, 3)],
                _A_SHAPES,
                _ZI_SHAPES,
            ),
            each_choice(
                [(2,), (5, 2)],
                [(5, 97, 2)],
                [(3, 2), (5, 3, 2)],
                [(3,), (5, 3)],
                _A_SHAPES,
                _ZI_SHAPES,
            ),
        )
    ),
)
@pytest.mark.parametrize("ssm", [state_space, diag_state_space])
def test_ssm_D_shape_handling(x_shape, A_shape, B_shape, C_shape, D_shape, zi_shape, ssm):
    unroll_factor = 4

    x = torch.randn(*x_shape)
    A = _generate_diagonalizable_matrix(A_shape)
    B = torch.randn(*B_shape)
    C = torch.randn(*C_shape)
    D = torch.randn(*D_shape) if len(D_shape) > 0 else torch.randn(1)
    zi = torch.randn(*zi_shape) if zi_shape is not None else None

    result = ssm(A=A, x=x, B=B, C=C, D=D, zi=zi, unroll_factor=unroll_factor)

    if zi is not None:
        y, zf = result
        assert zf.shape[-1] == zi_shape[-1]
    else:
        y = result

    assert y.shape[:2] == x.shape[:2]


_b, _N, _M, _F, _P = 3, 6, 4, 2, 5


@pytest.mark.parametrize("dtype", [torch.double, torch.cdouble])
@pytest.mark.parametrize(
    ("x_shape", "B_shape", "equation"),
    [
        ((_b, _N), (_M,), "bn,m->bnm"),
        ((_b, _N), (_b, _M), "bn,bm->bnm"),
        ((_b, _N, _F), (_M, _F), "bnf,mf->bnm"),
        ((_b, _N, _F), (_b, _M, _F), "bnf,bmf->bnm"),
        # batch == M: (M, F) must not be read as a batched (B, M) vector.
        ((2, _N, 2), (2, 2), "bnf,mf->bnm"),
        ((1, _N, 1), (1, 1), "bnf,mf->bnm"),
    ],
)
def test_ssm_B_layouts(x_shape, B_shape, equation, dtype):
    x = torch.randn(x_shape, dtype=dtype)
    B = torch.randn(B_shape, dtype=dtype)
    M = B_shape[-1] if len(x_shape) == 2 else B_shape[-2]
    expected = torch.einsum(equation, x, B)
    torch.testing.assert_close(_ssm_B(B, x, x_shape[0], M), expected)


@pytest.mark.parametrize("dtype", [torch.double, torch.cdouble])
@pytest.mark.parametrize(
    ("x_shape", "D_shape", "equation"),
    [
        ((_b, _N), (), "bn,->bn"),
        ((_b, _N), (1,), "bn,o->bn"),
        ((_b, _N), (_b,), "bn,b->bn"),
        ((_b, _N), (_P,), "bn,p->bnp"),
        ((_b, _N), (_b, _P), "bn,bp->bnp"),
        ((_b, _N, _F), (), "bnf,->bnf"),
        ((_b, _N, _F), (1,), "bnf,o->bnf"),
        ((_b, _N, _F), (_F,), "bnf,f->bn"),
        ((_b, _N, _F), (_b, _F), "bnf,bf->bn"),
        ((_b, _N, _F), (_P, _F), "bnf,pf->bnp"),
        ((_b, _N, _F), (_b, _P, _F), "bnf,bpf->bnp"),
        # batch == 1: a (1,) gain on vector input stays a scalar gain.
        ((1, _N, _F), (1,), "bnf,o->bnf"),
    ],
)
def test_ssm_D_layouts(x_shape, D_shape, equation, dtype):
    x = torch.randn(x_shape, dtype=dtype)
    D = torch.randn(D_shape, dtype=dtype)
    expected = torch.einsum(equation, x, D)
    # Zero states with D's outputs, C h = h without C.
    y = _ssm_C_D(torch.zeros_like(expected), x, None, D, x_shape[0], _M)
    torch.testing.assert_close(y, expected)


@pytest.mark.parametrize("dtype", [torch.double, torch.cdouble])
@pytest.mark.parametrize(
    ("C_shape", "equation"),
    [
        (None, "bnm->bnm"),
        ((_M,), "bnm,m->bn"),
        ((_b, _M), "bnm,bm->bn"),
        ((_P, _M), "bnm,pm->bnp"),
        ((_b, _P, _M), "bnm,bpm->bnp"),
    ],
)
def test_ssm_C_layouts(C_shape, equation, dtype):
    h = torch.randn(_b, _N, _M, dtype=dtype)
    x = torch.randn(_b, _N, dtype=dtype)
    if C_shape is None:
        C, expected = None, h
    else:
        C = torch.randn(C_shape, dtype=dtype)
        expected = torch.einsum(equation, h, C)
    torch.testing.assert_close(_ssm_C_D(h, x, C, None, _b, _M), expected)


@pytest.mark.parametrize("ssm", [state_space, diag_state_space])
def test_ssm_unbatched_B_when_batch_equals_state_dim(ssm):
    # batch == M == F, so the (M, F) matrix has the same shape as a (batch, M) one.
    x = torch.randn(2, 17, 2, dtype=torch.double)
    A = _generate_diagonalizable_matrix((2, 2)).double()
    B = torch.randn(2, 2, dtype=torch.double)

    y = ssm(A=A, x=x, B=B)

    torch.testing.assert_close(y, ssm(A=A, x=x, B=B.expand(2, 2, 2)))


@pytest.mark.parametrize("batched_A", [False, True])
@pytest.mark.parametrize(
    ("x_shape", "B_shape"),
    [
        ((2, 17), (2,)),
        ((2, 17), (2, 2)),
        ((2, 17, 2), (2, 2)),
        ((2, 17, 2), (2, 2, 2)),
    ],
)
def test_diag_state_space_B_layouts_match_state_space(x_shape, B_shape, batched_A):
    # batch == M == F: every 2D B shape collides, so only x.dim() tells them apart.
    x = torch.randn(x_shape, dtype=torch.double)
    A = _generate_diagonalizable_matrix((2, 2, 2) if batched_A else (2, 2)).double()
    B = torch.randn(B_shape, dtype=torch.double)

    y = diag_state_space(A=A, x=x, B=B)

    # A real system gives real outputs, even without C.
    torch.testing.assert_close(y, state_space(A=A, x=x, B=B))


@pytest.mark.parametrize("ssm", [state_space, diag_state_space])
@pytest.mark.parametrize(("x_shape", "B_shape"), [((3, 17), (3, 2)), ((3, 17, 2), (3,))])
def test_ssm_rejects_invalid_B(ssm, x_shape, B_shape):
    A = _generate_diagonalizable_matrix((3, 3))
    with pytest.raises(ValueError, match=f"Input matrix B .* for {len(x_shape)}D"):
        ssm(A=A, x=torch.randn(x_shape), B=torch.randn(B_shape))


@pytest.mark.parametrize("ssm", [state_space, diag_state_space])
def test_ssm_scalar_D_with_single_batch_vector_input(ssm):
    x = torch.randn(1, 17, 2)
    A = _generate_diagonalizable_matrix((3, 3))
    B, C, D = torch.randn(3, 2), torch.randn(2, 3), torch.randn(1)

    y = ssm(A=A, x=x, B=B, C=C, D=D)

    torch.testing.assert_close(y, ssm(A=A, x=x, B=B, C=C, D=D[0]))


@pytest.mark.parametrize(
    ("x_shape", "B_shape"),
    [
        ((_b, _N), (_M, _F)),
        ((_b, _N), ()),
        ((_b, _N), (_M + 1,)),
        ((_b, _N, _F), (_M,)),
        ((_b, _N, _F), (_b, _M)),
        ((_b, _N, _F), (_M, _F + 1)),
    ],
)
def test_ssm_B_rejects_invalid_shapes(x_shape, B_shape):
    with pytest.raises(ValueError, match=f"Input matrix B .* for {len(x_shape)}D"):
        _ssm_B(torch.zeros(B_shape), torch.zeros(x_shape), _b, _M)


@pytest.mark.parametrize(
    ("x_shape", "D_shape"),
    [
        ((_b, _N), (_P, _F)),
        ((_b, _N), (_b, _P, _F)),
        ((_b, _N, _F), (_b,)),
        ((_b, _N, _F), (_P,)),
        ((_b, _N, _F), (_b, _P)),
        ((_b, _N, _F), (_b, _P, _F + 1)),
        ((_b, _N, _F), (1, 1, 1, 1)),
    ],
)
def test_ssm_D_rejects_invalid_shapes(x_shape, D_shape):
    with pytest.raises(ValueError, match=f"Input matrix D .* for {len(x_shape)}D"):
        _ssm_C_D(torch.zeros(()), torch.zeros(x_shape), None, torch.zeros(D_shape), _b, _M)


@pytest.mark.parametrize("C_shape", [(), (_M + 1,), (_b, _P, _M + 1), (1, 1, 1, _M)])
def test_ssm_C_rejects_invalid_shapes(C_shape):
    with pytest.raises(ValueError, match="Output matrix C"):
        _ssm_C_D(
            torch.zeros(_b, _N, _M),
            torch.zeros(_b, _N),
            torch.zeros(C_shape),
            None,
            _b,
            _M,
        )


# diag_state_space(L=...) alone: A = diag(L), V = I, and no M x M work.
def _diag_L_inputs(complex_L, batched_L, x_dim, with_B):
    batch, N, M, F = 3, 11, 4, 5
    L = 0.9 * torch.rand((batch, M) if batched_L else (M,), dtype=torch.double)
    if complex_L:
        L = torch.polar(L, torch.rand_like(L) * 6.0)
    x = torch.randn((batch, N) if x_dim == 2 else (batch, N, F if with_B else M), dtype=L.dtype)
    B = None
    if with_B:
        shape = {2: (M,), 3: (M, F)}[x_dim]
        B = torch.randn((batch, *shape) if batched_L else shape, dtype=L.dtype)
    return L, x, B


def _diag_L_reference(L, x, B, **kwargs):
    """state_space with A = diag(L), and B = ones for the 2-D x it feeds to
    every state."""
    if B is None and x.dim() == 2:
        B = torch.ones(L.shape, dtype=L.dtype)
    return state_space(torch.diag_embed(L), x, B=B, **kwargs)


@pytest.mark.parametrize("complex_L", [False, True])
@pytest.mark.parametrize("batched_L", [False, True])
@pytest.mark.parametrize("x_dim", [2, 3])
@pytest.mark.parametrize("with_B", [False, True])
def test_diag_state_space_L_alone_matches_state_space(complex_L, batched_L, x_dim, with_B):
    L, x, B = _diag_L_inputs(complex_L, batched_L, x_dim, with_B)

    y = diag_state_space(x, L=L, B=B)

    torch.testing.assert_close(y, _diag_L_reference(L, x, B))


@pytest.mark.parametrize("complex_L", [False, True])
@pytest.mark.parametrize("batched_L", [False, True])
@pytest.mark.parametrize("extras", ["C_D_zi", "out_idx_zi"])
def test_diag_state_space_L_alone_C_D_zi_out_idx(complex_L, batched_L, extras):
    L, x, B = _diag_L_inputs(complex_L, batched_L, 3, True)
    batch, _, F = x.shape
    M = L.size(-1)
    kwargs = {"zi": torch.randn(batch, M, dtype=L.dtype)}
    if extras == "C_D_zi":
        kwargs |= {
            "C": torch.randn(2, M, dtype=L.dtype),
            "D": torch.randn(2, F, dtype=L.dtype),
        }
    else:
        kwargs["out_idx"] = 2

    y, zf = diag_state_space(x, L=L, B=B, **kwargs)
    y_ref, zf_ref = _diag_L_reference(L, x, B, **kwargs)

    torch.testing.assert_close(y, y_ref)
    torch.testing.assert_close(zf, zf_ref)


class _RecordShapes(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.numels = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        self.numels += [t.numel() for t in pytree.tree_leaves(out) if isinstance(t, torch.Tensor)]
        return out


def test_diag_state_space_L_alone_allocates_no_M_by_M_tensor():
    M = 64
    L = 0.5 * torch.rand(1, M, dtype=torch.cdouble)
    x = torch.randn(1, 4, M, dtype=torch.cdouble)
    B = torch.randn(1, M, dtype=torch.cdouble)
    C = torch.randn(M, dtype=torch.cdouble)
    zi = torch.randn(1, M, dtype=torch.cdouble)

    with _RecordShapes() as recorder:
        diag_state_space(x, L=L)
        diag_state_space(x[..., 0], L=L)
        diag_state_space(x[..., 0], L=L, B=B)
        diag_state_space(x[..., 0], L=L, C=C, zi=zi)

    assert recorder.numels
    assert max(recorder.numels) < M * M


# A complex model runs in complex and gives complex outputs.
@pytest.mark.parametrize("system", ["A", "L"])
def test_diag_state_space_complex_model_gives_complex_outputs(system):
    if system == "A":
        A = _generate_diagonalizable_matrix((2, 2)).double() + 0j
        given, B = {"A": A}, None
    else:
        L = 0.9 * torch.rand(2, dtype=torch.double) * (1 + 0.5j)
        A, given, B = torch.diag(L), {"L": L}, torch.ones(2, dtype=torch.cdouble)
    kwargs = {
        "x": torch.randn(3, 9, dtype=torch.cdouble),
        "C": torch.randn(2, dtype=torch.cdouble),
        "D": torch.tensor(0.5, dtype=torch.cdouble),
        "zi": torch.randn(3, 2, dtype=torch.cdouble),
    }

    y, zf = diag_state_space(**given, B=B, **kwargs)
    y_ref, zf_ref = state_space(A, B=B, **kwargs)

    torch.testing.assert_close(y, y_ref)
    torch.testing.assert_close(zf, zf_ref)


def test_diag_state_space_real_eigenvectors_with_C():
    # A symmetric A has real eigenvalues and eigenvectors: real states.
    A = torch.tensor([[0.5, 0.2], [0.2, -0.3]], dtype=torch.double)
    L, V = torch.linalg.eigh(A)
    x = torch.randn(3, 9, dtype=torch.double)
    C = torch.randn(2, dtype=torch.double)

    torch.testing.assert_close(diag_state_space(x, L=L, V=V, C=C), state_space(A, x, C=C))


# One input and one output from rest: diag_state_space folds B and C into
# residues and runs pfefilt. Each system names its model to diag_state_space
# and has the dense A that state_space takes. A complex L with a real x needs
# V or Vinv beside it.
_PFE_BATCH, _PFE_N, _PFE_M = 3, 17, 4
_PFE_SYSTEMS = [
    "A_real",  # complex and real poles: the half form
    "A_complex",
    "LV_real",
    "LV_complex",  # a real A's complex L and V, with a real x: the full form
    "LVinv_complex",
    "L_real",
    "L_complex_x",
]


def _pfe_model(system, batched):
    batch = (_PFE_BATCH,) if batched else ()
    M = _PFE_M
    scale = torch.linspace(0.6, 1.0, _PFE_BATCH)[:, None, None] if batched else 1.0
    eye = torch.eye(M, dtype=torch.double)

    def similar(Lam, dtype):
        S = (eye + 0.3 * torch.randn(*batch, M, M, dtype=dtype)).to(dtype)
        return S @ Lam.to(dtype) @ torch.linalg.inv(S)

    # eigenvalues 0.5 +- 0.3j, 0.4 and -0.2
    Lam = torch.block_diag(
        torch.tensor([[0.5, 0.3], [-0.3, 0.5]]), torch.diag(torch.tensor([0.4, -0.2]))
    )
    Lam = scale * Lam.double()
    if system == "A_real":
        A = similar(Lam, torch.double)
    elif system == "A_complex":
        A = similar(
            torch.diag_embed(torch.randn(*batch, M, dtype=torch.cdouble)) * 0.4, torch.cdouble
        )
    elif system == "LV_real":
        Q = torch.linalg.qr(torch.randn(*batch, M, M, dtype=torch.double)).Q
        L = torch.rand(*batch, M, dtype=torch.double) - 0.5
        A = Q @ torch.diag_embed(L) @ Q.mT
        return {"L": L, "V": Q}, A, torch.double
    elif system in ("LV_complex", "LVinv_complex"):
        A = similar(Lam, torch.double)
        L, V = torch.linalg.eig(A)
        given = {"V": V} if system == "LV_complex" else {"Vinv": torch.linalg.inv(V)}
        return {"L": L, **given}, A, torch.double
    else:
        L = torch.rand(*batch, M, dtype=torch.double) - 0.5
        if system == "L_complex_x":
            L = torch.polar(L.abs(), torch.rand_like(L) * 6.0)
        dtype = L.dtype
        return {"L": L}, torch.diag_embed(L), dtype
    return {"A": A}, A, A.dtype


def _pfe_reference(A, x, B, C, D, out_idx, identity):
    """state_space of the dense A, with the model's B for a 2-D x."""
    if B is None and identity:
        B = torch.ones(A.shape[:-1], dtype=A.dtype)
    return state_space(A, x, B=B, C=C, D=D, out_idx=out_idx)


@pytest.mark.parametrize("system", _PFE_SYSTEMS)
@pytest.mark.parametrize("batched", [False, True])
# the output, and whether B is given: without B, x drives every state of a
# diagonal model and the first of the others
@pytest.mark.parametrize(
    ("output", "with_B"), [("C", True), ("C_per_signal", False), ("out_idx", True)]
)
def test_diag_state_space_pfefilt_route_matches_state_space(system, batched, output, with_B):
    given, A, dtype = _pfe_model(system, batched)
    M = _PFE_M
    x = torch.randn(_PFE_BATCH, _PFE_N, dtype=dtype)
    B = torch.randn(*((_PFE_BATCH,) if batched else ()), M, dtype=dtype) if with_B else None
    C = {"C": (M,), "C_per_signal": (_PFE_BATCH, M)}.get(output)
    C = None if C is None else torch.randn(C, dtype=dtype)
    out_idx = 2 if output == "out_idx" else None
    # a gain per signal with a batched model, one gain otherwise; the other
    # tests of diag_state_space run without D
    D = torch.randn((_PFE_BATCH,) if batched else (), dtype=dtype)

    y = diag_state_space(x, **given, B=B, C=C, D=D, out_idx=out_idx)

    y_ref = _pfe_reference(A, x, B, C, D, out_idx, identity=set(given) == {"L"})
    torch.testing.assert_close(y, y_ref)


# The route with an initial state: the modes' sections start from V^{-1} zi.
# A given zi always takes the per-signal layout, batched model or not.
@pytest.mark.parametrize("system", ["A_real", "LVinv_complex", "A_complex", "L_real"])
def test_diag_state_space_pfefilt_route_with_zi_matches_state_space(system):
    given, A, dtype = _pfe_model(system, batched=False)
    M = _PFE_M
    x = torch.randn(_PFE_BATCH, _PFE_N, dtype=dtype)
    B = torch.randn(M, dtype=dtype)
    C = torch.randn(_PFE_BATCH, M, dtype=dtype)
    D = torch.randn((), dtype=dtype)
    zi = torch.randn(_PFE_BATCH, M, dtype=dtype)

    y, zf = diag_state_space(x, **given, B=B, C=C, D=D, zi=zi)

    y_ref, zf_ref = state_space(A, x, B=B, C=C, D=D, zi=zi)
    torch.testing.assert_close(y, y_ref)
    torch.testing.assert_close(zf, zf_ref)


def test_diag_state_space_pfefilt_route_with_one_zi_for_all_signals():
    given, A, dtype = _pfe_model("A_real", batched=False)
    x = torch.randn(_PFE_BATCH, _PFE_N, dtype=dtype)
    zi = torch.randn(_PFE_M, dtype=dtype)

    y, zf = diag_state_space(x, **given, out_idx=2, zi=zi)

    y_ref, zf_ref = state_space(A, x, out_idx=2, zi=zi)
    torch.testing.assert_close(y, y_ref)
    torch.testing.assert_close(zf, zf_ref)


def test_diag_state_space_with_zi_and_a_mode_that_b_cannot_reach():
    # a subnormal entry of B, which overflows q[0] / B as a zero would divide it
    L = torch.tensor([0.5, -0.3, 0.2], dtype=torch.double)
    B = torch.tensor([1.0, 1e-310, 2.0], dtype=torch.double)
    C, zi = torch.randn(3, dtype=torch.double), torch.randn(_PFE_BATCH, 3, dtype=torch.double)
    x = torch.randn(_PFE_BATCH, _PFE_N, dtype=torch.double)

    y, zf = diag_state_space(x, L=L, B=B, C=C, zi=zi)

    y_ref, zf_ref = state_space(torch.diag(L), x, B=B, C=C, zi=zi)
    torch.testing.assert_close(y, y_ref)
    torch.testing.assert_close(zf, zf_ref)


def test_diag_state_space_pfefilt_route_with_zi_and_no_steps():
    given, _, dtype = _pfe_model("A_real", batched=False)
    zi = torch.randn(_PFE_BATCH, _PFE_M, dtype=dtype)

    y, zf = diag_state_space(torch.empty(_PFE_BATCH, 0, dtype=dtype), **given, out_idx=1, zi=zi)

    assert y.shape == (_PFE_BATCH, 0)
    torch.testing.assert_close(zf, zi)


def test_diag_state_space_diagonal_A():
    # a repeated eigenvalue with two eigenvectors
    A = torch.diag(torch.tensor([0.5, 0.5], dtype=torch.double))
    x = torch.randn(_PFE_BATCH, _PFE_N, dtype=torch.double)
    B, C = torch.randn(2, dtype=torch.double), torch.randn(2, dtype=torch.double)

    torch.testing.assert_close(diag_state_space(x, A=A, B=B, C=C), state_space(A, x, B=B, C=C))


def test_diag_state_space_integer_model():
    # pfefilt doesn't take integers: the integer recurrence runs them
    x = torch.ones(1, 4, dtype=torch.long)
    L, C = torch.zeros(1, dtype=torch.long), torch.ones(1, dtype=torch.long)

    y = diag_state_space(x, L=L, C=C)

    torch.testing.assert_close(y, torch.tensor([[0, 1, 1, 1]]))


def test_diag_state_space_pfefilt_route_without_signals():
    given, _, dtype = _pfe_model("A_real", batched=False)
    x = torch.empty(0, _PFE_N, dtype=dtype)
    C, D = torch.randn(_PFE_M, dtype=dtype), torch.randn(0, dtype=dtype)

    y = diag_state_space(x, **given, C=C, D=D)

    assert y.shape == (0, _PFE_N)


def test_diag_state_space_one_output_in_bfloat16():
    # pfefilt doesn't take bfloat16, which has no complex dtype
    L = (torch.rand(_PFE_M) - 0.5).bfloat16()
    x, C = torch.randn(_PFE_BATCH, _PFE_N).bfloat16(), torch.randn(_PFE_M).bfloat16()

    y = diag_state_space(x, L=L, C=C)

    # bfloat16 states, rounded at every step
    y_ref = _pfe_reference(torch.diag(L.float()), x.float(), None, C.float(), None, None, True)
    torch.testing.assert_close(y.float(), y_ref, rtol=3e-2, atol=3e-2)


@pytest.mark.parametrize("system", ["A_real", "LVinv_complex"])
def test_diag_state_space_out_idx_off_the_route(system):
    # an integer unroll_factor keeps the modal path, which takes row out_idx of V
    given, A, dtype = _pfe_model(system, batched=False)
    x = torch.randn(_PFE_BATCH, _PFE_N, dtype=dtype)

    y = diag_state_space(x, **given, out_idx=2, unroll_factor=4)

    torch.testing.assert_close(y, state_space(A, x, out_idx=2))
