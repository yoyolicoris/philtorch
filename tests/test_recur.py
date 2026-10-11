import pytest
import torch

from philtorch import _recur
from philtorch._recur import _forward, recur_triton, recurrence

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
DEVICES = ["cpu", pytest.param("cuda", marks=requires_cuda)]
# How A is shared: (batch items, steps) it has, of B and T.
SHARING = ["one", "per_item", "per_step", "both"]


def _sequential(A, zi, x):
    """h[t] = A[t] h[t - 1] + x[t], one step at a time."""
    h, states = zi, []
    for t in range(x.size(1)):
        h = (A[:, t if A.size(1) > 1 else 0] @ h.unsqueeze(-1)).squeeze(-1) + x[:, t]
        states.append(h)
    return torch.stack(states, 1)


def _problem(B, T, M, sharing, dtype=torch.float64, device="cpu", seed=0):
    """A stable recurrence: each A[t] scaled to spectral norm 0.97."""
    gen = torch.Generator().manual_seed(seed)
    Ba = B if sharing in ("per_item", "both") else 1
    Ta = T if sharing in ("per_step", "both") else 1
    A = torch.randn(Ba, Ta, M, M, dtype=torch.complex128 if dtype.is_complex else torch.float64,
                    generator=gen)  # fmt: skip
    A = A / torch.linalg.matrix_norm(A, ord=2, keepdim=True) * 0.97
    x = torch.randn(B, T, M, dtype=A.dtype, generator=gen)
    zi = torch.randn(B, M, dtype=A.dtype, generator=gen)
    return (t.to(device, dtype) for t in (A, zi, x))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("sharing", SHARING)
@pytest.mark.parametrize("M", [1, 2, 3, 5, 20, 40, 100])
@pytest.mark.parametrize("dtype", [torch.float64, torch.complex128])
def test_recurrence_matches_sequential(device, sharing, M, dtype):
    if device == "cuda" and dtype.is_complex and M > 64:
        pytest.skip("past the Triton kernels' complex states")
    # 150 steps: three chunks of the Triton scan, the last one partial.
    A, zi, x = _problem(3, 150, M, sharing, dtype, device)
    torch.testing.assert_close(recurrence(A, zi, x), _sequential(A, zi, x))


@requires_cuda
@pytest.mark.parametrize("sharing", SHARING)
@pytest.mark.parametrize("M", [3, 8, 24])
@pytest.mark.parametrize("T", [50, 300])
def test_recurrence_large_batch(sharing, M, T):
    # Sequences that fit in one chunk, which runs its steps alone, and longer
    # ones, scanned whatever the batch.
    A, zi, x = _problem(256, T, M, sharing, device="cuda")
    torch.testing.assert_close(recurrence(A, zi, x), _sequential(A, zi, x))


@requires_cuda
@pytest.mark.parametrize("sharing", ["one", "both"])
@pytest.mark.parametrize("M", [3, 8])
def test_recurrence_many_chunks_repeatedly(sharing, M):
    # Hundreds of chunks, whose starts come from look-backs that race their
    # predecessors' publishing: every run must still be right.
    A, zi, x = _problem(4, 40000, M, sharing, device="cuda")
    expected = _sequential(A, zi, x)
    for _ in range(20):
        torch.testing.assert_close(recurrence(A, zi, x), expected)


@requires_cuda
@pytest.mark.parametrize("dtype, M", [(torch.float64, 40), (torch.complex128, 24)])
def test_recurrence_maps_in_scratch(dtype, M):
    # Time-varying maps too large for registers: their products through
    # scratch memory, across chunks whose look-backs race.
    A, zi, x = _problem(2, 1500, M, "both", dtype, "cuda")
    expected = _sequential(A, zi, x)
    for _ in range(3):
        torch.testing.assert_close(recurrence(A, zi, x), expected)


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.complex64])
def test_recurrence_ignores_memory_past_the_states(dtype):
    # The scratch tiles' rows from M on are never written: NaN left in freed
    # memory, where the allocator places them, must not reach the states.
    A, zi, x = _problem(2, 1500, 40 if dtype == torch.float32 else 24, "both", dtype, "cuda")
    for _ in range(4):
        torch.full((2**24,), float("nan"), device="cuda")
    assert not recurrence(A, zi, x).isnan().any()


@requires_cuda
@pytest.mark.parametrize(
    "dtype, tolerance",
    [
        (torch.float32, 1e-5),
        (torch.complex64, 1e-5),
        (torch.float16, 3e-3),
        (torch.bfloat16, 2e-2),
    ],
)
@pytest.mark.parametrize("sharing", SHARING)
@pytest.mark.parametrize("M", [1, 2, 3, 6])
def test_recurrence_low_precision(dtype, tolerance, sharing, M):
    A, zi, x = _problem(4, 300, M, sharing, dtype, "cuda")
    expected = _sequential(*(t.to(torch.complex128 if dtype.is_complex else torch.float64)
                             for t in (A, zi, x)))  # fmt: skip
    actual = recurrence(A, zi, x)
    assert actual.dtype == dtype
    error = (actual.to(expected.dtype) - expected).abs().max() / expected.abs().max()
    assert error < tolerance


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("sharing", SHARING)
@pytest.mark.parametrize("M", [1, 2, 3, 5])
@pytest.mark.parametrize("dtype", [torch.float64, torch.complex128])
def test_recurrence_derivatives(device, sharing, M, dtype):
    # Across a chunk boundary of the Triton scan, in both modes, to second order.
    A, zi, x = _problem(2, 70, M, sharing, dtype, device)
    inputs = tuple(t.requires_grad_() for t in (A, zi, x))
    assert torch.autograd.gradcheck(recurrence, inputs, check_forward_ad=True, fast_mode=True)
    assert torch.autograd.gradgradcheck(recurrence, inputs, fast_mode=True)


@requires_cuda
def test_recurrence_cuda_matches_cpu_gradients():
    A, zi, x = _problem(3, 200, 4, "both")
    grads = []
    for device in ("cpu", "cuda"):
        inputs = [t.to(device).requires_grad_() for t in (A, zi, x)]
        h = recurrence(*inputs)
        grads.append(torch.autograd.grad((h * h).sum(), inputs))
    for cpu, cuda in zip(*grads):
        torch.testing.assert_close(cuda.cpu(), cpu)


@pytest.mark.parametrize("device", DEVICES)
def test_recurrence_empty(device):
    A, zi, x = _problem(2, 5, 3, "both", device=device)
    assert recurrence(A[:, :0], zi, x[:, :0]).shape == (2, 0, 3)


@requires_cuda
def test_triton_state_limits():
    def applies(M, dtype=torch.float32, device="cuda"):
        A = torch.zeros(1, 1, M, M, dtype=dtype, device=device)
        return _recur._triton_applies(A, torch.zeros(1, 1, M, dtype=dtype, device=device))

    assert applies(128) and not applies(129)
    assert applies(64, torch.complex64) and not applies(65, torch.complex64)
    assert not applies(4, device="cpu")


def _kernel(time_invariant, M, dtype, device, B=1):
    """The launch _forward picks for A and x of these properties."""
    A = torch.zeros(1, 1 if time_invariant else 2, M, M, dtype=dtype, device=device)
    x = torch.zeros(B, 2, M, dtype=dtype, device=device)
    return next(run for applies, run in _recur._KERNELS if applies(A, x))


@pytest.mark.parametrize(
    "time_invariant, M, dtype, device, B, kernel",
    [
        (True, 1, torch.float64, "cpu", 1, "_lti_recur"),
        (True, 3, torch.float64, "cpu", 1, "_lti_recurN"),
        (False, 1, torch.float64, "cpu", 1, "_recurN"),
        *(
            pytest.param(*case, marks=requires_cuda)
            for case in [
                (True, 1, torch.float16, "cuda", 1, "_lti_recur"),
                (True, 2, torch.complex64, "cuda", 1, "_lti_recur2"),
                (True, 3, torch.float32, "cuda", 1, "recur_triton"),
                (False, 1, torch.complex64, "cuda", 1, "_scan"),
                (False, 1, torch.float16, "cuda", 1, "recur_triton"),
                (False, 2, torch.float32, "cuda", 1, "_pararnn"),
                (False, 3, torch.float64, "cuda", 1, "_pararnn"),
                (False, 2, torch.float32, "cuda", 65536, "_recur2"),
                (False, 2, torch.float16, "cuda", 1, "_recur2"),
                (False, 2, torch.complex64, "cuda", 1, "recur_triton"),
                (False, 3, torch.float16, "cuda", 1, "recur_triton"),
            ]
        ),
    ],
)
def test_dispatch(time_invariant, M, dtype, device, B, kernel):
    assert _kernel(time_invariant, M, dtype, device, B) is getattr(_recur, kernel)


@requires_cuda
@pytest.mark.parametrize("sharing", SHARING)
def test_recur_triton_custom_op(sharing):
    # The fake implementation and the op's registration, for torch.compile.
    A, zi, x = _problem(2, 70, 3, sharing, torch.float32, "cuda")
    torch.library.opcheck(recur_triton, (A, zi, x))


def test_recurrence_rejects_bad_shapes():
    A, zi, x = _problem(2, 5, 3, "both")
    with pytest.raises(ValueError, match="x must be"):
        recurrence(A, zi, x[0])
    with pytest.raises(ValueError, match="A must be"):
        recurrence(A[0], zi, x)
    with pytest.raises(ValueError, match="A must be"):
        recurrence(A[:, :2], zi, x)
    with pytest.raises(ValueError, match="zi must be"):
        recurrence(A, zi[:1], x)
    with pytest.raises(ValueError, match="share a dtype and a device"):
        recurrence(A.float(), zi, x)


def test_recurrence_raises_where_no_kernel_applies():
    # A device with no kernels, standing in for e.g. MPS beyond its one kernel.
    A, zi, x = _problem(2, 7, 3, "both", device="meta")
    with pytest.raises(NotImplementedError, match="time-varying A with M = 3 .* on meta"):
        _forward(A, zi, x)


@requires_cuda
def test_recurrence_raises_past_the_largest_triton_state():
    A, zi, x = _problem(2, 7, 129, "one", device="cuda")
    with pytest.raises(NotImplementedError, match="take M up to 128"):
        recurrence(A, zi, x)
