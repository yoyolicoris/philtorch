import pytest
import torch

from philtorch._recur import _forward, _triton_applies, recurrence

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
def test_triton_covers_cuda_without_a_native_kernel():
    x = torch.zeros(1, 1, 1, device="cuda")
    assert _triton_applies(x, 128) and not _triton_applies(x, 129)
    assert _triton_applies(x.cfloat(), 64) and not _triton_applies(x.cfloat(), 65)
    assert not _triton_applies(x.cpu(), 4)


def test_recurrence_rejects_bad_shapes():
    A, zi, x = _problem(2, 5, 3, "both")
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
