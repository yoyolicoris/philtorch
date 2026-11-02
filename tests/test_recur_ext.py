import pytest
import torch

from philtorch._recur import recurrence
from philtorch.lpv import linear_recurrence as lpv_linear_recurrence
from philtorch.lpv import state_space_recursion as lpv_state_space
from philtorch.lti import linear_recurrence as lti_linear_recurrence
from philtorch.mat import companion

from .test_lpv_filters import _generate_time_varying_coeffs
from .test_lti_ssm import _generate_random_filter_coeffs

_REQUIRES_GRAD_CASES = [
    (True, False, False),
    (False, True, False),
    (False, False, True),
    (True, True, False),
    (True, False, True),
    (False, True, True),
    (True, True, True),
]


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.cuda,
        ),
    ],
)
@pytest.mark.parametrize("batch", [True, False])
def test_lti_recur2_equiv(device: str, batch: bool):
    B = 3
    T = 101
    order = 2

    a = _generate_random_filter_coeffs(order, B if batch else 1)

    # Convert to torch tensors
    a_torch = torch.from_numpy(a).to(device)
    x_torch = torch.randn(B, T, order).to(device).to(dtype=a_torch.dtype)
    A = companion(a_torch).squeeze(0)

    zi = x_torch.new_zeros(B, order).normal_()

    lti_y = torch.ops.philtorch.lti_recur2(A, zi, x_torch)
    ltv_y = torch.ops.philtorch.recur2(
        A.unsqueeze(1).expand(-1, T, -1, -1) if batch else A.expand(T, -1, -1),
        zi,
        x_torch,
    )
    assert torch.allclose(lti_y, ltv_y), torch.max(torch.abs(lti_y - ltv_y))


@pytest.mark.parametrize("device", ["cpu"])
@pytest.mark.parametrize("batch", [True, False])
@pytest.mark.parametrize("order", [3, 5])
def test_lti_recurN_equiv(device: str, batch: bool, order: int):
    B = 4
    T = 101

    a = _generate_random_filter_coeffs(order, B if batch else 1)

    # Convert to torch tensors
    a_torch = torch.from_numpy(a).to(device)
    x_torch = torch.randn(B, T, order).to(device).to(dtype=a_torch.dtype)
    A = companion(a_torch).squeeze(0)

    zi = x_torch.new_zeros(B, order).normal_()

    lti_y = torch.ops.philtorch.lti_recurN(A, zi, x_torch)
    ltv_y = torch.ops.philtorch.recurN(
        A.unsqueeze(1).expand(-1, T, -1, -1) if batch else A.expand(T, -1, -1),
        zi,
        x_torch,
    )
    assert torch.allclose(lti_y, ltv_y)


@pytest.mark.parametrize(
    "dtype",
    [
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
        torch.complex64,
        torch.complex128,
        torch.int64,
    ],
)
@pytest.mark.parametrize("batch", [True, False])
def test_lti_recurN_cpu_dispatch_equiv(dtype: torch.dtype, batch: bool):
    batch_size = 3
    samples = 19
    order = 17
    matrix_shape = (batch_size, order, order) if batch else (order, order)
    if dtype == torch.int64:
        A = torch.zeros(matrix_shape, dtype=dtype)
        diagonal = torch.arange(order)
        A[..., diagonal, diagonal] = 1
        zi = torch.randint(-2, 3, (batch_size, order), dtype=dtype)
        x = torch.randint(-2, 3, (batch_size, samples, order), dtype=dtype)
    else:
        A = torch.randn(matrix_shape, dtype=dtype) / (2 * order)
        zi = torch.randn(batch_size, order, dtype=dtype)
        x = torch.randn(batch_size, samples, order, dtype=dtype)

    expected = torch.ops.philtorch.recurN(
        (A.unsqueeze(1).expand(-1, samples, -1, -1) if batch else A.expand(samples, -1, -1)),
        zi,
        x,
    )
    actual = torch.ops.philtorch.lti_recurN(A, zi, x)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        "meta",
        pytest.param(
            "cuda",
            marks=pytest.mark.cuda,
        ),
        pytest.param(
            "mps",
            marks=pytest.mark.skipif(
                not torch.backends.mps.is_available(), reason="MPS not available"
            ),
        ),
    ],
)
def test_lti_recur_rejects_zero_length(device: str):
    a = torch.empty(1, device=device)
    zi = torch.empty(2, device=device)
    x = torch.empty(2, 0, device=device)

    with pytest.raises(RuntimeError, match="at least one time step"):
        torch.ops.philtorch.lti_recur(a, zi, x)


_MPS = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")

# The dtypes of the Metal scalar recurrence, and the tolerance of each against a
# float64 (complex128) reference computed from the same inputs. float16 and
# bfloat16 accumulate in float32 and round only the output, to their storage
# dtype.
_MPS_DTYPES = [torch.float32, torch.float16, torch.bfloat16, torch.complex64]
_MPS_TOLERANCES = {
    torch.float32: {"rtol": 1e-5, "atol": 1e-5},
    torch.complex64: {"rtol": 1e-5, "atol": 1e-5},
    torch.float16: {"rtol": 2e-3, "atol": 2e-3},
    torch.bfloat16: {"rtol": 1.6e-2, "atol": 1.6e-2},
}
# A tile is 1024 steps, read in vectors of 4: lengths around the vector width,
# the tile, and a few tiles.
_MPS_LENGTHS = [1, 2, 3, 4, 5, 1023, 1024, 1025, 2049, 4099]
_MPS_OPS = {"lti_recur": torch.ops.philtorch.lti_recur, "scan": torch.ops.philtorch.scan}


def _reference_dtype(dtype: torch.dtype) -> torch.dtype:
    return torch.complex128 if dtype.is_complex else torch.float64


def _mps_randn(shape, dtype: torch.dtype, gen: torch.Generator) -> torch.Tensor:
    wide = torch.complex64 if dtype.is_complex else torch.float32
    return torch.randn(shape, generator=gen, dtype=wide).to(dtype)


def _mps_decay(shape, dtype: torch.dtype, gen: torch.Generator) -> torch.Tensor:
    """Coefficients of magnitude below 0.95, with a random sign or phase."""
    magnitude = 0.95 * torch.rand(shape, generator=gen)
    if dtype.is_complex:
        return torch.polar(magnitude, torch.rand(shape, generator=gen) * 2 * torch.pi).to(dtype)
    return (magnitude * (torch.randint(0, 2, shape, generator=gen) * 2 - 1)).to(dtype)


def _mps_args(op: str, dtype: torch.dtype, a_shape, batch_size: int, samples: int):
    """The arguments, on CPU, of ``lti_recur`` (a, zi, x) or ``scan`` (x, decay, zi).

    For ``scan``, ``a_shape`` is the shape of the decay.
    """
    gen = torch.Generator().manual_seed(0)
    a = _mps_decay(a_shape, dtype, gen)
    zi = _mps_randn((batch_size,), dtype, gen)
    x = _mps_randn((batch_size, samples), dtype, gen)
    return (a, zi, x) if op == "lti_recur" else (x, a, zi)


def _assert_mps_matches_reference(op: str, args, dtype: torch.dtype, **tolerance):
    """Run the op on MPS, and compare with the CPU op's float64 run of the same values."""
    reference_dtype = _reference_dtype(dtype)
    expected = _MPS_OPS[op](*(arg.to(reference_dtype) for arg in args))
    actual = _MPS_OPS[op](*(arg.to("mps") for arg in args))
    assert actual.dtype == dtype
    tolerance = tolerance or _MPS_TOLERANCES[dtype]
    torch.testing.assert_close(actual.cpu().to(reference_dtype), expected, **tolerance)


@_MPS
@pytest.mark.parametrize("dtype", _MPS_DTYPES)
@pytest.mark.parametrize("coefficient_layout", ["scalar", "shared", "batched"])
@pytest.mark.parametrize("samples", _MPS_LENGTHS)
def test_lti_recur_mps_boundaries(dtype: torch.dtype, coefficient_layout: str, samples: int):
    batch_size = 3
    coefficient_shape = {
        "scalar": (),
        "shared": (1,),
        "batched": (batch_size,),
    }[coefficient_layout]
    args = _mps_args("lti_recur", dtype, coefficient_shape, batch_size, samples)
    _assert_mps_matches_reference("lti_recur", args, dtype)


@_MPS
@pytest.mark.parametrize("dtype", _MPS_DTYPES)
@pytest.mark.parametrize("samples", _MPS_LENGTHS)
def test_scan_mps_boundaries(dtype: torch.dtype, samples: int):
    batch_size = 3
    args = _mps_args("scan", dtype, (batch_size, samples), batch_size, samples)
    _assert_mps_matches_reference("scan", args, dtype)


@_MPS
@pytest.mark.parametrize("dtype", [torch.float32, torch.complex64])
@pytest.mark.parametrize("op", ["lti_recur", "scan"])
def test_mps_scalar_recur_block_chunks(op: str, dtype: torch.dtype):
    # Past 1024 * 512 steps, the tiles' totals are scanned in several chunks of
    # 512 and each chunk's carry reaches the next. The coefficients are close to
    # 1, so that a tile's total still counts in the tiles after it (a ** 1024 is
    # at least 6e-3): the states then reach 60 and float32's own error, CPU's
    # included, is a few 1e-5, so atol is 1e-3, which a missing carry exceeds.
    batch_size, samples = 2, 1024 * 512 + 5
    gen = torch.Generator().manual_seed(0)
    shape = (batch_size,) if op == "lti_recur" else (batch_size, samples)
    a = 1 - 5e-3 * torch.rand(shape, generator=gen)
    if dtype.is_complex:
        a = torch.polar(a, 0.05 * torch.rand(shape, generator=gen))
    zi = _mps_randn((batch_size,), dtype, gen)
    x = _mps_randn((batch_size, samples), dtype, gen)
    args = (a, zi, x) if op == "lti_recur" else (x, a, zi)
    _assert_mps_matches_reference(op, args, dtype, rtol=1e-5, atol=1e-3)


@_MPS
@pytest.mark.parametrize("dtype", _MPS_DTYPES)
@pytest.mark.parametrize("op", ["lti_recur", "scan"])
def test_mps_scalar_recur_sliced_inputs(op: str, dtype: torch.dtype):
    # Views of larger tensors, with an odd storage offset and rows that don't
    # touch, which the op makes contiguous: its vector loads need element
    # alignment only.
    B, T = 3, 2049
    gen = torch.Generator().manual_seed(0)
    big_x = _mps_randn((B + 1, T + 5), dtype, gen)
    big_zi = _mps_randn((B + 2,), dtype, gen)
    x_view = lambda t: t[1:, 3 : 3 + T]  # noqa: E731
    batch_view = lambda t: t[1 : 1 + B]  # noqa: E731
    if op == "lti_recur":
        bigs = (_mps_decay((B + 2,), dtype, gen), big_zi, big_x)
        views = (batch_view, batch_view, x_view)
    else:
        bigs = (big_x, _mps_decay((B + 1, T + 5), dtype, gen), big_zi)
        views = (x_view, lambda t: t[1:, 1 : 1 + T], batch_view)
    mps_views = [view(big.to("mps")) for view, big in zip(views, bigs)]
    assert all(v.storage_offset() % 2 == 1 for v in mps_views)
    assert not mps_views[2 if op == "lti_recur" else 0].is_contiguous()

    reference_dtype = _reference_dtype(dtype)
    expected = _MPS_OPS[op](*(view(big).to(reference_dtype) for view, big in zip(views, bigs)))
    actual = _MPS_OPS[op](*mps_views)
    torch.testing.assert_close(actual.cpu().to(reference_dtype), expected, **_MPS_TOLERANCES[dtype])


def _outputs_and_gradients(fn, args, requires_grad, grad_output, device: str):
    """The output of ``fn(*args)`` on a device, and the gradients of the arguments that need one."""
    device_args = tuple(
        arg.to(device).detach().requires_grad_(needs_grad)
        for arg, needs_grad in zip(args, requires_grad)
    )
    output = fn(*device_args)
    computed = iter(
        torch.autograd.grad(
            output,
            [arg for arg, needs_grad in zip(device_args, requires_grad) if needs_grad],
            grad_outputs=grad_output.to(device),
        )
    )
    gradients = tuple(next(computed).cpu() if needs_grad else None for needs_grad in requires_grad)
    return output.detach().cpu(), gradients


def _assert_mps_gradients_match_cpu(fn, args, requires_grad, grad_output):
    expected_output, expected_gradients = _outputs_and_gradients(
        fn, args, requires_grad, grad_output, "cpu"
    )
    actual_output, actual_gradients = _outputs_and_gradients(
        fn, args, requires_grad, grad_output, "mps"
    )
    torch.testing.assert_close(actual_output, expected_output, rtol=1e-5, atol=1e-6)
    for actual, expected in zip(actual_gradients, expected_gradients):
        if actual is None or expected is None:
            assert actual is expected is None
        else:
            torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)


@_MPS
@pytest.mark.parametrize("dtype", [torch.float32, torch.complex64])
@pytest.mark.parametrize(
    ("x_requires_grad", "a_requires_grad", "zi_requires_grad"),
    _REQUIRES_GRAD_CASES,
)
@pytest.mark.parametrize("batch_decay", [True, False])
def test_lti_recur_mps_grad_equiv(
    dtype: torch.dtype,
    x_requires_grad: bool,
    a_requires_grad: bool,
    zi_requires_grad: bool,
    batch_decay: bool,
):
    batch_size = 3
    samples = 17
    gen = torch.Generator().manual_seed(0)
    a = _mps_decay((batch_size if batch_decay else 1,), dtype, gen) * 0.4
    zi = _mps_randn((batch_size,), dtype, gen)
    x = _mps_randn((batch_size, samples), dtype, gen)
    grad_output = _mps_randn(x.shape, dtype, gen)

    _assert_mps_gradients_match_cpu(
        torch.ops.philtorch.lti_recur,
        (a, zi, x),
        (a_requires_grad, zi_requires_grad, x_requires_grad),
        grad_output,
    )


@_MPS
@pytest.mark.parametrize("dtype", [torch.float32, torch.complex64])
@pytest.mark.parametrize(
    ("x_requires_grad", "decay_requires_grad", "zi_requires_grad"),
    _REQUIRES_GRAD_CASES,
)
@pytest.mark.parametrize("samples", [17, 2049])
def test_scan_mps_grad_equiv(
    dtype: torch.dtype,
    x_requires_grad: bool,
    decay_requires_grad: bool,
    zi_requires_grad: bool,
    samples: int,
):
    batch_size = 3
    gen = torch.Generator().manual_seed(0)
    x = _mps_randn((batch_size, samples), dtype, gen)
    decay = _mps_decay((batch_size, samples), dtype, gen) * 0.8
    zi = _mps_randn((batch_size,), dtype, gen)
    grad_output = _mps_randn(x.shape, dtype, gen)

    _assert_mps_gradients_match_cpu(
        torch.ops.philtorch.scan,
        (x, decay, zi),
        (x_requires_grad, decay_requires_grad, zi_requires_grad),
        grad_output,
    )


@_MPS
@pytest.mark.parametrize("dtype", [torch.float32, torch.complex64])
@pytest.mark.parametrize(
    ("linear_recurrence", "a_shape", "init_shape"),
    [
        pytest.param(lti_linear_recurrence, (), (), id="lti-scalar"),
        pytest.param(lti_linear_recurrence, (3,), (3,), id="lti-batched"),
        pytest.param(lpv_linear_recurrence, (1500,), (), id="lpv-shared"),
        pytest.param(lpv_linear_recurrence, (3, 1500), (3,), id="lpv-batched"),
    ],
)
def test_mps_linear_recurrence_matches_cpu(linear_recurrence, a_shape, init_shape, dtype):
    # The public scalar recurrences, forward and backward, which reach the Metal
    # kernels through the filters' engine.
    gen = torch.Generator().manual_seed(0)
    a = _mps_decay(a_shape, dtype, gen)
    init = _mps_randn(init_shape, dtype, gen)
    x = _mps_randn((3, 1500), dtype, gen)
    grad_output = _mps_randn(x.shape, dtype, gen)

    _assert_mps_gradients_match_cpu(
        linear_recurrence, (a, init, x), (True, True, True), grad_output
    )


@_MPS
@pytest.mark.parametrize("time_invariant", [True, False])
@pytest.mark.parametrize("dtype", [torch.float32, torch.int64])
def test_mps_recurrence_rejects_what_the_kernels_do_not_take(time_invariant: bool, dtype):
    # M = 2 in float32, and int64 with M = 1, which the CPU kernels take.
    M = 2 if dtype == torch.float32 else 1
    B, T = 2, 5
    A = torch.zeros(1, 1 if time_invariant else T, M, M, dtype=dtype, device="mps")
    zi = torch.zeros(B, M, dtype=dtype, device="mps")
    x = torch.zeros(B, T, M, dtype=dtype, device="mps")
    with pytest.raises(NotImplementedError, match="Metal kernels take M = 1"):
        recurrence(A, zi, x)


@_MPS
@pytest.mark.parametrize("op", ["lti_recur", "scan"])
def test_mps_scalar_recur_rejects_int64(op: str):
    ones = lambda *shape: torch.ones(shape, dtype=torch.int64, device="mps")  # noqa: E731
    args = (
        (ones(2), ones(2), ones(2, 5)) if op == "lti_recur" else (ones(2, 5), ones(2, 5), ones(2))
    )
    with pytest.raises(RuntimeError, match="MPS scalar recurrence takes"):
        _MPS_OPS[op](*args)


@_MPS
@pytest.mark.parametrize("op", ["lti_recur", "scan"])
def test_mps_scalar_recur_opcheck(op: str):
    # The schema and fake of each op, and its AOT dispatch, on MPS. With inputs
    # that require gradients, opcheck's autograd-registration check stops at
    # "NYI devices other than CPU/CUDA/XPU", which is PyTorch's, not the op's.
    args = _mps_args(op, torch.float32, (2,) if op == "lti_recur" else (2, 70), 2, 70)
    torch.library.opcheck(_MPS_OPS[op], tuple(arg.to("mps") for arg in args))


@pytest.mark.parametrize("order", [2, 3])
def test_recurN_extension_matches_unrolled(order: int):
    """Test direct recurN low/general orders against the block-unrolled recursion."""
    batch_size = 2
    N = 37

    _, a = _generate_time_varying_coeffs(batch_size, N, order, order)
    x = torch.randn(batch_size, N, order).double()
    A = companion(a).double()
    zi = torch.randn(batch_size, order).double()

    ext_output = torch.ops.philtorch.recurN(A, zi, x)
    torch_output = lpv_state_space(A, zi, x, unroll_factor=2)

    assert torch.allclose(ext_output, torch_output), torch.max(torch.abs(ext_output - torch_output))


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.cuda,
        ),
    ],
)
def test_recur2_extension_matches_unrolled(device):
    """Test that the recur2 extension matches the block-unrolled recursion."""
    batch_size = 2
    N = 17
    order = 2

    _, a = _generate_time_varying_coeffs(batch_size, N, order, order)
    x = torch.randn(batch_size, N, 2).to(device).double()
    A = companion(a).to(device).double()
    zi = torch.randn(batch_size, order).to(device).double()

    ext_output = torch.ops.philtorch.recur2(A, zi, x)
    torch_output = lpv_state_space(A, zi, x, unroll_factor=2)

    assert torch.allclose(ext_output, torch_output), torch.max(torch.abs(ext_output - torch_output))


def _cuda_scan_case(name: str):
    """Return an op, its CPU arguments, and the output it should give."""
    B, T = 4, 4096
    gen = torch.Generator().manual_seed(0)

    def randn(*shape):
        return torch.randn(shape, generator=gen, dtype=torch.float64)

    def stable(a):
        # Keep the recurrence bounded over T steps: |a| < 1, ||A||_2 < 1.
        if a.dim() >= 2:
            return a * 0.9 / torch.linalg.matrix_norm(a, ord=2)[..., None, None]
        return torch.tanh(a) * 0.9

    if name == "lpc":
        op, args = torch.ops.philtorch.lpc, (randn(B, T), randn(B, T, 2) * 0.2, randn(B, 2))
    elif name == "scan":
        op, args = torch.ops.philtorch.scan, (randn(B, T), stable(randn(B, T)), randn(B))
    else:
        op, a_shape = {
            "lti_recur-batched": (torch.ops.philtorch.lti_recur, (B,)),
            "lti_recur-shared": (torch.ops.philtorch.lti_recur, ()),
            "lti_recur2-batched": (torch.ops.philtorch.lti_recur2, (B, 2, 2)),
            "lti_recur2-shared": (torch.ops.philtorch.lti_recur2, (2, 2)),
            "recur2-batched": (torch.ops.philtorch.recur2, (B, T, 2, 2)),
            "recur2-shared": (torch.ops.philtorch.recur2, (T, 2, 2)),
            "recur3-batched": (torch.ops.philtorch.recur3, (B, T, 3, 3)),
            "recur3-shared": (torch.ops.philtorch.recur3, (T, 3, 3)),
            "lti_recur3-batched": (torch.ops.philtorch.lti_recur3, (B, 3, 3)),
            "lti_recur3-shared": (torch.ops.philtorch.lti_recur3, (3, 3)),
        }[name]
        if op is torch.ops.philtorch.lti_recur:
            args = (stable(randn(*a_shape)), randn(B), randn(B, T))
        else:
            M = a_shape[-1]
            args = (stable(randn(*a_shape)), randn(B, M), randn(B, T, M))
        if op in (torch.ops.philtorch.recur3, torch.ops.philtorch.lti_recur3):
            args = tuple(arg.float() for arg in args)  # their CUDA kernel's dtype
    return op, args, op(*args)


@pytest.mark.cuda
@pytest.mark.parametrize(
    "case",
    [
        "lti_recur-batched",
        "lti_recur-shared",
        "lti_recur2-batched",
        "lti_recur2-shared",
        "recur2-batched",
        "recur2-shared",
        "recur3-batched",
        "recur3-shared",
        "lti_recur3-batched",
        "lti_recur3-shared",
        "lpc",
        "scan",
    ],
)
def test_cuda_scan_runs_on_the_current_stream(case: str):
    """The CUDA recurrences run asynchronously on PyTorch's current stream.

    The inputs are produced on a side stream after a long delay. The call must
    return before the stream finishes, rather than synchronize the device, and
    once it finishes, the outputs must be right, which they wouldn't be if the
    scan ran on another stream and read the inputs before they were written.
    """
    op, args, expected = _cuda_scan_case(case)
    # A kernel's first launch loads its module lazily, which synchronizes.
    op(*[arg.cuda() * 1 for arg in args])
    torch.cuda.synchronize()

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        cuda_args = [arg.cuda() for arg in args]
        # About 0.4 s of GPU time, far longer than launching the op takes.
        torch.cuda._sleep(1_000_000_000)
        cuda_args = [arg * 1 for arg in cuda_args]
        actual = op(*cuda_args)
        assert not stream.query(), "the call waited for the stream to finish"
    stream.synchronize()

    torch.testing.assert_close(actual.cpu(), expected)


@pytest.mark.cuda
@pytest.mark.parametrize(
    ("case", "on_cpu"),
    [
        ("lti_recur-shared", 0),
        ("lti_recur2-batched", 0),
        ("recur2-batched", 0),
        ("recur3-shared", 1),
        ("lti_recur3-batched", 0),
        ("lpc", 1),
        ("scan", 1),
        ("lti_recur-shared", 1),
        ("scan", 2),
    ],
)
def test_cuda_ops_reject_coefficients_on_another_device(case: str, on_cpu: int):
    """Coefficients and initial states the kernels read directly must be on the input's device."""
    op, args, _ = _cuda_scan_case(case)
    cuda_args = [arg if i == on_cpu else arg.cuda() for i, arg in enumerate(args)]
    with pytest.raises(RuntimeError, match="same device"):
        op(*cuda_args)
