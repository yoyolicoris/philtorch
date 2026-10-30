import pytest
import torch
import torch.nn.functional as F

from philtorch.lpv import state_space_recursion as lpv_state_space
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


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
@pytest.mark.parametrize("coefficient_layout", ["scalar", "shared", "batched"])
@pytest.mark.parametrize("samples", [1, 2, 511, 512, 513, 1234])
def test_lti_recur_mps_boundaries(coefficient_layout: str, samples: int):
    batch_size = 3
    coefficient_shape = {
        "scalar": (),
        "shared": (1,),
        "batched": (batch_size,),
    }[coefficient_layout]
    a = torch.rand(coefficient_shape, dtype=torch.float32) * 1.5 - 0.75
    zi = torch.randn(batch_size, dtype=torch.float32)
    x = torch.randn(batch_size, samples, dtype=torch.float32)

    expected = torch.ops.philtorch.lti_recur(a, zi, x)
    actual = torch.ops.philtorch.lti_recur(a.to("mps"), zi.to("mps"), x.to("mps")).cpu()
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_lti_recur_mps_block_chunks():
    samples = 512 * 512 + 1
    a = torch.tensor([0.25], dtype=torch.float32)
    zi = torch.randn(1, dtype=torch.float32)
    x = torch.randn(1, samples, dtype=torch.float32)

    expected = torch.ops.philtorch.lti_recur(a, zi, x)
    actual = torch.ops.philtorch.lti_recur(a.to("mps"), zi.to("mps"), x.to("mps")).cpu()
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
@pytest.mark.parametrize(
    ("x_requires_grad", "a_requires_grad", "zi_requires_grad"),
    _REQUIRES_GRAD_CASES,
)
@pytest.mark.parametrize("batch_decay", [True, False])
def test_lti_recur_mps_grad_equiv(
    x_requires_grad: bool,
    a_requires_grad: bool,
    zi_requires_grad: bool,
    batch_decay: bool,
):
    batch_size = 3
    samples = 17
    a = torch.rand(batch_size if batch_decay else 1, dtype=torch.float32) * 0.75 - 0.375
    zi = torch.randn(batch_size, dtype=torch.float32)
    x = torch.randn(batch_size, samples, dtype=torch.float32)
    grad_output = torch.randn_like(x)

    requires_grad = (a_requires_grad, zi_requires_grad, x_requires_grad)

    def run(device: str):
        device_inputs = tuple(
            value.to(device).detach().requires_grad_(input_requires_grad)
            for value, input_requires_grad in zip((a, zi, x), requires_grad)
        )
        differentiable_inputs = tuple(
            value
            for value, input_requires_grad in zip(device_inputs, requires_grad)
            if input_requires_grad
        )
        output = torch.ops.philtorch.lti_recur(*device_inputs)
        computed_gradients = iter(
            torch.autograd.grad(
                output,
                differentiable_inputs,
                grad_outputs=grad_output.to(device),
            )
        )
        gradients = tuple(
            next(computed_gradients).cpu() if input_requires_grad else None
            for input_requires_grad in requires_grad
        )
        return output.detach().cpu(), gradients

    expected_output, expected_gradients = run("cpu")
    actual_output, actual_gradients = run("mps")

    torch.testing.assert_close(actual_output, expected_output, rtol=1e-5, atol=1e-6)
    for actual, expected in zip(actual_gradients, expected_gradients):
        if actual is None or expected is None:
            assert actual is expected is None
        else:
            torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)


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
    if device == "cuda":
        # CUDA uses the ParaRNN route here rather than the direct recur2 op.
        native_output = lpv_state_space(A, zi, x)
        assert torch.allclose(native_output, torch_output), torch.max(
            torch.abs(native_output - torch_output)
        )


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

    if name.startswith("pararnn"):
        # ParaRNN solves h[t] = A[t] h[t - 1] + x[t] with jac = -A and the
        # initial state as the first right-hand side; it has no CPU kernel.
        # Past 8 blocks of steps it switches to three dependent kernels.
        K = int(name.split("-")[1])
        if name.endswith("long"):
            T = 40_000
        A, zi, x = stable(randn(B, T, K, K)), randn(B, K), randn(B, T, K)
        reduce = getattr(torch.ops.parallel_reduce_cuda, f"parallel_reduce_block_diag_{K}x{K}_cuda")
        expected = (torch.ops.philtorch.recur2 if K == 2 else torch.ops.philtorch.recurN)(A, zi, x)
        return (
            lambda jac, rhs: reduce(jac, rhs)[:, 1:],
            (F.pad(-A, (0, 0, 0, 0, 1, 0)), torch.cat([zi.unsqueeze(1), x], dim=1)),
            expected,
        )
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
        "pararnn-2",
        "pararnn-3",
        "pararnn-2-long",
        "pararnn-3-long",
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


@pytest.mark.cuda
@pytest.mark.parametrize(("B", "steps"), [(65536, 4), (1, 2**21 + 1)], ids=["batch", "length"])
def test_pararnn_op_rejects_inputs_past_its_launch_limits(B, steps):
    """Called directly, the ParaRNN op names the limit instead of failing to launch."""
    jac = torch.zeros(B, steps, 2, 2, device="cuda")
    rhs = torch.zeros(B, steps, 2, device="cuda")
    with pytest.raises(RuntimeError, match="ParaRNN supports at most"):
        torch.ops.parallel_reduce_cuda.parallel_reduce_block_diag_2x2_cuda(jac, rhs)


@pytest.mark.cuda
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two CUDA devices")
def test_pararnn_op_rejects_inputs_on_different_cuda_devices():
    jac = torch.zeros(1, 4, 2, 2, device="cuda:0")
    rhs = torch.zeros(1, 4, 2, device="cuda:1")
    with pytest.raises(RuntimeError, match="same CUDA device"):
        torch.ops.parallel_reduce_cuda.parallel_reduce_block_diag_2x2_cuda(jac, rhs)
