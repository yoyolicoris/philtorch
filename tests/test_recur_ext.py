import pytest
import torch
from philtorch.mat import companion
from philtorch.lti import linear_recurrence
from philtorch.lpv import state_space_recursion as lpv_state_space

from .test_lti_ssm import _generate_random_filter_coeffs
from .test_lpv_filters import _generate_time_varying_coeffs


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
        (
            A.unsqueeze(1).expand(-1, samples, -1, -1)
            if batch
            else A.expand(samples, -1, -1)
        ),
        zi,
        x,
    )
    actual = torch.ops.philtorch.lti_recurN(A, zi, x)
    torch.testing.assert_close(actual, expected)


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
        pytest.param(
            "mps",
            marks=pytest.mark.skipif(
                not torch.backends.mps.is_available(), reason="MPS not available"
            ),
        ),
    ],
)
@pytest.mark.parametrize("batch", [True, False])
def test_lti_recur_equiv(device: str, batch: bool):
    B = 3
    T = 101
    dtype = torch.float32 if device == "mps" else torch.float64

    # Convert to torch tensors
    a_torch = torch.rand(B if batch else 1, device=device, dtype=dtype) * 2 - 1
    x_torch = torch.randn(B, T, device=device, dtype=dtype)
    zi = x_torch.new_zeros(B).normal_()

    lti_y = torch.ops.philtorch.lti_recur(a_torch, zi, x_torch)
    torch_y = linear_recurrence(a_torch, zi, x_torch)
    torch.testing.assert_close(lti_y, torch_y)


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        "meta",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA not available"
            ),
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
@pytest.mark.parametrize(
    "samples", [1, 2, 3, 7, 8, 15, 16, 17, 257, 511, 512, 513, 1234]
)
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
@pytest.mark.parametrize("batch_decay", [True, False])
def test_lti_recur_mps_grad_equiv(batch_decay: bool):
    batch_size = 3
    samples = 17
    a = torch.rand(batch_size if batch_decay else 1, dtype=torch.float32) * 0.75 - 0.375
    zi = torch.randn(batch_size, dtype=torch.float32)
    x = torch.randn(batch_size, samples, dtype=torch.float32)
    grad_output = torch.randn_like(x)

    def run(device: str):
        device_inputs = tuple(
            value.to(device).detach().requires_grad_() for value in (a, zi, x)
        )
        output = torch.ops.philtorch.lti_recur(*device_inputs)
        gradients = torch.autograd.grad(
            output, device_inputs, grad_outputs=grad_output.to(device)
        )
        return output.detach().cpu(), tuple(gradient.cpu() for gradient in gradients)

    expected_output, expected_gradients = run("cpu")
    actual_output, actual_gradients = run("mps")

    torch.testing.assert_close(actual_output, expected_output, rtol=1e-5, atol=1e-6)
    for actual, expected in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        # pytest.param(
        #     "cuda",
        #     marks=pytest.mark.skipif(
        #         not torch.cuda.is_available(), reason="CUDA not available"
        #     ),
        # ),
    ],
)
@pytest.mark.parametrize("order", [2, 3, 5])
def test_recurN_extension(device, order):
    """Test that the recur2 extension works correctly."""
    batch_size = 2
    N = 37

    _, a = _generate_time_varying_coeffs(batch_size, N, order, order)
    # x = _generate_test_signal(batch_size, N, "white_noise").cuda()
    x = torch.randn(batch_size, N, order).to(device).double()  # Simulated input
    A = companion(a).to(device).double()
    zi = torch.randn(batch_size, order).to(device).double()

    ext_output = torch.ops.philtorch.recurN(A, zi, x)
    torch_output = lpv_state_space(A, zi, x, unroll_factor=1)

    # Compare outputs
    assert torch.allclose(ext_output, torch_output), torch.max(
        torch.abs(ext_output - torch_output)
    )


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
def test_recur2_extension(device):
    """Test that the recur2 extension works correctly."""
    batch_size = 2
    N = 17
    order = 2

    _, a = _generate_time_varying_coeffs(batch_size, N, order, order)
    # x = _generate_test_signal(batch_size, N, "white_noise").cuda()
    x = torch.randn(batch_size, N, 2).to(device).double()  # Simulated input
    A = companion(a).to(device).double()
    zi = torch.randn(batch_size, order).to(device).double()

    ext_output = torch.ops.philtorch.recur2(A, zi, x)
    torch_output = lpv_state_space(A, zi, x, unroll_factor=1)

    # Compare outputs
    assert torch.allclose(ext_output, torch_output), torch.max(
        torch.abs(ext_output - torch_output)
    )
