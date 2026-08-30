import pytest
import numpy as np
import torch
from scipy import signal
from typing import Optional
from itertools import product, chain
from unittest.mock import Mock

import philtorch.lpv.filtering as lpv_filtering
from philtorch.lpv import lfilter
from .test_lti_lfilter import (
    _generate_random_filter_coeffs,
    _generate_random_signal,
    _generate_a,
)


@pytest.mark.parametrize("B", [1, 8])
@pytest.mark.parametrize("T", [32, 128])
@pytest.mark.parametrize("num_order", [1, 2, 4])
@pytest.mark.parametrize("den_order", [1, 3, 5])
@pytest.mark.parametrize(
    ("form", "backend"),
    chain(
        zip(["df2", "tdf2", "df1", "tdf1"], ["ssm"] * 4),
        zip(["df2", "df1", "tdf1"], ["torchlpc"] * 3),
    ),
)
def test_against_lti_scipy(
    B: int, T: int, num_order: int, den_order: int, form: str, backend: str
):
    """Test time-invariant filters against scipy.signal.lfilter"""

    # Generate test data
    b, a = _generate_random_filter_coeffs(num_order, den_order, B)
    x = _generate_random_signal(B, T)

    # Convert to torch tensors
    b_torch = torch.from_numpy(b)
    a_torch = torch.from_numpy(a)
    x_torch = torch.from_numpy(x)

    # Apply philtorch filter
    y_torch = lfilter(
        b_torch.unsqueeze(1).expand(-1, T, -1),
        a_torch.unsqueeze(1).expand(-1, T, -1),
        x_torch,
        form=form,
        backend=backend,
    )

    # Apply scipy filter
    y_scipy = np.stack(
        [signal.lfilter(b[i], [1.0] + a[i].tolist(), x[i]) for i in range(B)], axis=0
    )

    # Compare outputs
    assert np.allclose(y_torch.numpy(), y_scipy), np.max(
        np.abs(y_torch.numpy() - y_scipy)
    )


@pytest.mark.parametrize(
    ("backend", "expected_form"), [("ssm", "tdf2"), ("torchlpc", "df2")]
)
def test_backend_default_form(backend: str, expected_form: str):
    batch_size, time_steps, order = 2, 17, 2
    b = torch.randn(batch_size, time_steps, order + 1).double()
    a = torch.randn(batch_size, time_steps, order).double() * 0.1
    x = torch.randn(batch_size, time_steps).double()

    actual = lfilter(b, a, x, backend=backend)
    expected = lfilter(b, a, x, form=expected_form, backend=backend)

    assert torch.allclose(actual, expected)


@pytest.mark.parametrize("form", ["df1", "tdf1"])
def test_direct_form_forwards_unroll_factor(form: str, monkeypatch):
    batch_size, time_steps, order = 2, 17, 2
    b = torch.randn(batch_size, time_steps, order + 1)
    a = torch.randn(batch_size, time_steps, order) * 0.1
    x = torch.randn(batch_size, time_steps)
    recursion = Mock(wraps=lpv_filtering.state_space_recursion)
    monkeypatch.setattr(lpv_filtering, "state_space_recursion", recursion)

    lfilter(b, a, x, form=form, unroll_factor=3)

    recursion.assert_called_once()
    assert recursion.call_args.kwargs["unroll_factor"] == 3


@pytest.mark.parametrize("num_order", [1, 3, 5])
@pytest.mark.parametrize("den_order", [1, 2, 4])
def test_tdf2_zi(num_order: int, den_order: int):
    B = 3
    T = 100
    # Generate test data
    b, a = _generate_random_filter_coeffs(num_order, den_order, B)
    x = _generate_random_signal(B, T)
    zi = np.random.randn(B, max(num_order, den_order))

    # Convert to torch tensors
    b_torch = torch.from_numpy(b)
    a_torch = torch.from_numpy(a)
    x_torch = torch.from_numpy(x)
    zi_torch = torch.from_numpy(zi)

    # Apply philtorch filter
    y_torch, zf_torch = lfilter(
        b_torch.unsqueeze(1).expand(-1, T, -1),
        a_torch.unsqueeze(1).expand(-1, T, -1),
        x_torch,
        zi=zi_torch,
        form="tdf2",
    )

    # Apply scipy filter
    y_scipy, zf_scipy = zip(
        *[signal.lfilter(b[i], [1.0] + a[i].tolist(), x[i], zi=zi[i]) for i in range(B)]
    )

    y_scipy = np.stack(y_scipy, axis=0)
    zf_scipy = np.stack(zf_scipy, axis=0)
    # Compare outputs
    assert np.allclose(y_torch.numpy(), y_scipy), np.max(
        np.abs(y_torch.numpy() - y_scipy)
    )
    assert np.allclose(zf_torch.numpy(), zf_scipy), np.max(
        np.abs(zf_torch.numpy() - zf_scipy)
    )
