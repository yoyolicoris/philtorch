"""Bad arguments to the public functions raise ValueErrors with clear messages.

Each case makes one call with exactly one bad argument, every other argument
valid, so that the check under test is the one that fires.
"""

import re

import pytest
import torch

from philtorch import lpv, lti
from philtorch.estimation import kalman_filter
from philtorch.lpv.utils import diag_shift
from philtorch.mat import companion, find_eigenvectors, matrices_cumdot, matrix_power_accumulate
from philtorch.poly import polydiv

F64 = torch.float64


def _zeros(*shape):
    return torch.zeros(*shape, dtype=F64)


def _ones(*shape):
    return torch.ones(*shape, dtype=F64)


def _eye(n):
    return torch.eye(n, dtype=F64)


def _kalman_args(**overrides):
    """A valid model with 2 signals, 3 steps, 1 measurement and 2 states."""
    args = dict(
        y=_zeros(2, 3, 1),
        A=_eye(2),
        C=_ones(1, 2),
        Q=_eye(2),
        R=_eye(1),
        m0=_zeros(2),
        P0=_eye(2),
    )
    args.update(overrides)
    return args


@pytest.mark.parametrize(
    "call, match",
    [
        pytest.param(
            lambda: kalman_filter(**_kalman_args(y=_zeros(2, 3))),
            "Measurements y must be 3D",
            id="y-not-3d",
        ),
        pytest.param(
            lambda: kalman_filter(**_kalman_args(A=_zeros(2, 3))),
            "A must be square",
            id="A-not-square",
        ),
        pytest.param(
            lambda: kalman_filter(**_kalman_args(A=_zeros(2))),
            "A must be square",
            id="A-1d",
        ),
    ],
)
def test_kalman_rejects_bad_arguments(call, match):
    with pytest.raises(ValueError, match=match):
        call()


@pytest.mark.parametrize(
    "call, match",
    [
        # fir: b (B, T, M + 1), x (B, T), zi (B, M).
        pytest.param(
            lambda: lpv.fir(_ones(4, 2), _ones(1, 4)),
            "b must be 3D",
            id="fir-b-not-3d",
        ),
        pytest.param(
            lambda: lpv.fir(_ones(1, 4, 2), _ones(4)),
            "x must be 2D",
            id="fir-x-not-2d",
        ),
        pytest.param(
            lambda: lpv.fir(_ones(1, 5, 2), _ones(1, 4)),
            "first two dimensions of b",
            id="fir-b-time-mismatch",
        ),
        pytest.param(
            lambda: lpv.fir(_ones(1, 4, 2), _ones(1, 4), zi=_zeros(1)),
            "zi must be 2D",
            id="fir-zi-not-2d",
        ),
        pytest.param(
            lambda: lpv.fir(_ones(1, 4, 2), _ones(1, 4), zi=_zeros(2, 1)),
            "first dimension of zi",
            id="fir-zi-batch-mismatch",
        ),
        pytest.param(
            lambda: lpv.fir(_ones(1, 4, 2), _ones(1, 4), zi=_zeros(1, 2)),
            "second dimension of zi",
            id="fir-zi-order-mismatch",
        ),
        # allpole: a (B, T, M), x (B, T), zi (B, M).
        pytest.param(
            lambda: lpv.allpole(_zeros(4, 1), _ones(1, 4)),
            "a must be 3D",
            id="allpole-a-not-3d",
        ),
        pytest.param(
            lambda: lpv.allpole(_zeros(1, 4, 1), _ones(4)),
            "x must be 2D",
            id="allpole-x-not-2d",
        ),
        pytest.param(
            lambda: lpv.allpole(_zeros(1, 5, 1), _ones(1, 4)),
            "first two dimensions of a",
            id="allpole-a-time-mismatch",
        ),
        pytest.param(
            lambda: lpv.allpole(_zeros(1, 4, 1), _ones(1, 4), zi=_zeros(1)),
            "zi must be 2D",
            id="allpole-zi-not-2d",
        ),
        pytest.param(
            lambda: lpv.allpole(_zeros(1, 4, 1), _ones(1, 4), zi=_zeros(2, 1)),
            "first dimension of zi",
            id="allpole-zi-batch-mismatch",
        ),
        pytest.param(
            lambda: lpv.allpole(_zeros(1, 4, 1), _ones(1, 4), zi=_zeros(1, 2)),
            "second dimension of zi",
            id="allpole-zi-order-mismatch",
        ),
        # lfilter: b (T, Mb + 1) or (B, T, Mb + 1), a (T, Ma) or (B, T, Ma).
        pytest.param(
            lambda: lpv.lfilter(_ones(4, 2), _zeros(4, 1), _ones(1, 1, 4)),
            "x must be 1D or 2D",
            id="lfilter-x-3d",
        ),
        pytest.param(
            lambda: lpv.lfilter(_ones(2), _zeros(4, 1), _ones(4)),
            "b must be 2D or 3D",
            id="lfilter-b-1d",
        ),
        pytest.param(
            lambda: lpv.lfilter(_ones(4, 2), _zeros(1), _ones(4)),
            "a must be 2D or 3D",
            id="lfilter-a-1d",
        ),
        pytest.param(
            lambda: lpv.lfilter(_ones(5, 2), _zeros(4, 1), _ones(4), backend="torchlpc"),
            "number of time steps in b and a",
            id="torchlpc-b-time-mismatch",
        ),
        # A batch of one isn't broadcast to a batched x: torchlpc did.
        pytest.param(
            lambda: lpv.lfilter(_ones(1, 4, 2), _zeros(4, 1), _ones(3, 4), backend="torchlpc"),
            "Batch size of b must match batch size of x",
            id="lfilter-b-batch-of-one",
        ),
        pytest.param(
            lambda: lpv.lfilter(_ones(4, 2), _zeros(1, 4, 1), _ones(3, 4), backend="torchlpc"),
            "Batch size of a must match batch size of x",
            id="lfilter-a-batch-of-one",
        ),
        pytest.param(
            lambda: lpv.lfilter(
                _ones(4, 2), _zeros(4, 1), _ones(4), zi=_zeros(2), backend="torchlpc"
            ),
            "zi must match filter order",
            id="torchlpc-1d-zi-order-mismatch",
        ),
        pytest.param(
            lambda: lpv.lfilter(
                _ones(4, 2), _zeros(4, 1), _ones(4), zi=_zeros(1, 2), backend="torchlpc"
            ),
            "zi must match filter order",
            id="torchlpc-2d-zi-order-mismatch",
        ),
    ],
)
def test_lpv_filtering_rejects_bad_arguments(call, match):
    with pytest.raises(ValueError, match=match):
        call()


@pytest.mark.parametrize(
    "call, match",
    [
        pytest.param(
            lambda: lpv.linear_recurrence(_zeros(4), _zeros(()), _ones(1, 1, 4), unroll_factor=2),
            "Input x must be 2D",
            id="x-not-2d",
        ),
        pytest.param(
            lambda: lpv.linear_recurrence(_zeros(4), _zeros(()), _ones(4)),
            "Input x must be 2D",
            id="x-not-2d-default",
        ),
        pytest.param(
            lambda: lpv.linear_recurrence(
                _zeros(1, 1, 4), _zeros(()), _ones(1, 4), unroll_factor=2
            ),
            "State matrix a must be 1D or 2D",
            id="a-3d",
        ),
        pytest.param(
            lambda: lpv.linear_recurrence(_zeros(4), _zeros(1, 1), _ones(1, 4), unroll_factor=2),
            "Initial state init must be 1D or 0D",
            id="init-2d",
        ),
        pytest.param(
            lambda: lpv.linear_recurrence(_zeros(1, 4), _zeros(()), _ones(3, 4)),
            "Batch size of a must match batch size of x",
            id="a-batch-of-one",
        ),
    ],
)
def test_lpv_recur_rejects_bad_arguments(call, match):
    with pytest.raises(ValueError, match=match):
        call()


# lpv.state_space with 3 signals of 4 steps. For a 3-D x, B=None needs F = M,
# so x is (3, 4, 2) and A is (4, 2, 2): the shape checks on D then run after
# the recursion.
_LPV_A = _zeros(4, 2, 2)
_LPV_X2 = _ones(3, 4)
_LPV_X3 = _ones(3, 4, 2)


@pytest.mark.parametrize(
    "call, match",
    [
        pytest.param(
            lambda: lpv.state_space(_LPV_A, _ones(4)),
            "Input signal must be 2D or 3D",
            id="x-1d",
        ),
        pytest.param(
            lambda: lpv.state_space(_zeros(2, 2), _LPV_X2),
            "State matrix A must be 3D or 4D",
            id="A-2d",
        ),
        pytest.param(
            lambda: lpv.state_space(_zeros(4, 2, 3), _LPV_X2),
            "State matrix A must be square",
            id="A-not-square",
        ),
        pytest.param(
            lambda: lpv.state_space(_zeros(5, 2, 2), _LPV_X2),
            "same time dimension as x",
            id="A-time-mismatch",
        ),
        pytest.param(
            lambda: lpv.state_space(_LPV_A, _LPV_X2, C=_ones(2), out_idx=0),
            "C and out_idx cannot be used together",
            id="C-and-out_idx",
        ),
        # B of a shape that only takes a 2-D x, given a 3-D x (F = 5).
        pytest.param(
            lambda: lpv.state_space(_LPV_A, _ones(3, 4, 5), B=_ones(2)),
            "x must be 2D when B is of shape",
            id="B-M-with-3d-x",
        ),
        pytest.param(
            lambda: lpv.state_space(_LPV_A, _ones(3, 4, 5), B=_ones(4, 2)),
            "x must be 2D when B is of shape",
            id="B-NM-with-3d-x",
        ),
        pytest.param(
            lambda: lpv.state_space(_LPV_A, _ones(3, 4, 5), B=_ones(3, 2)),
            "x must be 2D when B is of shape",
            id="B-batch-M-with-3d-x",
        ),
        pytest.param(
            lambda: lpv.state_space(_LPV_A, _ones(3, 4, 5), B=_ones(3, 4, 2)),
            "x must be 2D when B is of shape",
            id="B-batch-NM-with-3d-x",
        ),
        # D of a shape that only takes a 2-D x, given a 3-D x (F = 2).
        pytest.param(
            lambda: lpv.state_space(_LPV_A, _LPV_X3, D=_ones(4)),
            re.escape("x must be 2D when D is of shape (4,)"),
            id="D-N-with-3d-x",
        ),
        pytest.param(
            lambda: lpv.state_space(_LPV_A, _LPV_X3, D=_ones(3)),
            re.escape("x must be 2D when D is of shape (3,)"),
            id="D-batch-with-3d-x",
        ),
        pytest.param(
            lambda: lpv.state_space(_LPV_A, _LPV_X3, D=_ones(7)),
            "x must be 2D when D is of shape",
            id="D-P-with-3d-x",
        ),
        pytest.param(
            lambda: lpv.state_space(_LPV_A, _LPV_X3, D=_ones(4, 7)),
            "x must be 2D when D is of shape",
            id="D-NP-with-3d-x",
        ),
        pytest.param(
            lambda: lpv.state_space(_LPV_A, _LPV_X3, D=_ones(3, 4)),
            "x must be 2D when D is of shape",
            id="D-batch-N-with-3d-x",
        ),
        pytest.param(
            lambda: lpv.state_space(_LPV_A, _LPV_X3, D=_ones(3, 7)),
            "x must be 2D when D is of shape",
            id="D-batch-P-with-3d-x",
        ),
        pytest.param(
            lambda: lpv.state_space(_LPV_A, _LPV_X3, D=_ones(3, 4, 7)),
            "x must be 2D when D is of shape",
            id="D-batch-NP-with-3d-x",
        ),
    ],
)
def test_lpv_ssm_rejects_bad_arguments(call, match):
    with pytest.raises(ValueError, match=match):
        call()


@pytest.mark.parametrize(
    "call, match",
    [
        pytest.param(
            lambda: diag_shift(_ones(4)),
            "at least 2 dimensions",
            id="coef-1d",
        ),
    ],
)
def test_lpv_utils_rejects_bad_arguments(call, match):
    with pytest.raises(ValueError, match=match):
        call()


@pytest.mark.parametrize(
    "call, match",
    [
        # comb_filter(a, delay, x)
        pytest.param(
            lambda: lti.comb_filter(_zeros(1, 1), 2, _ones(1, 4)),
            "a must be at most 1D",
            id="comb-a-2d",
        ),
        pytest.param(
            lambda: lti.comb_filter(_zeros(()), 2, _ones(4)),
            "x must be 2D",
            id="comb-x-1d",
        ),
        pytest.param(
            lambda: lti.comb_filter(_zeros(2), 2, _ones(1, 4)),
            "first dimension of a must match the batch size",
            id="comb-a-batch-mismatch",
        ),
        # lfiltic(b, a, y)
        pytest.param(
            lambda: lti.lfiltic(_ones(()), _zeros(1), _zeros(1)),
            "b must be at least 1D",
            id="lfiltic-b-0d",
        ),
        pytest.param(
            lambda: lti.lfiltic(_ones(2), _zeros(()), _zeros(1)),
            "a must be at least 1D",
            id="lfiltic-a-0d",
        ),
        # lfilter_zi(a, b)
        pytest.param(
            lambda: lti.lfilter_zi(_zeros(()), _ones(2)),
            "a must be at least 1D",
            id="lfilter_zi-a-0d",
        ),
        pytest.param(
            lambda: lti.lfilter_zi(_zeros(1), _ones(())),
            "b must be at least 1D",
            id="lfilter_zi-b-0d",
        ),
        # fir: b (B, M + 1), x (B, T), zi (B, M).
        pytest.param(
            lambda: lti.fir(_ones(2), _ones(1, 4)),
            "b must be 2D",
            id="fir-b-1d",
        ),
        pytest.param(
            lambda: lti.fir(_ones(1, 2), _ones(4)),
            "x must be 2D",
            id="fir-x-1d",
        ),
        pytest.param(
            lambda: lti.fir(_ones(2, 2), _ones(1, 4)),
            "first dimension of b must match the batch size",
            id="fir-b-batch-mismatch",
        ),
        pytest.param(
            lambda: lti.fir(_ones(1, 2), _ones(1, 4), zi=_zeros(1)),
            "zi must be 2D",
            id="fir-zi-1d",
        ),
        pytest.param(
            lambda: lti.fir(_ones(1, 2), _ones(1, 4), zi=_zeros(2, 1)),
            "first dimension of zi",
            id="fir-zi-batch-mismatch",
        ),
        pytest.param(
            lambda: lti.fir(_ones(1, 2), _ones(1, 4), zi=_zeros(1, 2)),
            "second dimension of zi",
            id="fir-zi-order-mismatch",
        ),
        # lfilter(b, a, x)
        pytest.param(
            lambda: lti.lfilter(_ones(2), _zeros(1), _ones(1, 1, 4)),
            "x must be 1D or 2D",
            id="lfilter-x-3d",
        ),
        pytest.param(
            lambda: lti.lfilter(_ones(()), _zeros(1), _ones(4)),
            "b must be 1D or 2D",
            id="lfilter-b-0d",
        ),
        pytest.param(
            lambda: lti.lfilter(_ones(2), _zeros(1, 1, 1), _ones(4)),
            "a must be 1D or 2D",
            id="lfilter-a-3d",
        ),
        # filtfilt(b, a, x)
        pytest.param(
            lambda: lti.filtfilt(_ones(2), _zeros(1), _ones(16), method="odd"),
            "Method must be either 'pad' or 'gust'",
            id="filtfilt-unknown-method",
        ),
        pytest.param(
            lambda: lti.filtfilt(_ones(2), _zeros(1), _ones(6)),
            "must be greater than pad length 6",
            id="filtfilt-x-too-short",
        ),
    ],
)
def test_lti_filtering_rejects_bad_arguments(call, match):
    with pytest.raises(ValueError, match=match):
        call()


@pytest.mark.parametrize(
    "call, match",
    [
        pytest.param(
            lambda: lti.cubic_spline(_ones(1, 4), 0),
            "integer >= 1",
            id="m-zero",
        ),
        pytest.param(
            lambda: lti.cubic_spline(_ones(1, 4), 1.5),
            "integer >= 1",
            id="m-not-integer",
        ),
    ],
)
def test_lti_interp_rejects_bad_arguments(call, match):
    with pytest.raises(ValueError, match=match):
        call()


@pytest.mark.parametrize(
    "call, match",
    [
        pytest.param(
            lambda: lti.linear_recurrence(_zeros(()), _zeros(()), _ones(1, 1, 4), unroll_factor=2),
            "Input x must be 2D",
            id="x-not-2d",
        ),
        pytest.param(
            lambda: lti.linear_recurrence(_zeros(()), _zeros(()), _ones(4)),
            "Input x must be 2D",
            id="x-not-2d-default",
        ),
        pytest.param(
            lambda: lti.linear_recurrence(_zeros(1, 1), _zeros(()), _ones(1, 4), unroll_factor=2),
            "State matrix a must be 1D or 0D",
            id="a-2d",
        ),
        pytest.param(
            lambda: lti.linear_recurrence(_zeros(()), _zeros(1, 1), _ones(1, 4), unroll_factor=2),
            "Initial state init must be 1D or 0D",
            id="init-2d",
        ),
    ],
)
def test_lti_recur_rejects_bad_arguments(call, match):
    with pytest.raises(ValueError, match=match):
        call()


@pytest.mark.parametrize(
    "call, match",
    [
        # state_space(A, x)
        pytest.param(
            lambda: lti.state_space(_eye(2), _ones(4)),
            "Input signal must be 2D or 3D",
            id="ss-x-1d",
        ),
        pytest.param(
            lambda: lti.state_space(_zeros(2), _ones(1, 4)),
            "State matrix A must be 2D or 3D",
            id="ss-A-1d",
        ),
        pytest.param(
            lambda: lti.state_space(_zeros(2, 3), _ones(1, 4)),
            "State matrix A must be square",
            id="ss-A-not-square",
        ),
        pytest.param(
            lambda: lti.state_space(_zeros(2, 2, 2), _ones(1, 4)),
            "Batch size of A must match batch size of x",
            id="ss-A-batch-mismatch",
        ),
        pytest.param(
            lambda: lti.state_space(_eye(2), _ones(1, 4), C=_ones(2), out_idx=0),
            "C and out_idx cannot be used together",
            id="ss-C-and-out_idx",
        ),
        # diag_state_space(x, L, V, Vinv, A), with A given alone.
        pytest.param(
            lambda: lti.diag_state_space(_ones(4), A=_eye(2)),
            "Input signal must be 2D or 3D",
            id="diag-x-1d",
        ),
        pytest.param(
            lambda: lti.diag_state_space(_ones(1, 4), A=_eye(2), C=_ones(2), out_idx=0),
            "C and out_idx cannot be used together",
            id="diag-C-and-out_idx",
        ),
        pytest.param(
            lambda: lti.diag_state_space(_ones(1, 4)),
            "Either L or A must be provided",
            id="diag-no-L-or-A",
        ),
        pytest.param(
            lambda: lti.diag_state_space(_ones(1, 4), A=_zeros(2)),
            "State matrix A must be 2D or 3D",
            id="diag-A-1d",
        ),
        pytest.param(
            lambda: lti.diag_state_space(_ones(1, 4), A=_zeros(2, 3)),
            "State matrix A must be square",
            id="diag-A-not-square",
        ),
        pytest.param(
            lambda: lti.diag_state_space(_ones(1, 4), A=_zeros(2, 2, 2)),
            "Batch size of A must match batch size of x",
            id="diag-A-batch-mismatch",
        ),
        # L a vector of M = 2 eigenvalues.
        pytest.param(
            lambda: lti.diag_state_space(_ones(1, 4), L=_zeros(2), V=_zeros(1, 2, 2)),
            "V must be 2D",
            id="diag-L-vector-V-3d",
        ),
        pytest.param(
            lambda: lti.diag_state_space(_ones(1, 4), L=_zeros(2), V=_eye(3)),
            "V must be square with size 2",
            id="diag-L-vector-V-wrong-size",
        ),
        pytest.param(
            lambda: lti.diag_state_space(_ones(1, 4), L=_zeros(2), Vinv=_zeros(1, 2, 2)),
            "Vinv must be 2D",
            id="diag-L-vector-Vinv-3d",
        ),
        pytest.param(
            lambda: lti.diag_state_space(_ones(1, 4), L=_zeros(2), Vinv=_eye(3)),
            "Vinv must be square with size 2",
            id="diag-L-vector-Vinv-wrong-size",
        ),
        pytest.param(
            lambda: lti.diag_state_space(_ones(1, 4), L=_zeros(2), A=_zeros(1, 2, 2)),
            "A must be 2D",
            id="diag-L-vector-A-3d",
        ),
        pytest.param(
            lambda: lti.diag_state_space(_ones(1, 4), L=_zeros(2), A=_eye(3)),
            "A must be square with size 2",
            id="diag-L-vector-A-wrong-size",
        ),
        # L a batch of 2 vectors of M = 3 eigenvalues.
        pytest.param(
            lambda: lti.diag_state_space(_ones(2, 4), L=_zeros(2, 3)),
            "V and Vinv cannot both be None",
            id="diag-L-batch-no-V",
        ),
        pytest.param(
            lambda: lti.diag_state_space(_ones(2, 4), L=_zeros(2, 3), V=_eye(3)),
            "V must be 3D",
            id="diag-L-batch-V-2d",
        ),
        pytest.param(
            lambda: lti.diag_state_space(_ones(2, 4), L=_zeros(2, 3), V=_zeros(2, 2, 2)),
            "V must be a batch of square matrices with size 3",
            id="diag-L-batch-V-wrong-size",
        ),
        pytest.param(
            lambda: lti.diag_state_space(_ones(2, 4), L=_zeros(2, 3), Vinv=_eye(3)),
            "Vinv must be 3D",
            id="diag-L-batch-Vinv-2d",
        ),
        pytest.param(
            lambda: lti.diag_state_space(_ones(2, 4), L=_zeros(2, 3), Vinv=_zeros(2, 2, 2)),
            "Vinv must be a batch of square matrices with size 3",
            id="diag-L-batch-Vinv-wrong-size",
        ),
        pytest.param(
            lambda: lti.diag_state_space(
                _ones(2, 4), L=_zeros(2, 3), V=_eye(3).expand(2, 3, 3), A=_eye(3)
            ),
            "A must be 3D",
            id="diag-L-batch-A-2d",
        ),
        pytest.param(
            lambda: lti.diag_state_space(
                _ones(2, 4), L=_zeros(2, 3), V=_eye(3).expand(2, 3, 3), A=_zeros(2, 2, 2)
            ),
            "A must be a batch of square matrices with size 3",
            id="diag-L-batch-A-wrong-size",
        ),
    ],
)
def test_lti_ssm_rejects_bad_arguments(call, match):
    with pytest.raises(ValueError, match=match):
        call()


@pytest.mark.parametrize(
    "call, match",
    [
        pytest.param(
            lambda: find_eigenvectors(_zeros(2), _zeros(2)),
            "Matrix A must be at least 2D",
            id="eigvecs-A-1d",
        ),
        pytest.param(
            lambda: find_eigenvectors(_eye(2), _zeros(())),
            "Eigenvalues must be at least 1D",
            id="eigvecs-eigenvalues-0d",
        ),
        pytest.param(
            lambda: find_eigenvectors(_zeros(2, 3), _zeros(3)),
            "Matrix A must be square",
            id="eigvecs-A-not-square",
        ),
        pytest.param(
            lambda: find_eigenvectors(_eye(2), _zeros(3)),
            "Eigenvalues must match the size of A",
            id="eigvecs-size-mismatch",
        ),
        pytest.param(
            lambda: companion(_zeros(())),
            "at least 1D",
            id="companion-a-0d",
        ),
        pytest.param(
            lambda: matrix_power_accumulate(_zeros(2), 2),
            "at least 2 dimensions",
            id="power-A-1d",
        ),
        pytest.param(
            lambda: matrix_power_accumulate(_zeros(2, 3), 2),
            "must be square",
            id="power-A-not-square",
        ),
        pytest.param(
            lambda: matrices_cumdot(_eye(2)),
            "at least 3 dimensions",
            id="cumdot-A-2d",
        ),
        pytest.param(
            lambda: matrices_cumdot(_zeros(4, 2, 3)),
            "must have square matrices",
            id="cumdot-A-not-square",
        ),
    ],
)
def test_mat_rejects_bad_arguments(call, match):
    with pytest.raises(ValueError, match=match):
        call()


@pytest.mark.parametrize(
    "call, match",
    [
        pytest.param(
            lambda: polydiv(_ones(()), _ones(2)),
            "u and v must be at least 1D",
            id="u-0d",
        ),
        pytest.param(
            lambda: polydiv(_ones(3), _ones(())),
            "u and v must be at least 1D",
            id="v-0d",
        ),
    ],
)
def test_poly_rejects_bad_arguments(call, match):
    with pytest.raises(ValueError, match=match):
        call()
