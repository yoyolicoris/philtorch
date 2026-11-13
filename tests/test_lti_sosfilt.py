import numpy as np
import pytest
import torch
from scipy import signal

from philtorch.lti import sosfilt, sosfilt_zi

F64 = torch.float64


def _random_sos(*lead: int, K: int) -> np.ndarray:
    """Stable sections b0 b1 b2 a1 a2, poles inside the unit circle, of shape (*lead, K, 5)."""
    shape = (*lead, K)
    r = np.random.rand(*shape) * 0.8 + 0.1
    theta = np.random.rand(*shape) * np.pi
    a1, a2 = -2 * r * np.cos(theta), r**2
    b = np.random.randn(*shape, 3)
    return np.concatenate([b, a1[..., None], a2[..., None]], axis=-1)


def _scipy_sos(sos: np.ndarray) -> np.ndarray:
    return np.insert(sos, 3, 1.0, axis=-1)


@pytest.mark.parametrize("K", [1, 3])
# (batched_sos, batched_x): a batched sos needs a batched x
@pytest.mark.parametrize("batched_sos, batched_x", [(False, False), (False, True), (True, True)])
@pytest.mark.parametrize("use_zi", [False, True])
def test_sosfilt_matches_scipy(K, batched_sos, batched_x, use_zi):
    B, N = 3, 40
    sos = _random_sos(B, K=K) if batched_sos else _random_sos(K=K)
    x = np.random.randn(B, N) if batched_x else np.random.randn(N)
    zi = np.random.randn(B, K, 2) if batched_sos else np.random.randn(K, 2)

    def reference(i):
        s = _scipy_sos(sos[i] if batched_sos else sos)
        xi = x[i] if batched_x else x
        if not use_zi:
            return signal.sosfilt(s, xi), None
        return signal.sosfilt(s, xi, zi=zi[i] if batched_sos else zi)

    refs = [reference(i) for i in range(B if batched_x else 1)]
    y_ref = np.stack([r[0] for r in refs]) if batched_x else refs[0][0]

    out = sosfilt(
        torch.from_numpy(sos),
        torch.from_numpy(x),
        zi=torch.from_numpy(zi) if use_zi else None,
    )
    y, zf = out if use_zi else (out, None)
    torch.testing.assert_close(y, torch.from_numpy(y_ref))
    if use_zi:
        zf_ref = np.stack([r[1] for r in refs]) if batched_x else refs[0][1]
        torch.testing.assert_close(zf, torch.from_numpy(zf_ref))


@pytest.mark.parametrize("K", [1, 3])
@pytest.mark.parametrize("batched", [False, True])
def test_sosfilt_zi_matches_scipy(K, batched):
    B = 3
    sos = _random_sos(B, K=K) if batched else _random_sos(K=K)
    zi = sosfilt_zi(torch.from_numpy(sos))
    if batched:
        expected = np.stack([signal.sosfilt_zi(_scipy_sos(s)) for s in sos])
    else:
        expected = signal.sosfilt_zi(_scipy_sos(sos))
    torch.testing.assert_close(zi, torch.from_numpy(expected))


@pytest.mark.parametrize("batched_sos", [False, True])
def test_sosfilt_in_chunks_equals_whole(batched_sos):
    B, K = 3, 3
    sos = torch.from_numpy(_random_sos(B, K=K) if batched_sos else _random_sos(K=K))
    x = torch.randn(B, 60, dtype=F64)
    zi = torch.randn(K, 2, dtype=F64)
    whole, zf_whole = sosfilt(sos, x, zi=zi)
    ys = []
    zf = zi
    for chunk in x.split([20, 25, 15], dim=-1):
        y, zf = sosfilt(sos, chunk, zi=zf)
        ys.append(y)
    torch.testing.assert_close(torch.cat(ys, dim=-1), whole)
    torch.testing.assert_close(zf, zf_whole)


def test_sosfilt_gradients():
    B, K, N = 2, 2, 6
    sos = torch.from_numpy(_random_sos(B, K=K)).requires_grad_()
    x = torch.randn(B, N, dtype=F64, requires_grad=True)
    zi = torch.randn(B, K, 2, dtype=F64, requires_grad=True)
    assert torch.autograd.gradcheck(sosfilt, (sos, x, zi))
    # a shared sos and zi, and no zi
    sos1 = sos[0].detach().clone().requires_grad_()
    zi1 = zi[0].detach().clone().requires_grad_()
    assert torch.autograd.gradcheck(sosfilt, (sos1, x, zi1))
    assert torch.autograd.gradcheck(sosfilt, (sos, x))


def test_sosfilt_compiled_matches_eager():
    sos = torch.from_numpy(_random_sos(2, K=3))
    x = torch.randn(2, 30, dtype=F64)
    zi = torch.randn(2, 3, 2, dtype=F64)
    compiled = torch.compile(sosfilt, backend="aot_eager", fullgraph=True)
    y, zf = compiled(sos, x, zi)
    y_ref, zf_ref = sosfilt(sos, x, zi)
    torch.testing.assert_close(y, y_ref)
    torch.testing.assert_close(zf, zf_ref)
