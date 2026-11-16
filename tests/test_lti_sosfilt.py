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


def _flat(rows: list[np.ndarray], interleave: bool = False):
    """The sections of rows of any lengths (none included), listed grouped by
    row or round-robin, each row's in order: (sos (S, 5), batch_index (S,))."""
    listing = [(i, j) for i, r in enumerate(rows) for j in range(len(r))]
    if interleave:
        listing.sort(key=lambda il: il[1])
    sos = np.stack([rows[i][j] for i, j in listing])
    return torch.from_numpy(sos), torch.tensor([i for i, _ in listing])


def _ragged(counts: list[int], interleave: bool = False):
    """Random rows with the given numbers of sections, flat: (sos, batch_index, rows)."""
    rows = [_random_sos(K=c) if c else np.zeros((0, 5)) for c in counts]
    return (*_flat(rows, interleave), rows)


def _scipy_rows(rows, x, zi=None):
    """SciPy's sosfilt on each row alone; a row without sections passes x on."""
    ys, zfs = [], []
    for i, r in enumerate(rows):
        if not len(r):
            ys.append(x[i])
        elif zi is None:
            ys.append(signal.sosfilt(_scipy_sos(r), x[i]))
        else:
            y, zf = signal.sosfilt(_scipy_sos(r), x[i], zi=zi[i])
            ys.append(y)
            zfs.append(zf)
    return np.stack(ys), zfs


@pytest.mark.parametrize("K", [1, 3])
@pytest.mark.parametrize("batched_x", [False, True])
@pytest.mark.parametrize("use_zi", [False, True])
def test_sosfilt_matches_scipy(K, batched_x, use_zi):
    B, N = 3, 40
    sos = _random_sos(K=K)
    x = np.random.randn(B, N) if batched_x else np.random.randn(N)
    zi = np.random.randn(B, K, 2) if batched_x else np.random.randn(K, 2)
    xs, zis = (x, zi) if batched_x else (x[None], zi[None])
    refs = [signal.sosfilt(_scipy_sos(sos), xi, zi=z) for xi, z in zip(xs, zis)]
    if not use_zi:
        refs = [(signal.sosfilt(_scipy_sos(sos), xi), None) for xi in xs]
    y_ref = np.stack([r[0] for r in refs])
    y_ref = y_ref if batched_x else y_ref[0]

    out = sosfilt(
        torch.from_numpy(sos),
        torch.from_numpy(x),
        zi=torch.from_numpy(zi) if use_zi else None,
    )
    y, zf = out if use_zi else (out, None)
    torch.testing.assert_close(y, torch.from_numpy(y_ref))
    if use_zi:
        zf_ref = np.stack([r[1] for r in refs])
        torch.testing.assert_close(zf, torch.from_numpy(zf_ref if batched_x else zf_ref[0]))


@pytest.mark.parametrize("use_zi", [False, True])
def test_sosfilt_batch_index_matches_scipy_per_row(use_zi):
    counts, N = [3, 0, 1, 2, 3, 1], 40
    sos, batch_index, rows = _ragged(counts)
    x = np.random.randn(len(counts), N)
    zi = [np.random.randn(c, 2) for c in counts]
    out = sosfilt(
        sos,
        torch.from_numpy(x),
        zi=torch.from_numpy(np.concatenate(zi)) if use_zi else None,
        batch_index=batch_index,
    )
    y, zf = out if use_zi else (out, None)
    y_ref, zf_ref = _scipy_rows(rows, x, zi if use_zi else None)
    torch.testing.assert_close(y, torch.from_numpy(y_ref))
    if use_zi:
        torch.testing.assert_close(zf, torch.from_numpy(np.concatenate(zf_ref)))


def test_sosfilt_batch_index_listing_order():
    counts = [3, 0, 1, 2, 3, 1]
    sos, index, _ = _ragged(counts)
    x = torch.randn(len(counts), 30, dtype=F64)
    zi = torch.randn(len(index), 2, dtype=F64)
    # round-robin over the rows, each row's sections still in their order
    perm = torch.cat([torch.arange(c) for c in counts]).argsort(stable=True)
    y, zf = sosfilt(sos, x, zi=zi, batch_index=index)
    y_i, zf_i = sosfilt(sos[perm], x, zi=zi[perm], batch_index=index[perm])
    torch.testing.assert_close(y_i, y)
    torch.testing.assert_close(zf_i, zf[perm])


def test_sosfilt_batch_index_one_signal():
    sos, _, _ = _ragged([3])
    x = torch.randn(40, dtype=F64)
    zi = torch.zeros(3, 2, dtype=F64)
    y, zf = sosfilt(sos, x, zi=zi, batch_index=torch.zeros(3, dtype=torch.long))
    torch.testing.assert_close(y, sosfilt(sos, x))
    assert zf.shape == (3, 2)


@pytest.mark.parametrize("batch_index", [False, True])
def test_sosfilt_in_chunks_equals_whole(batch_index):
    B, K = 3, 3
    sos = torch.from_numpy(_random_sos(K=K))
    index = torch.tensor([0, 0, 2]) if batch_index else None
    x = torch.randn(B, 60, dtype=F64)
    zi = torch.randn(3, 2, dtype=F64) if batch_index else torch.randn(B, K, 2, dtype=F64)
    whole, zf_whole = sosfilt(sos, x, zi=zi, batch_index=index)
    ys = []
    zf = zi
    for chunk in x.split([20, 25, 15], dim=-1):
        y, zf = sosfilt(sos, chunk, zi=zf, batch_index=index)
        ys.append(y)
    torch.testing.assert_close(torch.cat(ys, dim=-1), whole)
    torch.testing.assert_close(zf, zf_whole)


@pytest.mark.parametrize("K", [1, 3])
def test_sosfilt_zi_matches_scipy(K):
    sos = _random_sos(K=K)
    zi = sosfilt_zi(torch.from_numpy(sos))
    torch.testing.assert_close(zi, torch.from_numpy(signal.sosfilt_zi(_scipy_sos(sos))))


def test_sosfilt_zi_batch_index_matches_each_row():
    counts = [3, 0, 1, 2]
    sos, batch_index, rows = _ragged(counts, interleave=True)
    zi = sosfilt_zi(sos, batch_index)
    assert zi.shape == (sos.size(0), 2)
    expected = np.zeros((sos.size(0), 2))
    for row, r in enumerate(rows):
        if len(r):
            expected[(batch_index == row).numpy()] = signal.sosfilt_zi(_scipy_sos(r))
    torch.testing.assert_close(zi, torch.from_numpy(expected))
    # a steady step response: the output is the DC gain of the row's cascade
    x = torch.ones(len(counts), 20, dtype=F64)
    y, _ = sosfilt(sos, x, zi=zi, batch_index=batch_index)
    for row, r in enumerate(rows):
        gain = np.prod(r[:, :3].sum(-1) / (1 + r[:, 3:].sum(-1)))
        torch.testing.assert_close(y[row], torch.full((20,), float(gain), dtype=F64))


def test_sosfilt_batch_index_gradients():
    counts, N = [2, 0, 3, 1], 12
    sos, batch_index, rows = _ragged(counts, interleave=True)
    sos.requires_grad_()
    x = torch.randn(len(counts), N, dtype=F64, requires_grad=True)
    zi = torch.randn(sos.size(0), 2, dtype=F64, requires_grad=True)
    w = torch.randn(len(counts), N, dtype=F64)
    y, zf = sosfilt(sos, x, zi=zi, batch_index=batch_index)
    grads = torch.autograd.grad((y * w).sum() + zf.sum(), (sos, x, zi))
    # each row alone, through the shared form
    sos_ref, x_ref, zi_ref = (t.detach().clone().requires_grad_() for t in (sos, x, zi))
    loss = 0
    for row in range(len(counts)):
        sel = (batch_index == row).nonzero().squeeze(-1)
        if not len(sel):
            loss = loss + (x_ref[row] * w[row]).sum()
            continue
        y_r, zf_r = sosfilt(sos_ref[sel], x_ref[row], zi=zi_ref[sel])
        loss = loss + (y_r * w[row]).sum() + zf_r.sum()
    refs = torch.autograd.grad(loss, (sos_ref, x_ref, zi_ref))
    for g, ref in zip(grads, refs):
        torch.testing.assert_close(g, ref)


def test_sosfilt_compiled_matches_eager():
    sos = torch.from_numpy(_random_sos(K=3))
    x = torch.randn(2, 30, dtype=F64)
    zi = torch.randn(2, 3, 2, dtype=F64)
    compiled = torch.compile(sosfilt, backend="aot_eager", fullgraph=True)
    y, zf = compiled(sos, x, zi)
    y_ref, zf_ref = sosfilt(sos, x, zi)
    torch.testing.assert_close(y, y_ref)
    torch.testing.assert_close(zf, zf_ref)
