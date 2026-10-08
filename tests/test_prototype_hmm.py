import itertools

import pytest
import torch

from philtorch.prototype.hmm import hmm_forward, hmm_posteriors, viterbi

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
# The sequential recursions run anywhere; the parallel scans run Helion kernels on CUDA.
METHOD_DEVICES = [
    ("sequential", "cpu"),
    pytest.param("sequential", "cuda", marks=requires_cuda),
    pytest.param("parallel", "cuda", marks=requires_cuda),
]


def _model(batch_size, N, K, *, time_varying=True, seed=0, dtype=torch.float64):
    gen = torch.Generator().manual_seed(seed)
    shape = (batch_size, N, K, K) if time_varying else (K, K)
    log_trans = torch.randn(*shape, dtype=dtype, generator=gen).log_softmax(-1)
    log_emit = torch.randn(batch_size, N, K, dtype=dtype, generator=gen) * 2
    log_init = torch.randn(batch_size, K, dtype=dtype, generator=gen).log_softmax(-1)
    return log_emit, log_trans, log_init


def _brute_force(log_emit, log_trans, log_init):
    """Enumerate every state sequence z[0], ..., z[N]."""
    batch_size, N, K = log_emit.shape
    log_trans = log_trans.expand(batch_size, N, K, K)
    paths = torch.tensor(list(itertools.product(range(K), repeat=N + 1)), device=log_emit.device)
    n = torch.arange(N)
    joint = (
        log_init[:, paths[:, 0]]
        + log_trans[:, n, paths[:, :-1], paths[:, 1:]].sum(-1)
        + log_emit[:, n, paths[:, 1:]].sum(-1)
    )  # (B, K^(N+1))
    log_likelihood = joint.logsumexp(-1)
    one_hot = torch.nn.functional.one_hot(paths[:, 1:], K).to(joint.dtype)  # (P, N, K)
    posteriors = torch.einsum("bp,pnk->bnk", (joint - log_likelihood[:, None]).exp(), one_hot)
    best = joint.argmax(-1)
    return log_likelihood, posteriors.log(), joint.amax(-1), paths[best, 1:]


@pytest.mark.parametrize("method, device", METHOD_DEVICES)
@pytest.mark.parametrize("time_varying", [True, False])
@pytest.mark.parametrize("N", [1, 2, 5])
def test_hmm_matches_brute_force(method, device, time_varying, N):
    args = [t.to(device) for t in _model(2, N, 3, time_varying=time_varying)]
    log_likelihood, log_post, best_score, best_path = _brute_force(*args)

    ll, log_filtered = hmm_forward(*args, method=method)
    torch.testing.assert_close(ll, log_likelihood)
    # The last filtered distribution is also the last posterior.
    torch.testing.assert_close(log_filtered[:, -1], log_post[:, -1])

    ll, posteriors = hmm_posteriors(*args, method=method)
    torch.testing.assert_close(ll, log_likelihood)
    torch.testing.assert_close(posteriors, log_post)

    score, path = viterbi(*args, method=method)
    torch.testing.assert_close(score, best_score)
    torch.testing.assert_close(path, best_path)


@pytest.mark.parametrize("method, device", METHOD_DEVICES)
def test_hmm_posteriors_are_the_gradient_of_the_log_likelihood(method, device):
    log_emit, log_trans, log_init = (t.to(device) for t in _model(2, 7, 4))
    log_emit.requires_grad_()
    ll, _ = hmm_forward(log_emit, log_trans, log_init, method=method)
    (grad,) = torch.autograd.grad(ll.sum(), log_emit)
    _, log_post = hmm_posteriors(log_emit, log_trans, log_init, method=method)
    torch.testing.assert_close(grad, log_post.exp())


@pytest.mark.parametrize("method, device", METHOD_DEVICES)
def test_hmm_second_derivatives(method, device):
    log_emit, log_trans, log_init = (t.to(device) for t in _model(2, 4, 3))

    def log_likelihood(log_emit, log_trans):
        return hmm_forward(log_emit, log_trans, log_init, method=method)[0]

    def log_posteriors(log_emit, log_trans):
        return hmm_posteriors(log_emit, log_trans, log_init, method=method)[1]

    inputs = (log_emit.requires_grad_(), log_trans.requires_grad_())
    for fn in (log_likelihood, log_posteriors):
        assert torch.autograd.gradgradcheck(fn, inputs)


@requires_cuda
def test_hmm_parallel_needs_cuda():
    with pytest.raises(ValueError, match="CUDA"):
        hmm_forward(*_model(1, 3, 2), method="parallel")


@requires_cuda
def test_hmm_parallel_matches_sequential_in_float32():
    args = [t.cuda() for t in _model(3, 1000, 8, dtype=torch.float32)]
    # Over 1000 steps the log-messages reach thousands, so compare the
    # log-likelihood relatively and the distributions as probabilities.
    for fn in (hmm_forward, hmm_posteriors):
        expected_ll, expected = fn(*args, method="sequential")
        ll, actual = fn(*args, method="parallel")
        torch.testing.assert_close(ll, expected_ll, rtol=1e-5, atol=0)
        torch.testing.assert_close(actual.exp(), expected.exp(), rtol=0, atol=2e-4)
    score, path = viterbi(*args, method="parallel")
    expected_score, expected_path = viterbi(*args, method="sequential")
    torch.testing.assert_close(score, expected_score, rtol=1e-5, atol=1e-3)
    assert (path == expected_path).float().mean() > 0.999
