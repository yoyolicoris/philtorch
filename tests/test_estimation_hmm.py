import itertools

import pytest
import torch

from philtorch.estimation import hmm_filter, hmm_smoother, hmm_viterbi

# The HMM scans run Triton kernels, so they need CUDA.
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


def _model(batch_size, N, K, *, time_varying=True, seed=0, dtype=torch.float64, device="cuda"):
    gen = torch.Generator().manual_seed(seed)
    shape = (batch_size, N, K, K) if time_varying else (K, K)
    log_trans = torch.randn(*shape, dtype=dtype, generator=gen).log_softmax(-1)
    log_emit = torch.randn(batch_size, N, K, dtype=dtype, generator=gen) * 2
    log_init = torch.randn(batch_size, K, dtype=dtype, generator=gen).log_softmax(-1)
    return log_emit.to(device), log_trans.to(device), log_init.to(device)


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


def _sequential(log_emit, log_trans, log_init):
    """The classical forward, forward-backward and Viterbi recursions."""
    batch_size, N, K = log_emit.shape
    log_trans = log_trans.expand(batch_size, N, K, K)

    def step(n):
        return log_trans[:, n] + log_emit[:, n].unsqueeze(-2)

    alpha, alphas = log_init, []
    for n in range(N):
        alpha = torch.logsumexp(alpha.unsqueeze(-1) + step(n), dim=-2)
        alphas.append(alpha)
    alpha = torch.stack(alphas, dim=1)
    beta, betas = log_emit.new_zeros(batch_size, K), [log_emit.new_zeros(batch_size, K)]
    for n in range(N - 1, 0, -1):
        beta = torch.logsumexp(step(n) + beta.unsqueeze(-2), dim=-1)
        betas.append(beta)
    beta = torch.stack(betas[::-1], dim=1)
    log_likelihood = alpha[:, -1].logsumexp(-1)
    filtered = alpha - alpha.logsumexp(-1, keepdim=True)
    joint = alpha + beta
    posteriors = joint - joint.logsumexp(-1, keepdim=True)

    delta, pointers = log_init, []
    for n in range(N):
        delta, best = (delta.unsqueeze(-1) + step(n)).max(dim=-2)
        pointers.append(best)
    score, state = delta.max(dim=-1)
    path = [state]
    for n in range(N - 1, 0, -1):
        state = pointers[n].gather(-1, state.unsqueeze(-1)).squeeze(-1)
        path.append(state)
    return log_likelihood, filtered, posteriors, score, torch.stack(path[::-1], dim=1)


@pytest.mark.parametrize("time_varying", [True, False])
@pytest.mark.parametrize("N", [1, 2, 5])
def test_sequential_reference_matches_brute_force(time_varying, N):
    args = _model(2, N, 3, time_varying=time_varying, device="cpu")
    log_likelihood, log_post, best_score, best_path = _brute_force(*args)
    ll, filtered, posteriors, score, path = _sequential(*args)
    torch.testing.assert_close(ll, log_likelihood)
    torch.testing.assert_close(filtered[:, -1], log_post[:, -1])
    torch.testing.assert_close(posteriors, log_post)
    torch.testing.assert_close(score, best_score)
    torch.testing.assert_close(path, best_path)


@pytest.mark.parametrize("implementation", ["chunked", "differentiable"])
@pytest.mark.parametrize("time_varying", [True, False])
@pytest.mark.parametrize("N", [1, 2, 5])
def test_hmm_matches_brute_force(implementation, time_varying, N):
    args = _model(2, N, 3, time_varying=time_varying)
    if implementation == "differentiable":
        # Inputs that need gradients take the scan of differentiable products.
        args = tuple(t.clone().requires_grad_() for t in args)
    log_likelihood, log_post, best_score, best_path = _brute_force(*args)

    ll, log_filtered = hmm_filter(*args)
    torch.testing.assert_close(ll, log_likelihood)
    # The last filtered distribution is also the last posterior.
    torch.testing.assert_close(log_filtered[:, -1], log_post[:, -1])

    ll, posteriors = hmm_smoother(*args)
    torch.testing.assert_close(ll, log_likelihood)
    torch.testing.assert_close(posteriors, log_post)

    score, path = hmm_viterbi(*args)
    torch.testing.assert_close(score, best_score)
    torch.testing.assert_close(path, best_path)


def test_hmm_posteriors_are_the_gradient_of_the_log_likelihood():
    log_emit, log_trans, log_init = _model(2, 7, 4)
    log_emit.requires_grad_()
    ll, _ = hmm_filter(log_emit, log_trans, log_init)
    (grad,) = torch.autograd.grad(ll.sum(), log_emit)
    _, log_post = hmm_smoother(log_emit, log_trans, log_init)
    torch.testing.assert_close(grad, log_post.exp())


def test_hmm_second_derivatives():
    log_emit, log_trans, log_init = _model(2, 4, 3)

    def log_likelihood(log_emit, log_trans):
        return hmm_filter(log_emit, log_trans, log_init)[0]

    def log_posteriors(log_emit, log_trans):
        return hmm_smoother(log_emit, log_trans, log_init)[1]

    inputs = (log_emit.requires_grad_(), log_trans.requires_grad_())
    for fn in (log_likelihood, log_posteriors):
        assert torch.autograd.gradgradcheck(fn, inputs)


def test_hmm_needs_cuda():
    with pytest.raises(ValueError, match="CUDA"):
        hmm_filter(*_model(1, 3, 2, device="cpu"))


def test_hmm_matches_sequential_in_float32():
    args = _model(3, 1000, 8, dtype=torch.float32)
    ll, filtered, posteriors, score, path = _sequential(*args)
    # Over 1000 steps the log-messages reach thousands, so compare the
    # log-likelihood relatively and the distributions as probabilities.
    for fn, expected in ((hmm_filter, filtered), (hmm_smoother, posteriors)):
        actual_ll, actual = fn(*args)
        torch.testing.assert_close(actual_ll, ll, rtol=1e-5, atol=0)
        torch.testing.assert_close(actual.exp(), expected.exp(), rtol=0, atol=2e-4)
    actual_score, actual_path = hmm_viterbi(*args)
    torch.testing.assert_close(actual_score, score, rtol=1e-5, atol=1e-3)
    assert (actual_path == path).float().mean() > 0.999


@pytest.mark.parametrize("product", ["log", "max"])
def test_semiring_products_under_vmap(product):
    from philtorch.estimation._semiring import log_bmm, max_bmm

    op = log_bmm if product == "log" else max_bmm

    def reference(a, b):
        terms = a.unsqueeze(-1) + b.unsqueeze(-3)
        return terms.logsumexp(-2) if product == "log" else terms.amax(-2)

    gen = torch.Generator().manual_seed(0)
    a = torch.randn(4, 3, 5, 5, dtype=torch.float64, generator=gen).cuda()
    b = torch.randn(4, 3, 5, 5, dtype=torch.float64, generator=gen).cuda()
    torch.testing.assert_close(torch.vmap(op)(a, b), reference(a, b))
    # An unbatched input is shared by every vmapped call.
    torch.testing.assert_close(torch.vmap(op, in_dims=(0, None))(a, b[0]), reference(a, b[0]))
    # First and second derivatives through vmap.
    inputs = (
        a[:2, :2, :3, :3].clone().requires_grad_(),
        b[:2, :2, :3, :3].clone().requires_grad_(),
    )
    assert torch.autograd.gradgradcheck(torch.vmap(op), inputs)


def test_hmm_gradients_with_unreachable_states():
    # A left-to-right model that starts in state 0 and stays or advances:
    # most entries of the prior and the transitions are -inf.
    B, N, K = 2, 5, 3
    log_emit, _, _ = _model(B, N, K)
    stay_or_advance = torch.eye(K, dtype=torch.bool) | torch.eye(K, dtype=torch.bool).roll(1, 1)
    log_trans = torch.where(torch.triu(stay_or_advance), 0.0, float("-inf")).double().cuda()
    log_trans = log_trans - log_trans.logsumexp(-1, keepdim=True)
    log_init = torch.tensor([0.0, float("-inf"), float("-inf")], dtype=torch.float64).cuda()
    log_emit.requires_grad_()
    ll, _ = hmm_filter(log_emit, log_trans, log_init)
    (grad,) = torch.autograd.grad(ll.sum(), log_emit)
    expected_ll, log_post, _, _ = _brute_force(log_emit.detach(), log_trans, log_init.expand(B, K))
    torch.testing.assert_close(ll, expected_ll)
    torch.testing.assert_close(grad, log_post.exp())
    _, posteriors = hmm_smoother(log_emit, log_trans, log_init)
    (grad_post,) = torch.autograd.grad(posteriors.exp().sum(), log_emit)
    assert torch.isfinite(grad_post).all()


@pytest.mark.parametrize("K", [1, 2, 5, 16, 32, 40])
@pytest.mark.parametrize("N", [1, 63, 64, 65, 300, 5000])
@pytest.mark.parametrize("time_varying", [True, False])
def test_hmm_chunked_matches_differentiable(K, N, time_varying):
    """The chunked kernels, past one and two levels of chunks, against the matrix scan."""
    args = _model(2, N, K, time_varying=time_varying)
    grad_args = tuple(t.clone().requires_grad_() for t in args)
    for fn in (hmm_filter, hmm_smoother):
        ll, probs = fn(*args)
        expected_ll, expected = fn(*grad_args)
        torch.testing.assert_close(ll, expected_ll.detach(), rtol=1e-9, atol=1e-9)
        torch.testing.assert_close(probs, expected.detach(), rtol=1e-8, atol=1e-8)
    score, path = hmm_viterbi(*args)
    expected_score, expected_path = hmm_viterbi(*grad_args)
    torch.testing.assert_close(score, expected_score.detach(), rtol=1e-9, atol=1e-9)
    assert (path == expected_path).float().mean() > 0.999


def test_hmm_empty_sequence():
    log_emit, log_trans, log_init = _model(2, 1, 3)
    log_emit = log_emit[:, :0]
    ll, filtered = hmm_filter(log_emit, log_trans[0, 0], log_init)
    assert filtered.shape == (2, 0, 3) and ll.eq(0).all()
    score, path = hmm_viterbi(log_emit, log_trans[0, 0], log_init)
    assert path.shape == (2, 0)


def test_hmm_viterbi_gradient_splits_ties():
    # With uniform probabilities every path ties: the score's gradient is
    # split among them, so each step's sums to 1, as torch.amax's would.
    B, N, K = 1, 3, 2
    log_emit = torch.zeros(B, N, K, dtype=torch.float64, device="cuda", requires_grad=True)
    log_trans = torch.full((K, K), 0.5, dtype=torch.float64, device="cuda").log()
    log_init = torch.full((K,), 0.5, dtype=torch.float64, device="cuda").log()
    score, _ = hmm_viterbi(log_emit, log_trans, log_init)
    (grad,) = torch.autograd.grad(score.sum(), log_emit)
    torch.testing.assert_close(grad.sum(-1), torch.ones(B, N, dtype=torch.float64, device="cuda"))
