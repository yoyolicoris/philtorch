import itertools
import subprocess
import sys

import pytest
import torch

from philtorch.estimation import hmm_filter, hmm_smoother, hmm_viterbi

# The HMM scans run Triton kernels, so the tests that run them need CUDA; the
# references, validation and import tests run anywhere.
requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


def _model(batch_size, N, K, *, time_varying=True, seed=0, dtype=torch.float64, device="cuda"):
    gen = torch.Generator().manual_seed(seed)
    shape = (batch_size, max(N - 1, 0), K, K) if time_varying else (K, K)
    log_trans = torch.randn(*shape, dtype=dtype, generator=gen).log_softmax(-1)
    log_emit = torch.randn(batch_size, N, K, dtype=dtype, generator=gen) * 2
    log_init = torch.randn(batch_size, K, dtype=dtype, generator=gen).log_softmax(-1)
    return log_emit.to(device), log_trans.to(device), log_init.to(device)


def _brute_force(log_emit, log_trans, log_init):
    """Enumerate every state sequence z[0], ..., z[N - 1]."""
    batch_size, N, K = log_emit.shape
    log_trans = log_trans.expand(batch_size, N - 1, K, K)
    paths = torch.tensor(list(itertools.product(range(K), repeat=N)), device=log_emit.device)
    joint = (
        log_init[:, paths[:, 0]]
        + log_trans[:, torch.arange(N - 1), paths[:, :-1], paths[:, 1:]].sum(-1)
        + log_emit[:, torch.arange(N), paths].sum(-1)
    )  # (B, K^N)
    log_likelihood = joint.logsumexp(-1)
    one_hot = torch.nn.functional.one_hot(paths, K).to(joint.dtype)  # (P, N, K)
    posteriors = torch.einsum("bp,pnk->bnk", (joint - log_likelihood[:, None]).exp(), one_hot)
    best = joint.argmax(-1)
    return log_likelihood, posteriors.log(), joint.amax(-1), paths[best]


def _sequential(log_emit, log_trans, log_init):
    """The classical forward, forward-backward and Viterbi recursions."""
    batch_size, N, K = log_emit.shape
    log_trans = log_trans.expand(batch_size, N - 1, K, K)

    def step(n):
        """log p(z[n + 1] = j, y[n + 1] | z[n] = i) at [i, j]."""
        return log_trans[:, n] + log_emit[:, n + 1].unsqueeze(-2)

    alpha = log_init + log_emit[:, 0]
    alphas = [alpha]
    for n in range(N - 1):
        alpha = torch.logsumexp(alpha.unsqueeze(-1) + step(n), dim=-2)
        alphas.append(alpha)
    alpha = torch.stack(alphas, dim=1)
    beta, betas = log_emit.new_zeros(batch_size, K), [log_emit.new_zeros(batch_size, K)]
    for n in range(N - 2, -1, -1):
        beta = torch.logsumexp(step(n) + beta.unsqueeze(-2), dim=-1)
        betas.append(beta)
    beta = torch.stack(betas[::-1], dim=1)
    log_likelihood = alpha[:, -1].logsumexp(-1)
    filtered = alpha - alpha.logsumexp(-1, keepdim=True)
    joint = alpha + beta
    posteriors = joint - joint.logsumexp(-1, keepdim=True)

    delta, pointers = log_init + log_emit[:, 0], []
    for n in range(N - 1):
        delta, best = (delta.unsqueeze(-1) + step(n)).max(dim=-2)
        pointers.append(best)
    score, state = delta.max(dim=-1)
    path = [state]
    for n in range(N - 2, -1, -1):
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


@requires_cuda
def test_hmm_posteriors_are_the_gradient_of_the_log_likelihood():
    log_emit, log_trans, log_init = _model(2, 7, 4)
    log_emit.requires_grad_()
    ll, _ = hmm_filter(log_emit, log_trans, log_init)
    (grad,) = torch.autograd.grad(ll.sum(), log_emit)
    _, log_post = hmm_smoother(log_emit, log_trans, log_init)
    torch.testing.assert_close(grad, log_post.exp())


def _transitions(log_trans, kind):
    """The model's transitions, (B, N - 1, K, K), shared as (K, K), per step as
    (N - 1, K, K), constant per signal as (B, K, K), or as they are."""
    if kind in ("shared", "signal_constant"):
        # Drawn anew rather than taken from log_trans, which has no steps when N = 1.
        B, K = log_trans.size(0), log_trans.size(-1)
        shape = (K, K) if kind == "shared" else (B, K, K)
        gen = torch.Generator().manual_seed(1)
        drawn = torch.randn(*shape, dtype=log_trans.dtype, generator=gen).log_softmax(-1)
        return drawn.to(log_trans.device)
    return log_trans[0] if kind == "time" else log_trans


@requires_cuda
@pytest.mark.parametrize("trans", ["shared", "time", "signal_constant", "signal"])
@pytest.mark.parametrize("K", [3, 17])
def test_hmm_derivatives_to_second_order(trans, K):
    """gradcheck and gradgradcheck; K = 17 runs the log and max-plus chains in tiles.

    The numerical derivatives perturb every input entry, so K = 17 takes one
    sequence and one step fewer; B differs from N - 1 throughout, so a
    constant per signal is not read per step.
    """
    log_emit, log_trans, log_init = _model(2 if K < 10 else 1, 4 if K < 10 else 3, K)
    inputs = (log_emit, _transitions(log_trans, trans), log_init)
    inputs = tuple(t.requires_grad_() for t in inputs)

    def outputs(log_emit, log_trans, log_init):
        ll, filtered = hmm_filter(log_emit, log_trans, log_init)
        _, posteriors = hmm_smoother(log_emit, log_trans, log_init)
        score, _ = hmm_viterbi(log_emit, log_trans, log_init)
        return ll, filtered, posteriors, score

    assert torch.autograd.gradcheck(outputs, inputs)
    assert torch.autograd.gradgradcheck(outputs, inputs)


@requires_cuda
@pytest.mark.parametrize("trans", ["shared", "time", "signal_constant", "signal"])
@pytest.mark.parametrize(("N", "K"), [(1, 3), (66, 5), (300, 17), (1000, 2), (66, 33)])
def test_hmm_gradients_match_sequential(trans, N, K):
    """Gradients against autograd through the sequential recursions.

    300 and 1000 steps make the shared transitions' gradient sum over blocks;
    K = 33 tiles the derivatives' linear chains, over two chunks.
    """
    log_emit, log_trans, log_init = _model(2, N, K)
    inputs = tuple(t.requires_grad_() for t in (log_emit, _transitions(log_trans, trans), log_init))
    # A loss on every output, with random weights on the probabilities.
    w_filtered, w_posteriors = torch.randn_like(log_emit), torch.randn_like(log_emit)

    def loss(ll, filtered, posteriors, score):
        weighted = (filtered.exp() * w_filtered).sum() + (posteriors.exp() * w_posteriors).sum()
        return weighted + ll.sum() + score.sum()

    ll, filtered = hmm_filter(*inputs)
    _, posteriors = hmm_smoother(*inputs)
    score, _ = hmm_viterbi(*inputs)
    actual = torch.autograd.grad(loss(ll, filtered, posteriors, score), inputs)
    # The reference expands transitions to (B, N - 1, K, K), from (B, 1, K, K) per signal.
    reference = list(inputs)
    if trans == "signal_constant":
        reference[1] = reference[1][:, None]
    expected_ll, filtered, posteriors, score, _ = _sequential(*reference)
    expected = torch.autograd.grad(
        loss(expected_ll, filtered, posteriors, score), inputs, materialize_grads=True
    )
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a, e)


@requires_cuda
@pytest.mark.parametrize("trans", ["shared", "time"])
def test_hmm_second_order_matches_sequential(trans):
    """A Hessian-vector product against the sequential recursions, with K = 33
    tiling the linear chains of both derivative orders over two chunks."""
    log_emit, log_trans, log_init = _model(2, 66, 33)
    inputs = tuple(t.requires_grad_() for t in (log_emit, _transitions(log_trans, trans), log_init))
    w = torch.randn_like(log_emit)
    vectors = [torch.randn_like(t) for t in inputs]

    def hvp(ll, posteriors):
        grads = torch.autograd.grad(
            ll.sum() + (posteriors.exp() * w).sum(), inputs, create_graph=True
        )
        product = sum((g * v).sum() for g, v in zip(grads, vectors))
        return torch.autograd.grad(product, inputs, materialize_grads=True)

    ll, _ = hmm_filter(*inputs)
    _, posteriors = hmm_smoother(*inputs)
    actual = hvp(ll, posteriors)
    expected_ll, _, expected_posteriors, _, _ = _sequential(*inputs)
    for a, e in zip(actual, hvp(expected_ll, expected_posteriors)):
        torch.testing.assert_close(a, e)


def test_hmm_rejects_unsupported_shapes():
    log_emit, log_trans, log_init = _model(2, 4, 3, device="cpu")
    with pytest.raises(ValueError, match="log_emit"):
        hmm_filter(log_emit[0], log_trans, log_init)
    with pytest.raises(ValueError, match="log_trans"):
        hmm_filter(log_emit, log_trans[..., :2], log_init)
    # log_trans has the N - 1 transitions between the N states, not N.
    with pytest.raises(ValueError, match="log_trans"):
        hmm_filter(log_emit, torch.cat([log_trans, log_trans[:, :1]], dim=1), log_init)
    with pytest.raises(ValueError, match="log_trans"):
        hmm_filter(log_emit, log_trans[0, 0].expand(4, 3, 3), log_init)
    with pytest.raises(ValueError, match="log_init"):
        hmm_filter(log_emit, log_trans, log_init[:, :2])


def test_estimation_imports_without_triton(tmp_path):
    # The kernels are imported on first use, so the Kalman functions, and
    # the package, still import where Triton is missing. Run outside the
    # checkout, whose philtorch/ has no built extension, so the subprocess
    # imports the installed package as the tests do.
    code = "import sys; sys.modules['triton'] = None; import philtorch.estimation"
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


@requires_cuda
def test_hmm_chain_takes_an_expanded_gradient():
    # The gradient of a sum reaches the chain with every stride 0; the
    # kernels need each step's K entries contiguous.
    from philtorch.estimation.hmm import _LogChain

    log_emit, log_trans, log_init = _model(2, 70, 3, time_varying=False)
    first = (log_init + log_emit[:, 0]).requires_grad_()
    alpha = _LogChain.apply(first, log_trans[None, None], log_emit[:, 1:])
    (grad,) = torch.autograd.grad(alpha.sum(), first, retain_graph=True)
    (expected,) = torch.autograd.grad(alpha, first, torch.ones_like(alpha))
    torch.testing.assert_close(grad, expected)


@requires_cuda
def test_hmm_reads_an_ambiguous_3d_log_trans_per_step():
    # With B = N - 1, a (B, K, K) tensor is also (N - 1, K, K); like
    # kalman_filter, the HMM functions take it per step.
    log_emit, log_trans, log_init = _model(3, 4, 2)
    per_step = log_trans[0]
    torch.testing.assert_close(
        hmm_filter(log_emit, per_step, log_init),
        hmm_filter(log_emit, per_step.expand(3, 3, 2, 2), log_init),
    )


def test_hmm_needs_cuda():
    with pytest.raises(ValueError, match="hmm_filter runs Triton kernels on CUDA GPUs only"):
        hmm_filter(*_model(1, 3, 2, device="cpu"))


@requires_cuda
def test_hmm_needs_its_inputs_on_one_device():
    log_emit, log_trans, log_init = _model(1, 3, 2)
    with pytest.raises(ValueError, match="on CUDA GPUs only; got a tensor on cpu"):
        hmm_filter(log_emit, log_trans.cpu(), log_init)
    if torch.cuda.device_count() > 1:
        with pytest.raises(ValueError, match="inputs on one device"):
            hmm_filter(log_emit, log_trans.to("cuda:1"), log_init)


@requires_cuda
def test_hmm_needs_triton(tmp_path):
    # Outside the checkout, as in test_estimation_imports_without_triton.
    code = (
        "import sys; sys.modules['triton'] = None; import torch\n"
        "from philtorch.estimation import hmm_viterbi\n"
        "x = torch.zeros(1, 3, 2, device='cuda')\n"
        "hmm_viterbi(x, torch.zeros(2, 2, device='cuda'), torch.zeros(2, device='cuda'))"
    )
    run = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True)
    assert "hmm_viterbi runs Triton kernels, but Triton is not installed" in run.stderr


@requires_cuda
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


@requires_cuda
def test_hmm_gradients_with_unreachable_states():
    # A left-to-right model that starts in state 0 and stays or advances:
    # most entries of the prior and the transitions are -inf.
    B, N, K = 2, 5, 3
    log_emit, _, _ = _model(B, N, K)
    f64 = dict(dtype=torch.float64, device="cuda")
    stay = torch.eye(K, dtype=torch.bool, device="cuda")
    log_trans = torch.where(torch.triu(stay | stay.roll(1, 1)), 0.0, float("-inf")).to(**f64)
    log_trans = log_trans - log_trans.logsumexp(-1, keepdim=True)
    log_init = torch.tensor([0.0, float("-inf"), float("-inf")], **f64)
    log_emit.requires_grad_()
    ll, _ = hmm_filter(log_emit, log_trans, log_init)
    (grad,) = torch.autograd.grad(ll.sum(), log_emit)
    expected_ll, log_post, _, _ = _brute_force(log_emit.detach(), log_trans, log_init.expand(B, K))
    torch.testing.assert_close(ll, expected_ll)
    torch.testing.assert_close(grad, log_post.exp())
    _, posteriors = hmm_smoother(log_emit, log_trans, log_init)
    (grad_post,) = torch.autograd.grad(posteriors.exp().sum(), log_emit)
    assert torch.isfinite(grad_post).all()


@requires_cuda
def test_hmm_impossible_emissions():
    # Each state can't emit some observations: -inf emissions must give no
    # NaN in the messages or their gradients. The reference takes -1e4
    # instead, whose exp is 0 too, as torch's own logsumexp has NaN
    # gradients at -inf.
    emissions, log_trans, log_init = _model(2, 70, 3, time_varying=False)
    gen = torch.Generator().manual_seed(2)
    impossible = (torch.rand(emissions.shape, generator=gen) < 0.3).to(emissions.device)
    impossible[..., 0] = False  # state 0 can emit anything, so a path exists
    emissions.requires_grad_()
    log_emit = emissions.masked_fill(impossible, float("-inf"))
    finite = emissions.masked_fill(impossible, -1e4)
    expected_ll, _, expected, _, _ = _sequential(finite, log_trans, log_init)
    ll, posteriors = hmm_smoother(log_emit, log_trans, log_init)
    torch.testing.assert_close(ll, expected_ll)
    torch.testing.assert_close(posteriors.exp(), expected.exp())
    weights = torch.randn_like(emissions)
    (grad,) = torch.autograd.grad((posteriors.exp() * weights).sum(), emissions)
    (expected_grad,) = torch.autograd.grad((expected.exp() * weights).sum(), emissions)
    torch.testing.assert_close(grad, expected_grad)


@requires_cuda
@pytest.mark.parametrize("K", [1, 2, 5, 16, 17, 32, 40, 64])
@pytest.mark.parametrize(
    ("N", "time_varying"), [(1, True), (64, True), (65, False), (66, True), (300, False)]
)
def test_hmm_matches_sequential(K, N, time_varying):
    """The kernels around one chunk and past it, in registers (K <= 16) and in tiles.

    The chains run over the N - 1 transitions, so N = 64, 65 and 66 put 63,
    64 and 65 steps around the chunk size.
    """
    args = _model(2, N, K, time_varying=time_varying)
    ll, filtered, posteriors, score, path = _sequential(*args)
    for fn, expected in ((hmm_filter, filtered), (hmm_smoother, posteriors)):
        actual_ll, actual = fn(*args)
        torch.testing.assert_close(actual_ll, ll)
        torch.testing.assert_close(actual, expected)
    actual_score, actual_path = hmm_viterbi(*args)
    torch.testing.assert_close(actual_score, score)
    torch.testing.assert_close(actual_path, path)


@requires_cuda
def test_hmm_two_levels_of_chunks():
    # 5000 steps are 79 chunks of 64, whose own chain takes a second level.
    args = _model(2, 5000, 5, time_varying=False)
    ll, filtered, posteriors, score, path = _sequential(*args)
    actual_ll, actual = hmm_smoother(*args)
    torch.testing.assert_close(actual_ll, ll)
    torch.testing.assert_close(actual, posteriors)
    torch.testing.assert_close(hmm_filter(*args)[1], filtered)
    actual_score, actual_path = hmm_viterbi(*args)
    torch.testing.assert_close(actual_score, score)
    torch.testing.assert_close(actual_path, path)


@requires_cuda
@pytest.mark.parametrize("trans", ["shared", "time"])
def test_hmm_empty_sequence(trans):
    # No steps: zero log-likelihoods and scores that still take gradients,
    # all zero, with respect to every input, -inf entries included.
    log_emit, log_trans, log_init = _model(2, 1, 3, time_varying=trans == "time")
    log_emit = log_emit[:, :0]
    log_init[0, 0] = float("-inf")
    inputs = tuple(t.requires_grad_() for t in (log_emit, log_trans, log_init))
    ll, filtered = hmm_filter(*inputs)
    assert filtered.shape == (2, 0, 3) and ll.eq(0).all()
    smoothed_ll, posteriors = hmm_smoother(*inputs)
    assert posteriors.shape == (2, 0, 3) and smoothed_ll.eq(0).all()
    score, path = hmm_viterbi(*inputs)
    assert path.shape == (2, 0) and score.eq(0).all()
    total = ll.sum() + filtered.sum() + smoothed_ll.sum() + posteriors.sum() + score.sum()
    for grad, t in zip(torch.autograd.grad(total, inputs), inputs, strict=True):
        assert grad.shape == t.shape and grad.eq(0).all()


@requires_cuda
@pytest.mark.parametrize("N", [2, 7, 200])
def test_hmm_viterbi_path_is_optimal_under_ties(N):
    # Two states that must alternate, with nothing to tell them apart: the
    # two alternating paths tie. Picking each step's best state on its own
    # would stay in state 0, a path the model forbids; the traceback returns
    # one of the two, scoring the best score.
    f64 = dict(dtype=torch.float64, device="cuda")
    log_emit = torch.zeros(1, N, 2, **f64)
    log_trans = torch.tensor([[float("-inf"), 0.0], [0.0, float("-inf")]], **f64)
    log_init = torch.full((2,), 0.5, **f64).log()
    score, path = hmm_viterbi(log_emit, log_trans, log_init)
    steps = path[0, 1:] != path[0, :-1]
    assert steps.all(), "the path takes a forbidden transition"
    torch.testing.assert_close(score, log_init[path[:, 0]])


@requires_cuda
def test_hmm_viterbi_gradient_under_ties():
    # With uniform probabilities every path ties: the score's gradient is
    # that of the decoded one, so each step's sums to 1.
    B, N, K = 1, 3, 2
    f64 = dict(dtype=torch.float64, device="cuda")
    log_emit = torch.zeros(B, N, K, **f64, requires_grad=True)
    log_trans = torch.full((K, K), 0.5, **f64).log()
    log_init = torch.full((K,), 0.5, **f64).log()
    score, _ = hmm_viterbi(log_emit, log_trans, log_init)
    (grad,) = torch.autograd.grad(score.sum(), log_emit)
    torch.testing.assert_close(grad.sum(-1), torch.ones(B, N, **f64))
