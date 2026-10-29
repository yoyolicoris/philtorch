import pytest
import torch

from philtorch import _recur
from philtorch._recur import _forward, recur_triton, recurrence

DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)]
# How A is shared: (batch items, steps) it has, of B and T.
SHARING = ["one", "per_item", "per_step", "both"]


def _sequential(A, zi, x):
    """h[t] = A[t] h[t - 1] + x[t], one step at a time."""
    h, states = zi, []
    for t in range(x.size(1)):
        h = (A[:, t if A.size(1) > 1 else 0] @ h.unsqueeze(-1)).squeeze(-1) + x[:, t]
        states.append(h)
    return torch.stack(states, 1)


def _problem(B, T, M, sharing, dtype=torch.float64, device="cpu", seed=0):
    """A stable recurrence: each A[t] scaled to spectral norm 0.97."""
    gen = torch.Generator().manual_seed(seed)
    Ba = B if sharing in ("per_item", "both") else 1
    Ta = T if sharing in ("per_step", "both") else 1
    A = torch.randn(Ba, Ta, M, M, dtype=torch.complex128 if dtype.is_complex else torch.float64,
                    generator=gen)  # fmt: skip
    A = A / torch.linalg.matrix_norm(A, ord=2, keepdim=True) * 0.97
    x = torch.randn(B, T, M, dtype=A.dtype, generator=gen)
    zi = torch.randn(B, M, dtype=A.dtype, generator=gen)
    return (t.to(device, dtype) for t in (A, zi, x))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("sharing", SHARING)
@pytest.mark.parametrize("M", [1, 2, 3, 5, 20, 40, 100])
@pytest.mark.parametrize("dtype", [torch.float64, torch.complex128])
def test_recurrence_matches_sequential(device, sharing, M, dtype):
    if device == "cuda" and dtype.is_complex and M > 64:
        pytest.skip("past the Triton kernels' complex states")
    # 150 steps: three chunks of the Triton scan, the last one partial.
    A, zi, x = _problem(3, 150, M, sharing, dtype, device)
    torch.testing.assert_close(recurrence(A, zi, x), _sequential(A, zi, x))


@pytest.mark.cuda
@pytest.mark.parametrize("sharing", SHARING)
@pytest.mark.parametrize("M", [3, 8, 24])
@pytest.mark.parametrize("T", [50, 300])
def test_recurrence_large_batch(sharing, M, T):
    # Sequences that fit in one chunk, which runs its steps alone, and longer
    # ones, scanned whatever the batch.
    A, zi, x = _problem(256, T, M, sharing, device="cuda")
    torch.testing.assert_close(recurrence(A, zi, x), _sequential(A, zi, x))


@pytest.mark.cuda
@pytest.mark.parametrize("sharing", ["one", "both"])
@pytest.mark.parametrize("M", [3, 8])
def test_recurrence_many_chunks_repeatedly(sharing, M):
    # Hundreds of chunks, whose starts come from look-backs that race their
    # predecessors' publishing: every run must still be right.
    A, zi, x = _problem(4, 40000, M, sharing, device="cuda")
    expected = _sequential(A, zi, x)
    for _ in range(20):
        torch.testing.assert_close(recurrence(A, zi, x), expected)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype, M", [(torch.float64, 40), (torch.complex128, 24)])
def test_recurrence_maps_in_scratch(dtype, M):
    # Time-varying maps too large for registers: their products through
    # scratch memory, across chunks whose look-backs race.
    A, zi, x = _problem(2, 1500, M, "both", dtype, "cuda")
    expected = _sequential(A, zi, x)
    for _ in range(3):
        torch.testing.assert_close(recurrence(A, zi, x), expected)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.complex64])
def test_recurrence_ignores_memory_past_the_states(dtype):
    # The scratch tiles' rows from M on are never written: NaN left in freed
    # memory, where the allocator places them, must not reach the states.
    A, zi, x = _problem(2, 1500, 40 if dtype == torch.float32 else 24, "both", dtype, "cuda")
    for _ in range(4):
        torch.full((2**24,), float("nan"), device="cuda")
    assert not recurrence(A, zi, x).isnan().any()


@pytest.mark.cuda
@pytest.mark.parametrize(
    "dtype, tolerance",
    [
        (torch.float32, 1e-5),
        (torch.complex64, 1e-5),
        (torch.float16, 3e-3),
        (torch.bfloat16, 2e-2),
    ],
)
@pytest.mark.parametrize("sharing", SHARING)
@pytest.mark.parametrize("M", [1, 2, 3, 6])
def test_recurrence_low_precision(dtype, tolerance, sharing, M):
    A, zi, x = _problem(4, 300, M, sharing, dtype, "cuda")
    expected = _sequential(*(t.to(torch.complex128 if dtype.is_complex else torch.float64)
                             for t in (A, zi, x)))  # fmt: skip
    actual = recurrence(A, zi, x)
    assert actual.dtype == dtype
    error = (actual.to(expected.dtype) - expected).abs().max() / expected.abs().max()
    assert error < tolerance


@pytest.mark.cuda
@pytest.mark.parametrize("sharing", SHARING)
@pytest.mark.parametrize("M", [16, 32])
def test_recurrence_float32_chunk_maps(sharing, M):
    # Near rotations, whose chunk maps keep their size: at 16 and 32 states,
    # Triton would compute the maps' products in TF32, of 10-bit mantissas.
    A, zi, x = _problem(4, 300, M, sharing, device="cuda")
    A = torch.linalg.qr(A)[0] * 0.999
    expected = _sequential(A, zi, x)
    actual = recurrence(*(t.float() for t in (A, zi, x)))
    assert (actual.double() - expected).abs().max() / expected.abs().max() < 1e-5


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("sharing", SHARING)
@pytest.mark.parametrize("M", [1, 2, 3, 5])
@pytest.mark.parametrize("dtype", [torch.float64, torch.complex128])
def test_recurrence_derivatives(device, sharing, M, dtype):
    # Across a chunk boundary of the Triton scan, in both modes, to second order.
    A, zi, x = _problem(2, 70, M, sharing, dtype, device)
    inputs = tuple(t.requires_grad_() for t in (A, zi, x))
    assert torch.autograd.gradcheck(recurrence, inputs, check_forward_ad=True, fast_mode=True)
    assert torch.autograd.gradgradcheck(recurrence, inputs, fast_mode=True, check_fwd_over_rev=True)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("sharing", SHARING)
@pytest.mark.parametrize("T", [1, 2])
def test_recurrence_derivatives_of_short_sequences(device, sharing, T):
    # One- and two-step sequences, whose second derivatives reach empty ones.
    A, zi, x = _problem(2, T, 3, sharing, torch.complex128, device)
    inputs = tuple(t.requires_grad_() for t in (A, zi, x))
    assert torch.autograd.gradcheck(recurrence, inputs, check_forward_ad=True)
    assert torch.autograd.gradgradcheck(recurrence, inputs)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("sharing", SHARING)
@pytest.mark.parametrize("vmap_A", [False, True])
@pytest.mark.parametrize("M", [1, 3])
def test_recurrence_vmap(device, sharing, vmap_A, M):
    # The vmapped dimension joins the batch: the same states, and gradients,
    # as one call per index, whether A is vmapped too or shared over it.
    problems = [list(_problem(2, 40, M, sharing, device=device, seed=v)) for v in range(3)]
    A, zi, x = (torch.stack(t) for t in zip(*problems))
    if not vmap_A:
        A = A[0]
    in_dims = (0 if vmap_A else None, 0, 0)
    A_of = (lambda v: A[v]) if vmap_A else (lambda v: A)
    expected = torch.stack([recurrence(A_of(v), zi[v], x[v]) for v in range(3)])
    torch.testing.assert_close(torch.func.vmap(recurrence, in_dims)(A, zi, x), expected)

    def loss(A, zi, x):
        return recurrence(A, zi, x).abs().square().sum()

    grads = torch.func.vmap(torch.func.grad(loss, argnums=(0, 1, 2)), in_dims)(A, zi, x)
    for v in range(3):
        expected = torch.func.grad(loss, argnums=(0, 1, 2))(A_of(v), zi[v], x[v])
        for g, e in zip(grads, expected):
            torch.testing.assert_close(g[v], e)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("sharing", ["one", "per_item"])
@pytest.mark.parametrize("dtype", [torch.float64, torch.complex128])
@pytest.mark.parametrize("T", [1300, 1536])
def test_recurrence_gradients_of_long_sequences(device, sharing, dtype, T):
    # The gradient of A shared over time sums over more steps than one block
    # of _CONTRACT_BLOCK: blocks, the last padded with zeros unless 2 * 1536
    # fills them, then their sum.
    A, zi, x = _problem(2, T, 3, sharing, dtype, device)
    inputs = [t.requires_grad_() for t in (A, zi, x)]
    g = torch.randn_like(x)
    actual = torch.autograd.grad(recurrence(*inputs), inputs, g)
    expected = torch.autograd.grad(_sequential(*inputs), inputs, g)
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a, e)


@pytest.mark.cuda
def test_recurrence_cuda_matches_cpu_gradients():
    A, zi, x = _problem(3, 200, 4, "both")
    grads = []
    for device in ("cpu", "cuda"):
        inputs = [t.to(device).requires_grad_() for t in (A, zi, x)]
        h = recurrence(*inputs)
        grads.append(torch.autograd.grad((h * h).sum(), inputs))
    for cpu, cuda in zip(*grads):
        torch.testing.assert_close(cuda.cpu(), cpu)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("compiled", [False, True])
def test_recurrence_empty(device, compiled):
    # No steps: no states, and zero gradients, compiled or not.
    A, zi, x = _problem(2, 5, 3, "one", device=device)
    inputs = [t.requires_grad_() for t in (A, zi, x[:, :0])]
    run = recurrence
    if compiled:
        torch._dynamo.reset()
        run = torch.compile(recurrence, backend="aot_eager", fullgraph=True)
    h = run(*inputs)
    assert h.shape == (2, 0, 3)
    for grad in torch.autograd.grad(h.sum(), inputs):
        assert not grad.any()
    if not compiled:
        # Forward mode too: no steps, no tangent.
        tangents = tuple(torch.ones_like(t) for t in inputs)
        _, tangent = torch.func.jvp(recurrence, tuple(t.detach() for t in inputs), tangents)
        assert tangent.shape == (2, 0, 3)


@pytest.mark.parametrize("device", DEVICES)
def test_recurrence_tangent_of_the_initial_state_alone(device):
    # A tangent on zi only: the tangent recurrence runs from it with no input.
    A, zi, x = _problem(2, 30, 3, "both", device=device)
    # Only zi dual, so A's and x's tangents reach the jvp as None.
    import torch.autograd.forward_ad as fwAD

    dzi = torch.randn_like(zi)
    with fwAD.dual_level():
        tangent = fwAD.unpack_dual(recurrence(A, fwAD.make_dual(zi, dzi), x)).tangent
    torch.testing.assert_close(tangent, recurrence(A, dzi, torch.zeros_like(x)))


@pytest.mark.cuda
def test_chunk_lengths_are_powers_of_two():
    # The time-invariant maps' power squares its way to the chunk length, and
    # the scratch maps' product ends in their second buffer after an even
    # number of steps: both need a power of 2.
    from philtorch import _recur_kernels

    for BM in (2, 4, 8, 16, 32, 64, 128):
        for complex_ in (False, True):
            for varying in (False, True):
                chunk = _recur_kernels._chunk(BM, complex_, varying)
                assert chunk >= 2 and chunk & (chunk - 1) == 0


@pytest.mark.cuda
def test_triton_state_limits():
    def applies(M, dtype=torch.float32, device="cuda"):
        A = torch.zeros(1, 1, M, M, dtype=dtype, device=device)
        return _recur._triton_applies(A, torch.zeros(1, 1, M, dtype=dtype, device=device))

    assert applies(128) and not applies(129)
    assert applies(64, torch.complex64) and not applies(65, torch.complex64)
    assert applies(4, torch.int64) and not applies(4, torch.int32)
    assert not applies(4, device="cpu")


def _kernel(time_invariant, M, dtype, device, B=1, T=2):
    """The launch _forward picks for A and x of these properties."""
    A = torch.zeros(1, 1, M, M, dtype=dtype, device=device).expand(
        1, 1 if time_invariant else T, M, M
    )
    x = torch.zeros(B, T, M, dtype=dtype, device=device)
    return next(run for applies, run in _recur._KERNELS if applies(A, x))


# Inputs of each kind the dispatch tells apart, and the kernel each reaches.
DISPATCH = [
    (True, 1, torch.float64, "cpu", 1, "_lti_recur"),
    (True, 3, torch.float64, "cpu", 1, "_lti_recurN"),
    (False, 1, torch.float64, "cpu", 1, "_recurN"),
    *(
        pytest.param(*case, marks=pytest.mark.cuda)
        for case in [
            (True, 1, torch.float16, "cuda", 1, "_lti_recur"),
            (True, 2, torch.complex64, "cuda", 1, "_lti_recur2"),
            (True, 3, torch.float32, "cuda", 1, "recur_triton"),
            (False, 1, torch.complex64, "cuda", 1, "_scan"),
            (False, 1, torch.float16, "cuda", 1, "recur_triton"),
            (False, 2, torch.float32, "cuda", 1, "_pararnn"),
            (False, 3, torch.float64, "cuda", 1, "_pararnn"),
            (False, 2, torch.float32, "cuda", 65536, "_recur2"),
            (False, 2, torch.float16, "cuda", 1, "_recur2"),
            (False, 2, torch.complex64, "cuda", 1, "recur_triton"),
            (False, 3, torch.float16, "cuda", 1, "recur_triton"),
        ]
    ),
]


# Integers, which the native kernels of order M take as int64 only.
INTEGER_DISPATCH = [
    (True, 1, torch.int64, "cpu", 1, "_lti_recur"),
    (True, 2, torch.int64, "cpu", 1, "_lti_recurN"),
    (False, 2, torch.int64, "cpu", 1, "_recurN"),
    *(
        pytest.param(*case, marks=pytest.mark.cuda)
        for case in [
            (True, 1, torch.int64, "cuda", 1, "_lti_recur"),
            (True, 2, torch.int64, "cuda", 1, "_lti_recur2"),
            (False, 1, torch.int64, "cuda", 1, "recur_triton"),
            (False, 2, torch.int64, "cuda", 1, "_recur2"),
            (False, 3, torch.int64, "cuda", 1, "recur_triton"),
        ]
    ),
]


@pytest.mark.parametrize("time_invariant, M, dtype, device, B, kernel", DISPATCH + INTEGER_DISPATCH)
def test_dispatch(time_invariant, M, dtype, device, B, kernel):
    assert _kernel(time_invariant, M, dtype, device, B) is getattr(_recur, kernel)


@pytest.mark.cuda
@pytest.mark.parametrize(
    "M, B, T, kernel",
    [(3, 65536, 2, "recur_triton"), (2, 1, 2**21, "_recur2"), (3, 1, 2**21, "recur_triton")],
    ids=["batch-3x3", "length-2x2", "length-3x3"],
)
def test_dispatch_past_pararnn_limits(M, B, T, kernel):
    # ParaRNN takes at most 65535 batch items and 2^21 - 1 steps in float32;
    # past them, the next kernel that takes the input runs.
    assert _kernel(False, M, torch.float32, "cuda", B, T) is getattr(_recur, kernel)


@pytest.mark.cuda
@pytest.mark.parametrize("sharing", SHARING)
def test_recur_triton_custom_op(sharing):
    # The fake implementation and the op's gradients, for torch.compile.
    A, zi, x = _problem(2, 70, 3, sharing, torch.float32, "cuda")
    torch.library.opcheck(recur_triton, (A, zi, x))
    torch.library.opcheck(recur_triton, tuple(t.requires_grad_() for t in (A, zi, x)))


@pytest.mark.parametrize("per_item", [False, True])
@pytest.mark.parametrize("time_invariant, M, dtype, device, B, kernel", DISPATCH)
def test_recurrence_compiles(time_invariant, M, dtype, device, B, kernel, per_item):
    # Compiled, recurrence runs each kernel's op whole, in one graph, and its
    # states and registered gradients match eager's, with A shared over the
    # batch or not.
    sharing = {(True, False): "one", (True, True): "per_item", (False, False): "per_step",
               (False, True): "both"}[time_invariant, per_item]  # fmt: skip
    A, zi, x = _problem(max(B, 3), 40, M, sharing, dtype, device)
    inputs = [t.requires_grad_() for t in (A, zi, x)]

    torch._dynamo.reset()
    compiled = torch.compile(recurrence, backend="aot_eager", fullgraph=True)
    expected_h, actual_h = recurrence(*inputs), compiled(*inputs)
    low = dtype in (torch.float16, torch.bfloat16)
    tolerance = {"rtol": 1e-2, "atol": 1e-2} if low else {}
    torch.testing.assert_close(actual_h, expected_h, **tolerance)
    expected = torch.autograd.grad(expected_h.abs().square().sum(), inputs)
    actual = torch.autograd.grad(actual_h.abs().square().sum(), inputs)
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a, e, **tolerance)


def test_recurrence_rejects_bad_shapes():
    A, zi, x = _problem(2, 5, 3, "both")
    with pytest.raises(ValueError, match="x must be"):
        recurrence(A, zi, x[0])
    with pytest.raises(ValueError, match="A must be"):
        recurrence(A[0], zi, x)
    with pytest.raises(ValueError, match="A must be"):
        recurrence(A[:, :2], zi, x)
    with pytest.raises(ValueError, match="zi must be"):
        recurrence(A, zi[:1], x)
    with pytest.raises(ValueError, match="share a dtype and a device"):
        recurrence(A.float(), zi, x)


@pytest.mark.parametrize(
    "device",
    ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)],
)
def test_recurrence_in_integers(device):
    # The Fibonacci numbers, exact past 2^53, through the native kernels'
    # int64; time-invariant M = 2 is lti_recur2's on CUDA.
    A = torch.tensor([[[[1, 1], [1, 0]]]], device=device)
    zi = torch.tensor([[1, 0]], device=device)
    x = torch.zeros(1, 90, 2, dtype=torch.long, device=device)
    h = recurrence(A, zi, x)
    fib = [1, 1]
    while len(fib) < 91:
        fib.append(fib[-1] + fib[-2])
    assert h.dtype == torch.long
    assert h[0, :, 0].tolist() == fib[1:]


def _integer_problem(B, T, M, sharing, dtype, device):
    """Small integers, whose states grow, and overflow, as integer arithmetic does."""
    gen = torch.Generator().manual_seed(0)
    Ba = B if sharing in ("per_item", "both") else 1
    Ta = T if sharing in ("per_step", "both") else 1
    A = torch.randint(-2, 3, (Ba, Ta, M, M), generator=gen)
    x = torch.randint(-5, 6, (B, T, M), generator=gen)
    zi = torch.randint(-5, 6, (B, M), generator=gen)
    return [t.to(device, dtype) for t in (A, zi, x)]


@pytest.mark.parametrize(
    "device",
    ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)],
)
@pytest.mark.parametrize("per_item", [False, True])
def test_recurrence_in_integers_of_order_one(device, per_item):
    # Powers of 3, exact up to 3^39 < 2^63, through lti_recur's int64.
    A = torch.full((2 if per_item else 1, 1, 1, 1), 3, device=device)
    zi = torch.ones(2, 1, dtype=torch.long, device=device)
    x = torch.zeros(2, 39, 1, dtype=torch.long, device=device)
    h = recurrence(A, zi, x)
    assert h.dtype == torch.long
    assert h[:, :, 0].tolist() == [[3**n for n in range(1, 40)]] * 2


@pytest.mark.cuda
@pytest.mark.parametrize("sharing", SHARING)
@pytest.mark.parametrize("M", [1, 3, 40])
def test_recurrence_in_integers_against_sequential(sharing, M):
    # Exact, overflow included: int64's wrap-around survives the scan's
    # reordering. M = 40 takes the scratch maps, or, time-invariant, the
    # power squared elementwise.
    A, zi, x = _integer_problem(3, 300, M, sharing, torch.int64, "cuda")
    # CUDA's matmuls take no integers: the reference runs on CPU.
    expected = _sequential(*(t.cpu() for t in (A, zi, x)))
    assert torch.equal(recurrence(A, zi, x).cpu(), expected)


@pytest.fixture
def eight_threads():
    threads = torch.get_num_threads()
    torch.set_num_threads(8)
    yield
    torch.set_num_threads(threads)


# (B, T, sharing, M) that the CPU kernels split over time at 8 threads: they
# take min(8 // B, T // 1024) chunks, when that is more than 2 for a
# constant A, and more than M + 2 for a varying one.
SPLIT_CASES = [
    (2, 5000, "per_item", 1),
    (2, 5000, "per_item", 3),
    (2, 8200, "both", 1),
    (1, 8200, "per_step", 2),
]


@pytest.mark.parametrize("dtype", [torch.float64, torch.complex128])
@pytest.mark.parametrize("B, T, sharing, M", SPLIT_CASES)
def test_cpu_recurrence_split_over_time(eight_threads, B, T, sharing, M, dtype):
    A, zi, x = _problem(B, T, M, sharing, dtype)
    torch.testing.assert_close(recurrence(A, zi, x), _sequential(A, zi, x))


def test_cpu_recurrence_split_over_time_underflows(eight_threads):
    # A chunk's transition matrix underflows to zero within its 1250 steps.
    A, zi, x = _problem(1, 5000, 2, "one")
    A = A * (0.5 / 0.97)
    torch.testing.assert_close(recurrence(A, zi, x), _sequential(A, zi, x))


def test_cpu_recurrence_unstable_is_not_split(eight_threads):
    # A chunk's transition matrix, 2^1250 I, overflows: its product with a zero
    # state would be NaN, so the sequence runs unsplit.
    A = 2 * torch.eye(2, dtype=torch.float64).expand(1, 1, 2, 2)
    zi, x = torch.zeros(1, 2, dtype=torch.float64), torch.zeros(1, 5000, 2, dtype=torch.float64)
    x[:, -1] = 1
    torch.testing.assert_close(recurrence(A, zi, x), _sequential(A, zi, x))


def test_cpu_recurrence_split_over_time_regrows_after_underflow(eight_threads):
    # Chunk 1's transition matrix passes through 1e-320 I, a denormal, and then
    # grows back to 1e30 I.
    A = torch.eye(2, dtype=torch.float64).repeat(1, 8200, 1, 1)
    A[:, 1030:1032] *= 1e-160
    A[:, 1032:1067] *= 1e10
    zi, x = torch.zeros(1, 2, dtype=torch.float64), torch.zeros(1, 8200, 2, dtype=torch.float64)
    x[:, 0] = 1
    torch.testing.assert_close(recurrence(A, zi, x), _sequential(A, zi, x))


def test_cpu_recurrence_of_no_sequences():
    A, zi, x = torch.eye(2), torch.zeros(0, 2), torch.zeros(0, 5000, 2)
    assert torch.ops.philtorch.lti_recurN(A, zi, x).shape == (0, 5000, 2)


def test_cpu_recurrence_rejects_an_initial_state_of_another_batch():
    A, zi, x = _problem(4, 8, 2, "per_step")
    with pytest.raises(RuntimeError, match="zi"):
        torch.ops.philtorch.recurN(A[0], zi[:3], x)


@pytest.mark.cuda
@pytest.mark.parametrize("op", ["lti_recur", "scan"])
def test_cuda_scalar_recurrence_rejects_an_initial_state_of_another_batch(op):
    # The CUDA kernel reads zi[b] for every sequence b of x.
    a, zi, x = (
        torch.ones(4, device="cuda"),
        torch.ones(3, device="cuda"),
        torch.ones(4, 8, device="cuda"),
    )
    args = (a, zi, x) if op == "lti_recur" else (x, x, zi)
    with pytest.raises(RuntimeError, match=r"\(B,\)"):
        getattr(torch.ops.philtorch, op)(*args)


@pytest.mark.parametrize("sharing", ["one", "per_step"])
def test_cpu_recurrence_split_over_time_in_integers(eight_threads, sharing):
    A, zi, x = _integer_problem(1, 8200, 2, sharing, torch.int64, "cpu")
    assert torch.equal(recurrence(A, zi, x), _sequential(A, zi, x))


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.int32, torch.bool])
def test_recurrence_takes_integers_as_int64_only(dtype):
    A = torch.eye(3, dtype=dtype, device="cuda")[None, None]
    zi = torch.zeros(1, 3, dtype=dtype, device="cuda")
    x = torch.ones(1, 5, 3, dtype=dtype, device="cuda")
    with pytest.raises(NotImplementedError, match="complex and int64 dtypes"):
        recurrence(A, zi, x)


def test_recurrence_raises_where_no_kernel_applies():
    # A device with no kernels, standing in for e.g. MPS beyond its one kernel.
    A, zi, x = _problem(2, 7, 3, "both", device="meta")
    with pytest.raises(NotImplementedError, match="time-varying A with M = 3 .* on meta"):
        _forward(A, zi, x)


@pytest.mark.cuda
def test_recurrence_raises_past_the_largest_triton_state():
    A, zi, x = _problem(2, 7, 129, "one", device="cuda")
    with pytest.raises(NotImplementedError, match="take M up to 128"):
        recurrence(A, zi, x)


def _near_unit_problem(B, T, M, sharing, dtype, device):
    """Poles at radius ~0.99 (M = 2: rotated by small angles) and x ~ 0.01 N(0, 1)."""
    gen = torch.Generator().manual_seed(0)
    Ba = B if sharing in ("per_item", "both") else 1
    Ta = T if sharing in ("per_step", "both") else 1
    radius = 0.985 + 0.01 * torch.rand(Ba, Ta, 1, 1, dtype=torch.float64, generator=gen)
    angle = 0.3 * torch.rand(Ba, Ta, 1, 1, dtype=torch.float64, generator=gen)
    rotation = torch.cat([angle.cos(), -angle.sin(), angle.sin(), angle.cos()], -1)
    A = radius * (rotation.view(Ba, Ta, 2, 2) if M == 2 else torch.ones(Ba, Ta, 1, 1))
    x = 0.01 * torch.randn(B, T, M, dtype=torch.float64, generator=gen)
    zi = 0.01 * torch.randn(B, M, dtype=torch.float64, generator=gen)
    return (t.to(device, dtype) for t in (A, zi, x))


# Operators whose kernels take float16 and bfloat16: (time_invariant, M, device, kernel).
LOW_PRECISION_OPERATORS = [
    (True, 1, "cpu", "_lti_recur"),
    (True, 2, "cpu", "_lti_recurN"),
    (False, 1, "cpu", "_recurN"),
    (False, 2, "cpu", "_recurN"),
    *(
        pytest.param(*case, marks=pytest.mark.cuda)
        for case in [
            (True, 1, "cuda", "_lti_recur"),
            (True, 2, "cuda", "_lti_recur2"),
            (False, 2, "cuda", "_recur2"),
        ]
    ),
]


@pytest.mark.parametrize("time_invariant, M, device, kernel", LOW_PRECISION_OPERATORS)
@pytest.mark.parametrize("batched_A", [False, True])
@pytest.mark.parametrize("dtype, tolerance", [(torch.float16, 2**-11), (torch.bfloat16, 2**-8)])
def test_native_kernels_accumulate_low_precision_in_float32(
    time_invariant, M, device, kernel, batched_A, dtype, tolerance
):
    # The kernels' state is float32: the error is that of rounding each output once.
    sharing = (
        ("per_item" if batched_A else "one")
        if time_invariant
        else ("both" if batched_A else "per_step")
    )
    A, zi, x = _near_unit_problem(4, 4000, M, sharing, dtype, device)
    assert _kernel(time_invariant, M, dtype, device) is getattr(_recur, kernel)
    # The reference runs in float64 on the same 16-bit inputs.
    expected = _sequential(*(t.cpu().double() for t in (A, zi, x)))
    actual = recurrence(A, zi, x)
    assert actual.dtype == dtype
    error = (actual.cpu().double() - expected).norm() / expected.norm()
    assert error < tolerance
