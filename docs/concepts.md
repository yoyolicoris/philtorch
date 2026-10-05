# Concepts

## Time-invariant and time-varying filters

PhilTorch's filters live in two namespaces:

- {mod}`philtorch.lti`, for linear time-invariant (LTI) filters, whose coefficients are fixed;
- {mod}`philtorch.lpv`, for linear parameter-varying (LPV) filters, whose coefficients carry a time dimension aligned with the input, so the filter can change at every sample.

With constant coefficients, an LPV function computes what its LTI counterpart does. With time-varying ones, the filter structures differ: the direct and transposed forms are different filters, and each function's docstring states which step's coefficients weight each sample.

## Shapes

Signals are batched along the first dimension and run along the last: $(B, N)$ for $B$ signals of $N$ samples. `lfilter` and {func}`~philtorch.lti.filtfilt` also accept an unbatched signal $(N)$, and return unbatched outputs when every input is unbatched; the other functions need the batch dimension.

| | LTI | LPV |
| --- | --- | --- |
| Numerator $b$ | $(B, M_b + 1)$, or $(M_b + 1)$ to share it | $(B, N, M_b + 1)$ or $(N, M_b + 1)$ |
| Denominator $a$ | $(B, M_a)$ or $(M_a)$ | $(B, N, M_a)$ or $(N, M_a)$ |
| State matrix $A$ | $(M, M)$ or $(B, M, M)$ | $(N, M, M)$ or $(B, N, M, M)$ |
| Initial state `zi` | $(B, M)$, or $(M)$ to share it | $(B, M)$ or $(M)$ |

Coefficients and states keep the input's dtype and device, and gradients flow through all of them.

## Coefficients

The denominator omits its leading coefficient, which is taken to be 1:

$$
H(z) = \frac{b_0 + b_1 z^{-1} + \cdots + b_{M_b} z^{-M_b}}{1 + a_1 z^{-1} + \cdots + a_{M_a} z^{-M_a}}.
$$

To use coefficients from SciPy, divide `b` and `a` by `a[0]`, then pass `a[1:]`, as the [](quickstart) does. The state-space backends build the filter from the {func}`~philtorch.mat.companion` matrix of `a`.

## State-space convention

Every state-space function follows the textbook convention. With $\mathbf{h}$ the state and $\mathbf{x}$ the input,

$$
\begin{aligned}
\mathbf{h}[n + 1] &= A[n]\, \mathbf{h}[n] + B[n]\, \mathbf{x}[n], \\
\mathbf{y}[n] &= C[n]\, \mathbf{h}[n] + D[n]\, \mathbf{x}[n],
\end{aligned}
$$

starting from $\mathbf{h}[0] = $ `zi`, where the matrices at step $n$ take the state from step $n$ to $n + 1$; for an LTI model they are constant. {func}`~philtorch.lti.state_space` returns the outputs and, when `zi` is given, the final state $\mathbf{h}[N]$. {func}`~philtorch.lti.state_space_recursion` returns the states after each input, $\mathbf{h}[1], \dots, \mathbf{h}[N]$.

## Initial and final states

A filter returns its final state only when it is given an initial state `zi`, and the final state can be passed as the next chunk's `zi` to filter a long signal in pieces. What the state holds depends on the structure:

| Form | `zi` holds |
| --- | --- |
| `"tdf2"` | The transposed direct form II state, the same as SciPy's `zi`; {func}`~philtorch.lti.lfiltic` and {func}`~philtorch.lti.lfilter_zi` build it |
| `"df2"` | The past values of the internal signal $w$, newest first |
| `"df1"`, `"tdf1"` | Nothing: these forms raise an error if given `zi` |

```{include} ../README.md
:start-after: <!-- docs-api-guide-start -->
:end-before: <!-- docs-api-guide-end -->
```

```{include} ../README.md
:start-after: <!-- docs-performance-start -->
:end-before: <!-- docs-performance-end -->
```
