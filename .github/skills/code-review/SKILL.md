# GitHub Copilot Code Review Skill for PhilTorch

**Description:**
Provides specialized code review guidance for the PhilTorch repository. It enforces pure functional design, PyTorch autograd correctness, C++/CUDA parallel scan performance, and numerical parity with SciPy.

**When to Use:**
- Reviewing pull requests and new code contributions.
- Auditing custom `torch.autograd.Function` implementations.
- Checking C++/CUDA kernel extensions (including Helion kernels) for filtering and matrix recursions.
- Verifying fallback PyTorch implementations (associative scans).

## Core Review Principles

1. **Pure Functional Design:** PhilTorch focuses on stateless, functional implementations for discrete-time linear filters. Flag any PRs that introduce stateful objects or classes where a functional approach (e.g., `lfilter`, `state_space`) is required.
2. **Performance First:** Digital filters are recursively defined and hard to parallelize. Ensure block-based parallel associative scans or custom CUDA kernels are highly optimized, avoid unnecessary tensor copies, and handle memory allocation efficiently.
3. **Differentiability:** All operations must be cleanly differentiable. Pay close attention to `backward` passes, checking that gradients match the forward definitions mathematically.

## Detailed Review Scope

### 1. PyTorch & Autograd Correctness
- **Backward Pass:** Verify `ctx.save_for_backward` only saves necessary tensors to prevent memory bloat. Check if `backward` correctly calculates gradients with respect to all inputs (handling `None` gracefully for non-differentiable inputs).
- **Tensor Contiguity:** Ensure tensors passed to C++/CUDA extensions are contiguous (`tensor.contiguous()`) to prevent memory stride errors.
- **Device & Dtype Handling:** Ensure new tensors are created on the correct `device` and with the proper `dtype` matching input tensors.
- **Fallback Mechanisms:** Check that pure PyTorch fallbacks (e.g., using `unroll_factor` and associative scans) are logically correct when C++ extensions are not compiled.

### 2. C++ / CUDA & Dependencies
- **Kernel Safety:** Check for out-of-bounds memory accesses in CUDA/Helion kernels, particularly at sequence boundaries during parallel associative scans.
- **Synchronization:** Ensure appropriate thread synchronization (`__syncthreads()`) is used to avoid race conditions in custom extensions.
- **Environment & Build:** Verify that dependency updates in `pixi.toml` and CMake/C++ build configurations do not break PyTorch >= 2.13 compatibility or GPU-accelerated environments.

### 3. API & Correctness
- **SciPy Parity:** Ensure filter outputs (like `lfilter`, `filtfilt`, `lfilter_zi`) maintain strict numerical parity with `scipy.signal` conventions (e.g., ensuring $a_0 = 1$ is handled properly).
- **Time-Varying Support:** Verify that LPV (Linear Parameter-Varying) implementations correctly handle multidimensional tensors with an additional time dimension without broadcasting errors.

## Output Format & Severity Levels

Structure your feedback clearly using the following severity labels:
- 🔴 **[blocking]** - Must fix before merge (e.g., autograd bugs, CUDA race conditions, failing tests, stateful regressions).
- 🟡 **[important]** - Should fix, discuss if disagree (e.g., unnecessary memory copies, sub-optimal unroll factors, missing PyTorch native fallbacks).
- 🟢 **[nit]** - Nice to have, not blocking (e.g., variable naming, inline comments).
- 💡 **[suggestion]** - Alternative approach to consider (e.g., suggesting a more efficient tensor operation).

For 🔴 and 🟡 findings, always provide a concise code snippet demonstrating the correct approach.
