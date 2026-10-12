"""Run PhilTorch's CUDA tests against a built CUDA wheel on a Modal GPU.

Used by .github/workflows/build-cuda-wheels.yml. The wheel is installed with
--no-deps next to the official PyTorch build it was compiled against, so the
exact CI artifact is what gets tested, and then the tests from tests/ marked
``cuda`` run against the installed package, with coverage: pytest collects only
the modules that have them. Their report is
written to coverage.xml, for the workflow to upload::

    SMOKE_PYTHON=3.13 SMOKE_TORCH=2.14.1 \\
    SMOKE_TORCH_INDEX_URL=https://download.pytorch.org/whl/cu132 SMOKE_GPU=T4 \\
    modal run scripts/modal_smoke_test.py --wheel repaired/philtorch-...whl

Modal reads its credentials from MODAL_TOKEN_ID and MODAL_TOKEN_SECRET.
"""

import os
import re
import sys
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parents[1]

# Read at import time, because the image and GPU must be known before Modal
# builds them. The container imports this module again without these
# variables; its defaults there are never used to build anything.
PYTHON = os.environ.get("SMOKE_PYTHON", "3.13")
TORCH = os.environ.get("SMOKE_TORCH", "2.14.1")
TORCH_INDEX_URL = os.environ.get("SMOKE_TORCH_INDEX_URL", "https://download.pytorch.org/whl/cu132")
GPU = os.environ.get("SMOKE_GPU", "T4")

# The cuda mark applied, on a test or on a parameter, as tests/test_ci_paths.py
# reads it: pytest collects only the modules that have it.
CUDA_MARK = re.compile(r"(@|marks=)pytest\.mark\.cuda\b")

# Every op the CUDA build registers a CUDA kernel for.
CUDA_OPS = (
    "philtorch::lti_recur",
    "philtorch::lti_recur2",
    "philtorch::recur2",
    "philtorch::scan",
    "philtorch::lpc",
    "parallel_reduce_cuda::parallel_reduce_block_diag_2x2_cuda",
    "parallel_reduce_cuda::parallel_reduce_block_diag_3x3_cuda",
)

# Fail before the tests if they could only be skipped or would test the wrong
# build: no usable GPU, a different philtorch, or missing CUDA kernels.
PRECHECK = f"""
import sys
import torch
import philtorch

assert torch.cuda.is_available(), "torch sees no usable CUDA device"
major, minor = torch.cuda.get_device_capability()
print(f"torch {{torch.__version__}} (CUDA {{torch.version.cuda}}) on "
      f"{{torch.cuda.get_device_name()}}, sm_{{major}}{{minor}}")
print(f"philtorch {{philtorch.__version__}} from {{philtorch.__file__}}")
expected = sys.argv[1]
assert philtorch.__version__ == expected, f"expected philtorch {{expected}}"
missing = [op for op in {CUDA_OPS!r}
           if not torch._C._dispatch_has_kernel_for_dispatch_key(op, "CUDA")]
assert not missing, f"no CUDA kernel registered for {{missing}}"
"""

image = (
    modal.Image.debian_slim(python_version=PYTHON)
    # torch.compile's Triton backend compiles small C launchers at runtime.
    .apt_install("build-essential")
    .pip_install(f"torch=={TORCH}", index_url=TORCH_INDEX_URL)
    .pip_install("pytest", "pytest-cov", "numpy", "scipy")
)
app = modal.App("philtorch-cuda-smoke-test", image=image)


@app.function(gpu=GPU, timeout=45 * 60)
def run_tests(
    wheel_name: str, wheel: bytes, version: str, files: dict[str, bytes], modules: list[str]
) -> bytes:
    import subprocess

    work = Path("/tmp/smoke")
    for name, content in files.items():
        path = work / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    (work / wheel_name).write_bytes(wheel)

    def run(*cmd):
        print("+", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True, cwd=work)

    run("nvidia-smi")
    run(sys.executable, "-m", "pip", "install", "--no-deps", str(work / wheel_name))
    run(sys.executable, "-c", PRECHECK, version)
    pytest_args = [*modules, "-m", "cuda", "-p", "no:cacheprovider", "-ra"]
    pytest_args += ["--cov-branch", "--cov-report=xml"]
    run(sys.executable, "-m", "pytest", *pytest_args)
    return (work / "coverage.xml").read_bytes()


@app.local_entrypoint()
def main(wheel: str):
    wheel_path = Path(wheel)
    # Wheel names are {name}-{version}-{tags}.whl, and versions hold no "-".
    version = wheel_path.name.split("-")[1]
    # The tests, which import helpers from one another, and pyproject.toml for
    # their pytest and coverage settings.
    tests = sorted((REPO_ROOT / "tests").glob("*.py"))
    files = {
        str(path.relative_to(REPO_ROOT)): path.read_bytes()
        for path in [*tests, REPO_ROOT / "pyproject.toml"]
    }
    modules = [
        str(path.relative_to(REPO_ROOT))
        for path in tests
        if path.name.startswith("test_") and CUDA_MARK.search(path.read_text())
    ]
    coverage = run_tests.remote(wheel_path.name, wheel_path.read_bytes(), version, files, modules)
    Path("coverage.xml").write_bytes(coverage)
