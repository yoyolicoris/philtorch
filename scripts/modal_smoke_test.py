"""Run PhilTorch's CUDA tests against a built CUDA wheel on a Modal GPU.

Used by .github/workflows/build-cuda-wheels.yml. The wheel is installed with
--no-deps next to the official PyTorch build it was compiled against, so the
exact CI artifact is what gets tested, and then the CUDA-parametrized tests
from tests/ run against the installed package::

    SMOKE_PYTHON=3.13 SMOKE_TORCH=2.14.1 \\
    SMOKE_TORCH_INDEX_URL=https://download.pytorch.org/whl/cu132 SMOKE_GPU=T4 \\
    modal run scripts/modal_smoke_test.py --wheel repaired/philtorch-...whl

Modal reads its credentials from MODAL_TOKEN_ID and MODAL_TOKEN_SECRET.
"""

import os
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

# These check repository tooling (scripts/, setup.py, build_support.py) rather
# than the installed package. CPU CI runs them.
TOOLING_TESTS = (
    "test_cuda_wheel_version.py",
    "test_generate_simple_index.py",
    "test_setup_force_cuda.py",
    "test_torch_requirement.py",
)

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
    # helion is the "extra" dependency in pyproject.toml. Its kernels only run
    # on CUDA, so without it here no CI job would run their tests.
    .pip_install("pytest", "numpy", "scipy", "helion>=1.0.0,<2")
    # Run the Helion kernels with their default configs. Autotuning them on
    # every run took about 10 minutes, and the tests check results, not speed.
    .env({"HELION_AUTOTUNE_EFFORT": "none"})
)
app = modal.App("philtorch-cuda-smoke-test", image=image)


# Most of the run is CPU-bound: Helion autotuning and torch.compile compile
# kernels in parallel subprocesses, and Modal's default share is under a core.
@app.function(gpu=GPU, cpu=4, timeout=45 * 60)
def run_tests(wheel_name: str, wheel: bytes, version: str, tests: dict[str, bytes]) -> None:
    import subprocess

    work = Path("/tmp/smoke")
    for name, content in tests.items():
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
    pytest_args = ["tests", "-k", "cuda", "-p", "no:cacheprovider", "-ra"]
    pytest_args += [f"--ignore=tests/{name}" for name in TOOLING_TESTS]
    run(sys.executable, "-m", "pytest", *pytest_args)


@app.local_entrypoint()
def main(wheel: str):
    wheel_path = Path(wheel)
    # Wheel names are {name}-{version}-{tags}.whl, and versions hold no "-".
    version = wheel_path.name.split("-")[1]
    tests = {
        str(path.relative_to(REPO_ROOT)): path.read_bytes()
        for path in sorted((REPO_ROOT / "tests").glob("*.py"))
    }
    run_tests.remote(wheel_path.name, wheel_path.read_bytes(), version, tests)
