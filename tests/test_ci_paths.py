"""The CUDA workflow runs on changes to exactly the test modules the GPU job runs,
and to the tests/ modules they import."""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
# The cuda mark applied, on a test or on a parameter.
CUDA_MARK = re.compile(r"(@|marks=)pytest\.mark\.cuda\b")


def test_cuda_workflow_runs_on_every_cuda_test_module():
    workflow = (REPO_ROOT / ".github/workflows/build-cuda-wheels.yml").read_text()
    for module in sorted((REPO_ROOT / "tests").glob("test_*.py")):
        if CUDA_MARK.search(module.read_text()):
            # Once: the list is both the pull request filter and a dev push's GPU-test check.
            entry = f'- "tests/{module.name}"'
            assert workflow.count(entry) == 1, f"{module.name} is missing from the path filter"


def test_cuda_workflow_runs_on_no_other_test_module():
    # A module without CUDA tests would start the CUDA build for nothing.
    workflow = (REPO_ROOT / ".github/workflows/build-cuda-wheels.yml").read_text()
    for name in set(re.findall(r'- "tests/(test_\w+\.py)"', workflow)):
        module = REPO_ROOT / "tests" / name
        assert module.exists() and CUDA_MARK.search(module.read_text()), (
            f"{name} has no cuda-marked tests: remove it from the path filter"
        )


def test_cuda_workflow_runs_on_what_cuda_test_modules_import():
    # A change to a helper the GPU tests use must rerun them.
    workflow = (REPO_ROOT / ".github/workflows/build-cuda-wheels.yml").read_text()
    for module in sorted((REPO_ROOT / "tests").glob("test_*.py")):
        text = module.read_text()
        if CUDA_MARK.search(text):
            for name in re.findall(r"^from \.(\w+) import", text, re.MULTILINE):
                entry = f'- "tests/{name}.py"'
                assert entry in workflow, (
                    f"{module.name} imports tests/{name}.py, which the path filter misses"
                )
