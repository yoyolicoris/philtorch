"""The CUDA workflow runs on changes to every test module the GPU job runs."""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
# The cuda mark applied, on a test or on a parameter.
CUDA_MARK = re.compile(r"(@|marks=)pytest\.mark\.cuda\b")


def test_cuda_workflow_runs_on_every_cuda_test_module():
    workflow = (REPO_ROOT / ".github/workflows/build-cuda-wheels.yml").read_text()
    for module in sorted((REPO_ROOT / "tests").glob("test_*.py")):
        if CUDA_MARK.search(module.read_text()):
            # Once in the push filter and once in the pull request's.
            entry = f'- "tests/{module.name}"'
            assert workflow.count(entry) == 2, f"{module.name} is missing from a path filter"
