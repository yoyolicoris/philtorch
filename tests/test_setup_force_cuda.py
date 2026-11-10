import os
import subprocess
import sys
from pathlib import Path
from runpy import run_path

import pytest

ROOT = Path(__file__).resolve().parents[1]
resolve_cuda_build = run_path(str(ROOT / "build_support.py"))["resolve_cuda_build"]


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (("0", "/opt/cuda", False), False),  # default: CPU without a visible GPU
        (("0", "/opt/cuda", True), True),
        (("0", None, True), False),  # no toolkit
        (("", "/opt/cuda", False), False),  # empty selects automatically
        (("", "/opt/cuda", True), True),
        (("1", "/opt/cuda", False, "8.0 8.6", "13.0"), True),  # forced without a GPU
        (("1", "/opt/cuda", True, None, "13.0"), True),  # a visible GPU needs no arch list
    ],
)
def test_cuda_build_is_selected(args, expected):
    assert resolve_cuda_build(*args) is expected


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (("1", None, False), "PHILTORCH_FORCE_CUDA=1 was requested, but CUDA_HOME is not set"),
        (("1", "/opt/cuda", False, "8.0 8.6", None), "PyTorch has no CUDA support"),
        *[
            ((value, "/opt/cuda", False), "PHILTORCH_FORCE_CUDA must be either '0' or '1'")
            for value in ["yes", "true", "2", " 1"]
        ],
        *[
            (("1", "/opt/cuda", False, arch, "13.0"), "TORCH_CUDA_ARCH_LIST is unset or 'native'")
            for arch in [None, "", "native"]
        ],
    ],
)
def test_cuda_build_is_rejected(args, message):
    with pytest.raises(RuntimeError, match=message):
        resolve_cuda_build(*args)


def test_setup_py_applies_force_cuda_setting():
    # Guards the wiring, not the helper: setup.py must pass the variable to it.
    pytest.importorskip("setuptools")
    pytest.importorskip("torch")
    result = subprocess.run(
        [sys.executable, "setup.py", "--name"],
        cwd=ROOT,
        env={**os.environ, "PHILTORCH_FORCE_CUDA": "yes"},
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "PHILTORCH_FORCE_CUDA must be either '0' or '1'" in result.stderr
