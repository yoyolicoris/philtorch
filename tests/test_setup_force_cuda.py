import os
import subprocess
import sys

import pytest

from ._cases import REPO_ROOT, load_script

resolve_cuda_build = load_script("build_support.py").resolve_cuda_build


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        pytest.param(("0", "/opt/cuda", False), False, id="auto-no-gpu"),
        pytest.param(("0", "/opt/cuda", True), True, id="auto-gpu"),
        pytest.param(("0", None, True), False, id="auto-no-toolkit"),
        pytest.param(("", "/opt/cuda", False), False, id="empty-no-gpu"),
        pytest.param(("", "/opt/cuda", True), True, id="empty-gpu"),
        pytest.param(("1", "/opt/cuda", False, "8.0 8.6", "13.0"), True, id="forced-no-gpu"),
        # A visible GPU needs no arch list.
        pytest.param(("1", "/opt/cuda", True, None, "13.0"), True, id="forced-gpu"),
    ],
)
def test_cuda_build_is_selected(args, expected):
    assert resolve_cuda_build(*args) is expected


@pytest.mark.parametrize(
    ("args", "message"),
    [
        pytest.param(
            ("1", None, False),
            "PHILTORCH_FORCE_CUDA=1 was requested, but CUDA_HOME is not set",
            id="no-toolkit",
        ),
        pytest.param(
            ("1", "/opt/cuda", False, "8.0 8.6", None),
            "PyTorch has no CUDA support",
            id="cpu-torch",
        ),
        *[
            pytest.param(
                (value, "/opt/cuda", False),
                "PHILTORCH_FORCE_CUDA must be either '0' or '1'",
                id=f"value-{value!r}",
            )
            for value in ["yes", "true", "2", " 1"]
        ],
        *[
            pytest.param(
                ("1", "/opt/cuda", False, arch, "13.0"),
                "TORCH_CUDA_ARCH_LIST is unset or 'native'",
                id=f"arch-{arch!r}",
            )
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
        cwd=REPO_ROOT,
        env={**os.environ, "PHILTORCH_FORCE_CUDA": "yes"},
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "PHILTORCH_FORCE_CUDA must be either '0' or '1'" in result.stderr
