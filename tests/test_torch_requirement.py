from pathlib import Path
from runpy import run_path

import pytest

ROOT = Path(__file__).resolve().parents[1]
torch_requirement = run_path(str(ROOT / "build_support.py"))["torch_requirement"]


def test_unpinned_builds_keep_the_supported_range():
    assert torch_requirement("") == "torch >= 2.4.0"
    assert torch_requirement() == "torch >= 2.4.0"


def test_pinned_builds_require_the_build_minor():
    # Any patch of the minor the extension was compiled against.
    assert torch_requirement("2.14.1") == "torch == 2.14.*"


@pytest.mark.parametrize("value", ["2.14", "2.14.1+cu132", "latest", " 2.14.1"])
def test_pin_must_be_an_exact_version(value):
    with pytest.raises(RuntimeError, match="PHILTORCH_TORCH_PIN must be an exact X.Y.Z"):
        torch_requirement(value)
