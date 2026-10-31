import re
from pathlib import Path
from runpy import run_path

import pytest

ROOT = Path(__file__).resolve().parents[1]
build_support = run_path(str(ROOT / "build_support.py"))
torch_requirement = build_support["torch_requirement"]


def test_unpinned_builds_require_the_minimum_torch():
    assert torch_requirement("") == "torch >= 2.13.0"
    assert torch_requirement() == "torch >= 2.13.0"


def test_minimum_torch_matches_the_stable_abi_target():
    header = (ROOT / "philtorch" / "csrc" / "stable_support.h").read_text()
    target = re.search(
        r"#define TORCH_TARGET_VERSION"
        r" \(\(\(0ULL \+ (\d+)\) << 56\) \| \(\(0ULL \+ (\d+)\) << 48\)\)",
        header,
    )
    assert target is not None
    assert ".".join(target.groups()) == build_support["MIN_TORCH"]


def test_pinned_builds_require_the_build_minor():
    # Any patch of the minor the extension was compiled against.
    assert torch_requirement("2.14.1") == "torch == 2.14.*"


@pytest.mark.parametrize("value", ["2.14", "2.14.1+cu132", "latest", " 2.14.1"])
def test_pin_must_be_an_exact_version(value):
    with pytest.raises(RuntimeError, match="PHILTORCH_TORCH_PIN must be an exact X.Y.Z"):
        torch_requirement(value)
