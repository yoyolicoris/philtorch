import re
from pathlib import Path
from runpy import run_path

ROOT = Path(__file__).resolve().parents[1]
MIN_TORCH = run_path(str(ROOT / "build_support.py"))["MIN_TORCH"]
CSRC = ROOT / "philtorch" / "csrc"


def target_version(source):
    target = re.search(
        r"#define TORCH_TARGET_VERSION"
        r" \(\(\(0ULL \+ (\d+)\) << 56\) \| \(\(0ULL \+ (\d+)\) << 48\)\)",
        source.read_text(),
    )
    assert target is not None
    return tuple(int(part) for part in target.groups())


def test_minimum_torch_matches_the_stable_abi_target():
    # The CPU and CUDA sources target MIN_TORCH, which the wheels require.
    floor = tuple(int(part) for part in MIN_TORCH.split("."))
    assert target_version(CSRC / "stable_support.h") == floor
    # The Metal source targets the oldest torch it needs, which can be older.
    assert target_version(CSRC / "scalar_recur_mps.mm") <= floor
