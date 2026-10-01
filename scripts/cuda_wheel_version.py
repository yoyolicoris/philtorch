"""Compute the PEP 440 version for a PhilTorch CUDA wheel build.

The version is the setuptools_scm base version plus a local label identifying
the exact PyTorch/CUDA build, e.g. ``0.6.0+torch2.14.1.cu130``. The selected
PyTorch/CUDA combination must come from ``cuda_wheel_matrix.json``, so the
wheel workflow matrix is derived from one checked-in file instead of guessing
versions in several places.

Intended use in the CUDA wheel workflow::

    VERSION="$(python scripts/cuda_wheel_version.py --torch 2.14.1)"
    SETUPTOOLS_SCM_PRETEND_VERSION="$VERSION" python -m build ...

``--list`` prints the expanded build matrix (Python x PyTorch/CUDA) as JSON
for the workflow to consume.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from packaging.version import InvalidVersion, Version

REPO_ROOT = Path(__file__).parents[1]
DEFAULT_MATRIX = REPO_ROOT / "cuda_wheel_matrix.json"
SCHEMA_VERSION = 1

_TORCH_RE = re.compile(r"^\d+\.\d+\.\d+$")
_CUDA_RE = re.compile(r"^\d+\.\d+$")
_PYTHON_RE = re.compile(r"^\d+\.\d+$")


_BUILD_FIELDS = frozenset({"torch", "cuda", "torch_index_url", "cuda_arch_list"})


def load_matrix(path):
    """Read the matrix file, raising RuntimeError with context on failure."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except OSError as error:
        raise RuntimeError(f"cannot read CUDA wheel matrix {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise RuntimeError(f"CUDA wheel matrix {path} is not valid JSON: {error}") from error
    if not isinstance(data, dict):
        raise TypeError(f"CUDA wheel matrix {path} must be a JSON object")
    return data


def validate_matrix(data, path="<matrix>"):
    """Fail-loud schema check so a malformed matrix never reaches the workflow."""
    if data.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError(f"{path}: schema_version must be {SCHEMA_VERSION}")
    platform = data.get("platform")
    if not isinstance(platform, str) or not platform:
        raise RuntimeError(f"{path}: 'platform' must be a non-empty string")
    python_versions = data.get("python_versions")
    if (
        not isinstance(python_versions, list)
        or not python_versions
        or any(not isinstance(v, str) or not _PYTHON_RE.match(v) for v in python_versions)
    ):
        raise RuntimeError(f"{path}: 'python_versions' must be a non-empty list like ['3.10']")
    builds = data.get("builds")
    if not isinstance(builds, list) or not builds:
        raise RuntimeError(f"{path}: 'builds' must be a non-empty list")
    seen = set()
    for i, build in enumerate(builds):
        where = f"{path}: builds[{i}]"
        if not isinstance(build, dict):
            raise TypeError(f"{where} must be an object")
        unknown = set(build) - _BUILD_FIELDS
        if unknown:
            raise RuntimeError(f"{where}: unknown field(s) {sorted(unknown)}")
        torch_version = build.get("torch")
        cuda_version = build.get("cuda")
        if not isinstance(torch_version, str) or not _TORCH_RE.match(torch_version):
            raise RuntimeError(f"{where}: 'torch' must be an exact X.Y.Z version")
        if not isinstance(cuda_version, str) or not _CUDA_RE.match(cuda_version):
            raise RuntimeError(f"{where}: 'cuda' must be an X.Y version")
        index_url = build.get("torch_index_url")
        cuda_tag = "cu" + cuda_version.replace(".", "")
        if not isinstance(index_url, str) or not index_url.endswith("/" + cuda_tag):
            raise RuntimeError(f"{where}: 'torch_index_url' must end with '/{cuda_tag}'")
        arch_list = build.get("cuda_arch_list")
        if not isinstance(arch_list, str) or not arch_list.split():
            raise RuntimeError(f"{where}: 'cuda_arch_list' must be a non-empty string")
        key = torch_version
        if key in seen:
            raise RuntimeError(
                f"{where}: duplicate torch version {key} (one CUDA build per PyTorch version)"
            )
        seen.add(key)
    return data


def find_build(data, torch_version):
    """Return the matrix build entry for an exact PyTorch version."""
    for build in data["builds"]:
        if build["torch"] == torch_version:
            return build
    supported = ", ".join(build["torch"] for build in data["builds"])
    raise RuntimeError(
        f"torch {torch_version} is not in the CUDA wheel matrix (supported: {supported})"
    )


def base_version(root=REPO_ROOT, override=None):
    """Return the public base version, deterministically stripped of any local segment."""
    if override is not None:
        raw = override
    else:
        from setuptools_scm import get_version

        raw = get_version(root=str(root), fallback_version="0.4")
    try:
        public = Version(raw).public
    except InvalidVersion as error:
        raise RuntimeError(f"base version {raw!r} is not valid PEP 440: {error}") from error
    if not public:
        raise RuntimeError(f"base version {raw!r} has an empty public part")
    return public


def cuda_wheel_version(base, torch_version, cuda_version):
    """Build ``{base}+torch{X.Y.Z}.cu{NNN}`` and prove it is valid, deterministic PEP 440."""
    cuda_tag = "cu" + cuda_version.replace(".", "")
    label = f"torch{torch_version}.{cuda_tag}"
    version = f"{base}+{label}"
    parsed = Version(version)
    if str(parsed) != version:
        raise RuntimeError(f"version {version!r} does not normalize to itself (got {parsed})")
    if parsed.local != label:
        raise RuntimeError(f"version {version!r} has unexpected local segment {parsed.local!r}")
    return version


def expand_matrix(data):
    """Cartesian product of Python versions and torch/CUDA builds, in file order."""
    return [
        {"python": python_version, **build}
        for python_version in data["python_versions"]
        for build in data["builds"]
    ]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matrix", default=str(DEFAULT_MATRIX), help="path to cuda_wheel_matrix.json"
    )
    parser.add_argument(
        "--base-version",
        default=None,
        help="override the setuptools_scm base version (for testing)",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--torch", help="exact PyTorch version from the matrix, e.g. 2.14.1")
    group.add_argument(
        "--list", action="store_true", help="print the expanded build matrix as JSON"
    )
    args = parser.parse_args(argv)

    data = validate_matrix(load_matrix(args.matrix), args.matrix)
    if args.list:
        json.dump(expand_matrix(data), sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0

    build = find_build(data, args.torch)
    print(
        cuda_wheel_version(base_version(override=args.base_version), build["torch"], build["cuda"])
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, TypeError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from None
