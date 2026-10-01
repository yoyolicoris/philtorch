"""Compute the PEP 440 version for a PhilTorch CUDA wheel build.

The version is the setuptools_scm base version plus a local label identifying
the exact PyTorch/CUDA build, e.g. ``0.6.0+torch2.14.1.cu130``. The selected
PyTorch/CUDA combination must come from ``cuda_wheel_matrix.json``, so the
wheel workflow matrix is derived from one checked-in file instead of guessing
versions in several places.

Intended use in the CUDA wheel workflow::

    VERSION="$(python scripts/cuda_wheel_version.py --torch 2.14.1 --cuda 13.0)"
    SETUPTOOLS_SCM_PRETEND_VERSION_FOR_PHILTORCH="$VERSION" python -m build ...

The ``_FOR_PHILTORCH`` form scopes the override to this project, so packages
built from source in the same environment keep their own versions.

``--list`` prints the expanded build matrix (Python x PyTorch/CUDA) as JSON
on a single line, so it can be written straight to ``$GITHUB_OUTPUT``.
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

# Patterns are matched with fullmatch, which unlike match plus "$" rejects a
# trailing newline.
_TORCH_RE = re.compile(r"\d+\.\d+\.\d+")
_CUDA_RE = re.compile(r"\d+\.\d+")
_PYTHON_RE = re.compile(r"\d+\.\d+")
# One TORCH_CUDA_ARCH_LIST entry, e.g. "8.6", "9.0a" or "12.0+PTX". "native"
# needs a visible GPU (see build_support.py), and named architectures such as
# "Ampere" hide what is built, so only explicit compute capabilities pass.
_ARCH_RE = re.compile(r"(\d+)\.(\d+)a?(\+PTX)?")

_BUILD_FIELDS = frozenset({"torch", "cuda", "torch_index_url", "cuda_arch_list"})

# Oldest CUDA the CUDA sources build with unchanged. recur2.cuh treats
# thrust::tuple and cuda::std::tuple as one type; CUDA 12.6 keeps them
# separate, so it fails to compile there. 12.9 is the oldest verified to work.
_MIN_CUDA = (12, 9)


def _cuda_tag(cuda_version):
    return "cu" + cuda_version.replace(".", "")


def _cuda_tuple(cuda_version):
    return tuple(int(part) for part in cuda_version.split("."))


def _arch_support_error(cuda_version, arch_list):
    """Return why nvcc for ``cuda_version`` cannot build ``arch_list``, or None.

    CUDA 13 dropped everything older than Turing (7.5), which an arch list
    copied from a CUDA 12 build would still contain.
    """
    if _cuda_tuple(cuda_version) < (13, 0):
        return None
    for arch in arch_list.split():
        major, minor = (int(part) for part in _ARCH_RE.fullmatch(arch).group(1, 2))
        if (major, minor) < (7, 5):
            return f"CUDA 13 dropped {arch}"
    return None


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
        or any(not isinstance(v, str) or not _PYTHON_RE.fullmatch(v) for v in python_versions)
    ):
        raise RuntimeError(f"{path}: 'python_versions' must be a non-empty list like ['3.10']")
    if len(set(python_versions)) != len(python_versions):
        raise RuntimeError(f"{path}: 'python_versions' contains a duplicate")
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
        if not isinstance(torch_version, str) or not _TORCH_RE.fullmatch(torch_version):
            raise RuntimeError(f"{where}: 'torch' must be an exact X.Y.Z version")
        if not isinstance(cuda_version, str) or not _CUDA_RE.fullmatch(cuda_version):
            raise RuntimeError(f"{where}: 'cuda' must be an X.Y version")
        if _cuda_tuple(cuda_version) < _MIN_CUDA:
            raise RuntimeError(
                f"{where}: CUDA {cuda_version} is older than "
                f"{'.'.join(map(str, _MIN_CUDA))}, the oldest the CUDA sources build with"
            )
        index_url = build.get("torch_index_url")
        cuda_tag = _cuda_tag(cuda_version)
        if not isinstance(index_url, str) or not index_url.endswith("/" + cuda_tag):
            raise RuntimeError(f"{where}: 'torch_index_url' must end with '/{cuda_tag}'")
        arch_list = build.get("cuda_arch_list")
        if (
            not isinstance(arch_list, str)
            or not arch_list.split()
            or any(not _ARCH_RE.fullmatch(arch) for arch in arch_list.split())
        ):
            raise RuntimeError(
                f"{where}: 'cuda_arch_list' must be space-separated compute capabilities "
                "like '8.6 9.0 12.0+PTX'"
            )
        arch_error = _arch_support_error(cuda_version, arch_list)
        if arch_error:
            raise RuntimeError(f"{where}: 'cuda_arch_list': {arch_error}")
        key = (torch_version, cuda_version)
        if key in seen:
            raise RuntimeError(f"{where}: duplicate torch/CUDA combination {key}")
        seen.add(key)
    return data


def find_build(data, torch_version, cuda_version):
    """Return the matrix build entry for an exact PyTorch/CUDA pair."""
    for build in data["builds"]:
        if build["torch"] == torch_version and build["cuda"] == cuda_version:
            return build
    supported = ", ".join(
        f"{build['torch']}+{_cuda_tag(build['cuda'])}" for build in data["builds"]
    )
    raise RuntimeError(
        f"torch {torch_version} with CUDA {cuda_version} is not in the CUDA wheel "
        f"matrix (supported: {supported})"
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
    return public


def cuda_wheel_version(base, torch_version, cuda_version):
    """Build ``{base}+torch{X.Y.Z}.cu{NNN}`` and prove it is valid, deterministic PEP 440."""
    label = f"torch{torch_version}.{_cuda_tag(cuda_version)}"
    version = f"{base}+{label}"
    parsed = Version(version)
    # A version holds at most one "+", so a round trip also proves that the
    # local segment is exactly the label.
    if str(parsed) != version:
        raise RuntimeError(f"version {version!r} does not normalize to itself (got {parsed})")
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
    parser.add_argument(
        "--cuda",
        default=None,
        help="CUDA version for --torch, e.g. 13.0 (required with --torch)",
    )
    args = parser.parse_args(argv)

    data = validate_matrix(load_matrix(args.matrix), args.matrix)
    if args.list:
        print(json.dumps(expand_matrix(data), separators=(",", ":")))
        return 0

    if args.cuda is None:
        parser.error("--cuda is required with --torch")
    build = find_build(data, args.torch, args.cuda)
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
