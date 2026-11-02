"""Read ``torch_matrix.json``, the PyTorch versions the wheel workflows build and test.

Each wheel is built once and tested on several PyTorch versions; the file's
``policy`` says which build each. The subcommands print what the workflows
need, so that no workflow guesses a version: ``version`` the PEP 440 version of
a CUDA wheel, and ``legs`` the legs of the jobs, as ``key=value`` lines for
``$GITHUB_OUTPUT``::

    VERSION="$(python scripts/torch_matrix.py version --label cu13)"
    python scripts/torch_matrix.py legs --event "$GITHUB_EVENT_NAME" >> "$GITHUB_OUTPUT"

A CUDA wheel's version is the setuptools_scm base version plus a local label
naming its CUDA major, e.g. ``0.9.0+cu13``. It is passed through
``SETUPTOOLS_SCM_PRETEND_VERSION_FOR_PHILTORCH``, which scopes the override to
this project, so packages built from source in the same environment keep their
own versions.
"""

from __future__ import annotations

import argparse
import json
import re
import runpy
import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).parents[1]
INDEX_URL = "https://download.pytorch.org/whl/"

# Patterns are matched with fullmatch, which unlike match plus "$" rejects a
# trailing newline.
TORCH_RE = re.compile(r"\d+\.\d+\.\d+")
_MINOR_RE = re.compile(r"\d+\.\d+")  # a CUDA or a CPython version
# One TORCH_CUDA_ARCH_LIST entry, e.g. "8.6", "9.0a" or "12.0+PTX". "native"
# needs a visible GPU (see build_support.py), and named architectures such as
# "Ampere" hide what is built, so only explicit compute capabilities pass.
_ARCH_RE = re.compile(r"(\d+)\.(\d+)a?(\+PTX)?")
_FIELDS = {"version", "cuda", "python"}

# Defined in the package, which can't import this file; it imports only re.
MIN_CUDA = runpy.run_path(str(REPO_ROOT / "philtorch" / "_build_check.py"))["_MIN_CUDA"]


def version_tuple(version):
    return tuple(int(part) for part in version.split("."))


def cuda_major(cuda_version):
    return cuda_version.split(".")[0]


def torch_minor(torch_version):
    return torch_version.rsplit(".", 1)[0]


def oldest_python(pyproject_text):
    """The minor of the oldest CPython ``requires-python`` allows, e.g. 11."""
    requires = tomllib.loads(pyproject_text)["project"]["requires-python"]
    match = re.fullmatch(r">=3\.(\d+)", requires)
    if not match:
        raise RuntimeError(f"requires-python must be like '>=3.11', got {requires!r}")
    return int(match[1])


def validate_matrix(data, oldest, path="<matrix>"):
    """Check what a hand edit could get wrong; ``oldest`` is that of ``oldest_python``."""

    def fail(message):
        raise RuntimeError(f"{path}: {message}")

    platform = data.get("platform")
    if not isinstance(platform, str) or not platform:
        fail("'platform' must be a non-empty string")
    # The container every wheel builds in. Its name must carry the platform, so
    # the wheel is built on the glibc that auditwheel then certifies.
    build_image = data.get("build_image")
    if not isinstance(build_image, str) or platform not in build_image:
        fail(f"'build_image' must be an image for {platform!r}")
    torch = data.get("torch")
    if not isinstance(torch, list) or not torch:
        fail("'torch' must be a non-empty list")
    for i, entry in enumerate(torch):
        where = f"torch[{i}]"
        if not isinstance(entry, dict) or set(entry) != _FIELDS:
            fail(f"{where} must be an object with the fields {sorted(_FIELDS)}")
        if not isinstance(entry["version"], str) or not TORCH_RE.fullmatch(entry["version"]):
            fail(f"{where}: 'version' must be an exact X.Y.Z version")
        cuda = entry["cuda"]
        if (
            not isinstance(cuda, list)
            or not cuda
            or any(not isinstance(c, str) or not _MINOR_RE.fullmatch(c) for c in cuda)
        ):
            fail(f"{where}: 'cuda' must be a non-empty list of X.Y versions")
        keys = [version_tuple(c) for c in cuda]
        if keys != sorted(set(keys)):
            fail(f"{where}: 'cuda' must be ascending, without duplicates")
        if keys[0] < MIN_CUDA:
            fail(
                f"{where}: CUDA {cuda[0]} is older than {'.'.join(map(str, MIN_CUDA))}, "
                "the oldest the CUDA sources build with"
            )
        python = entry["python"]
        if not isinstance(python, str) or not _MINOR_RE.fullmatch(python):
            fail(f"{where}: 'python' must be an X.Y version")
        if version_tuple(python) < (3, oldest):
            fail(f"{where}: python {python} is older than requires-python (>=3.{oldest})")
    minors = [version_tuple(torch_minor(entry["version"])) for entry in torch]
    if minors != sorted(set(minors)):
        fail("'torch' must be ascending, with one entry per minor")
    arch_lists = data.get("cuda_arch_lists")
    if not isinstance(arch_lists, dict):
        fail("'cuda_arch_lists' must be an object")
    for major in sorted({cuda_major(c) for entry in torch for c in entry["cuda"]}, key=int):
        arch_list = arch_lists.get(major)
        archs = arch_list.split() if isinstance(arch_list, str) else []
        if not archs or any(not _ARCH_RE.fullmatch(arch) for arch in archs):
            fail(
                f"cuda_arch_lists[{major!r}] must be space-separated compute capabilities "
                "like '8.6 9.0 12.0+PTX'"
            )
        # CUDA 13 dropped everything older than Turing (7.5), which a list
        # copied from a CUDA 12 build would still contain.
        for arch in archs:
            if int(major) >= 13 and tuple(map(int, _ARCH_RE.fullmatch(arch).group(1, 2))) < (7, 5):
                fail(f"cuda_arch_lists[{major!r}]: CUDA 13 dropped {arch}")
    return data


def build_pairs(torch):
    """The ``(torch entry, CUDA)`` each CUDA wheel is built with, by ascending major.

    The oldest torch that has the major, and its oldest CUDA of that major.
    """
    pairs = {}
    for entry in torch:  # oldest first, CUDA ascending
        for cuda in entry["cuda"]:
            pairs.setdefault(cuda_major(cuda), (entry, cuda))
    return [pairs[major] for major in sorted(pairs, key=int)]


def legs(data, event, build_python):
    """The workflow's outputs for a GitHub event, as ``{name: string or list of legs}``."""
    torch = data["torch"]
    floor, newest = torch[0], torch[-1]
    pairs = build_pairs(torch)

    def cuda_leg(entry, cuda, python):
        return {
            "label": f"cu{cuda_major(cuda)}",
            "torch": entry["version"],
            "cuda": cuda,
            "python": python,
            "index_url": f"{INDEX_URL}cu{cuda.replace('.', '')}",
        }

    floor_leg = (floor["version"], build_python)
    if event == "pull_request":
        cpu = [(newest["version"], newest["python"]), floor_leg]
    else:
        cpu = [(entry["version"], entry["python"]) for entry in reversed(torch)] + [floor_leg]
    if event in ("release", "workflow_dispatch"):
        tests = [(entry, cuda) for entry in reversed(torch) for cuda in entry["cuda"]]
    else:  # one GPU test: the newest torch with its newest CUDA
        tests = [(newest, newest["cuda"][-1])]
    return {
        "floor": floor["version"],
        "build_python": build_python,
        "build_image": data["build_image"],
        "platform": data["platform"],
        "cpu_tests": [{"torch": t, "python": p} for t, p in dict.fromkeys(cpu)],
        "cuda_builds": [
            cuda_leg(entry, cuda, build_python)
            | {"arch_list": data["cuda_arch_lists"][cuda_major(cuda)]}
            for entry, cuda in pairs
        ],
        "cuda_tests": [cuda_leg(entry, cuda, entry["python"]) for entry, cuda in tests],
    }


def base_version(root=REPO_ROOT):
    """The setuptools_scm version without its local segment."""
    # Imported here so that the other subcommands need no packaging.
    from packaging.version import Version
    from setuptools_scm import get_version

    return Version(get_version(root=str(root), fallback_version="0.4")).public


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--root",
        default=str(REPO_ROOT),
        help="directory holding torch_matrix.json and pyproject.toml",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    version = commands.add_parser("version", help="print the version of a CUDA wheel")
    version.add_argument("--label", required=True, help="the wheel's local label, e.g. cu13")
    legs_parser = commands.add_parser("legs", help="print the workflow's legs for an event")
    legs_parser.add_argument("--event", required=True, help="the GitHub event name")
    args = parser.parse_args(argv)

    root = Path(args.root)
    oldest = oldest_python((root / "pyproject.toml").read_text(encoding="utf-8"))
    path = root / "torch_matrix.json"
    data = validate_matrix(json.loads(path.read_text(encoding="utf-8")), oldest, str(path))

    if args.command == "version":
        print(f"{base_version()}+{args.label}")
        return 0

    for key, value in legs(data, args.event, f"3.{oldest}").items():
        print(f"{key}={value if isinstance(value, str) else json.dumps(value)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from None
