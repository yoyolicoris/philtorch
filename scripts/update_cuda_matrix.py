#!/usr/bin/env python3
"""Rebuild PhilTorch's PyTorch-dependent configuration from upstream.

Rewrites ``cuda_wheel_matrix.json``, ``pyproject.toml`` (requires-python, the
Python classifiers and ruff's target) and the CUDA index table of ``README.md``
under ``--root`` (default: the repository root), from PyPI, download.pytorch.org,
pytorch/pytorch's release branches, the CPython release cycle and quay.io.
A file is rewritten only when its content changes, and nothing is written if
any step fails. Standard library only; network access goes through ``fetch``.
"""

from __future__ import annotations

import argparse
import ast
import calendar
import datetime
import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import cuda_wheel_version as cwv  # noqa: E402

PYPI_URL = "https://pypi.org/pypi/torch/json"
INDEX_ROOT_URL = "https://download.pytorch.org/whl/"
RELEASE_CYCLE_URL = "https://peps.python.org/api/release-cycle.json"
ARCH_TABLE_URL = (
    "https://raw.githubusercontent.com/pytorch/pytorch/release/{minor}"
    "/.ci/wheel/linux/build_env_setup.py"
)
# Newest first, so the first page has the newest dated tag.
QUAY_TAGS_URL = (
    "https://quay.io/api/v1/repository/pypa/manylinux_2_28_x86_64/tag/"
    "?onlyActiveTags=true&limit=100"
)
BUILD_IMAGE_REPOSITORY = "quay.io/pypa/manylinux_2_28_x86_64"
PAGES_URL = "https://yoyolicoris.github.io/philtorch/"
NEWEST_MINORS = 3
TABLE_START = "<!-- cuda-index-table:start -->"
TABLE_END = "<!-- cuda-index-table:end -->"
FILES = ("cuda_wheel_matrix.json", "pyproject.toml", "README.md")


def fetch(url):
    """Return the body of ``url`` as text."""
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read().decode("utf-8")


_key = cwv._version_tuple
_minor = cwv.torch_minor


def parse_pypi_torch(data):
    """Return ``{version: CPython minors}`` of the stable releases.

    A file counts if it isn't yanked and is a Linux x86_64 wheel; its
    ``python_version`` is the CPython tag, ``cp314`` for free-threaded too.
    """
    result = {}
    for version, files in data["releases"].items():
        if cwv._TORCH_RE.fullmatch(version):
            pythons = {
                f"3.{match[1]}"
                for file in files
                if not file["yanked"]
                and "manylinux" in file["filename"]
                and "x86_64" in file["filename"]
                and (match := re.fullmatch(r"cp3(\d+)", file["python_version"]))
            }
            if pythons:
                result[version] = pythons
    return result


def parse_index_versions(html, tag):
    """Return the torch versions an index lists a Linux x86_64 wheel of."""
    return set(
        re.findall(rf">torch-(\d+\.\d+\.\d+)\+{tag}-[^<>\"#]*manylinux[^<>\"#]*_x86_64\.whl<", html)
    )


def parse_arch_table(source, cuda):
    """Return PyTorch's x86_64 architecture list for ``cuda``, or None.

    ``TORCH_CUDA_ARCH_LIST_TABLE`` maps a CUDA version to compute capabilities
    as ``cc * 10`` integers per host. Release wheels carry no PTX
    (``_ptx_arches`` is empty for them), so the list has none either.
    """
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.AnnAssign):
            targets = [node.target]
        elif isinstance(node, ast.Assign):
            targets = node.targets
        else:
            continue
        if any(getattr(t, "id", None) == "TORCH_CUDA_ARCH_LIST_TABLE" for t in targets):
            archs = ast.literal_eval(node.value).get(cuda, {}).get("x86_64")
            return " ".join(f"{cc // 10}.{cc % 10}" for cc in sorted(archs)) if archs else None
    return None


def end_of_life(entry):
    """The date an entry of the release cycle ends; a bare month ends on its last day."""
    text = entry["end_of_life"]
    if re.fullmatch(r"\d{4}-\d{2}", text):
        year, month = map(int, text.split("-"))
        return datetime.date(year, month, calendar.monthrange(year, month)[1])
    return datetime.date.fromisoformat(text)


def newest_dated_tag(data):
    """The newest ``YYYY.MM.DD-N`` tag of a quay.io listing."""
    names = [tag["name"] for tag in data["tags"]]
    return max(
        (n for n in names if re.fullmatch(r"\d{4}\.\d{2}\.\d{2}-\d+", n)),
        key=lambda n: _key(n.replace("-", ".")),
    )


def compute_builds(old, oldest_python):
    """One build per CUDA version, for the latest patch of the newest torch minors."""
    pypi = parse_pypi_torch(json.loads(fetch(PYPI_URL)))
    latest = {}
    for version in pypi:
        if _minor(version) not in latest or _key(version) > _key(latest[_minor(version)]):
            latest[_minor(version)] = version
    versions = [latest[minor] for minor in sorted(latest, key=_key)[-NEWEST_MINORS:]]
    tags = {
        tag: cuda
        for tag in sorted(set(re.findall(r'href="(cu\d{3})/"', fetch(INDEX_ROOT_URL))))
        if _key(cuda := f"{tag[2:-1]}.{tag[-1]}") >= cwv._MIN_CUDA
    }
    listed = {
        tag: parse_index_versions(fetch(f"{INDEX_ROOT_URL}{tag}/torch/"), tag) for tag in tags
    }
    existing = {(_minor(b["torch"]), b["cuda"]): b["cuda_arch_list"] for b in old["builds"]}
    builds = []
    for version in versions:
        minor = _minor(version)
        pythons = sorted((p for p in pypi[version] if _key(p)[1] >= oldest_python), key=_key)
        try:
            source = fetch(ARCH_TABLE_URL.format(minor=minor))
        except urllib.error.HTTPError as error:
            if error.code != 404:
                raise
            source = ""  # no table on that branch: keep the existing lists
        tags_for = [tag for tag in tags if version in listed[tag]]
        if not tags_for:
            raise RuntimeError(f"torch {version} is on no CUDA index of {INDEX_ROOT_URL}")
        for tag in tags_for:
            cuda = tags[tag]
            arch_list = parse_arch_table(source, cuda) or existing.get((minor, cuda))
            if arch_list is None:
                raise RuntimeError(
                    f"no x86_64 arch list for torch {version} with CUDA {cuda} in "
                    f"release/{minor} of pytorch/pytorch, and no existing build to keep"
                )
            builds.append(
                {
                    "torch": version,
                    "cuda": cuda,
                    "torch_index_url": INDEX_ROOT_URL + tag,
                    "cuda_arch_list": arch_list,
                    "python_versions": pythons,
                }
            )
    return builds


def matrix_pythons(matrix):
    return {python for build in matrix["builds"] for python in build["python_versions"]}


def _sub_once(pattern, replacement, text, what, flags=re.MULTILINE):
    new, count = re.subn(pattern, replacement, text, flags=flags)
    if count != 1:
        raise RuntimeError(f"expected one {what}, found {count}")
    return new


def update_pyproject(text, oldest, pythons):
    """Set requires-python, the Python classifiers and ruff's target, textually."""
    text = _sub_once(
        r'^(requires-python = ">=3\.)\d+(")$', rf"\g<1>{oldest}\g<2>", text, "requires-python"
    )
    classifiers = "".join(
        f'    "Programming Language :: Python :: {p}",\n' for p in sorted(pythons, key=_key)
    )
    text = _sub_once(
        r'(?:^    "Programming Language :: Python :: 3\.\d+",\n)+',
        lambda _: classifiers,
        text,
        "block of Python classifiers",
    )
    return _sub_once(
        r'^(target-version = "py3)\d+(")$', rf"\g<1>{oldest}\g<2>", text, "ruff target-version"
    )


def render_index_table(builds):
    """The README's CUDA index table: a row per torch minor, newest first."""
    cudas = sorted({b["cuda"] for b in builds}, key=_key)
    minors = sorted({_minor(b["torch"]) for b in builds}, key=_key, reverse=True)
    present = {(_minor(b["torch"]), b["cuda"]) for b in builds}
    lines = [
        "| PyTorch | " + " | ".join(f"CUDA {cuda}" for cuda in cudas) + " |",
        "| --- |" + " --- |" * len(cudas),
    ]
    for minor in minors:
        cells = []
        for cuda in cudas:
            path = f"whl/torch{minor}-{cwv._cuda_tag(cuda)}/"
            cells.append(f"[`{path}`]({PAGES_URL}{path})" if (minor, cuda) in present else "—")
        lines.append(f"| {minor} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def update_readme(text, builds):
    return _sub_once(
        rf"({re.escape(TABLE_START)}\n).*?(\n{re.escape(TABLE_END)})",
        lambda m: m[1] + render_index_table(builds) + m[2],
        text,
        f"{TABLE_START} block in README.md",
        flags=re.DOTALL,
    )


def run(root, today):
    """Compute the new text of the files under ``root`` that change."""
    old_text = {name: (Path(root) / name).read_text(encoding="utf-8") for name in FILES}
    old = cwv.validate_matrix(json.loads(old_text[FILES[0]]), FILES[0])
    oldest = int(re.search(r'^requires-python = ">=3\.(\d+)"$', old_text[FILES[1]], re.M)[1])
    cycle = json.loads(fetch(RELEASE_CYCLE_URL))
    while end_of_life(cycle[f"3.{oldest}"]) <= today:
        oldest += 1
    new = dict(old, builds=compute_builds(old, oldest))
    if matrix_pythons(new) - matrix_pythons(old):
        tag = newest_dated_tag(json.loads(fetch(QUAY_TAGS_URL)))
        new["build_image"] = f"{BUILD_IMAGE_REPOSITORY}:{tag}"
    cwv.validate_matrix(new, "the updated matrix")
    new_text = {
        FILES[0]: json.dumps(new, indent=2) + "\n",
        FILES[1]: update_pyproject(old_text[FILES[1]], oldest, matrix_pythons(new)),
        FILES[2]: update_readme(old_text[FILES[2]], new["builds"]),
    }
    return {name: text for name, text in new_text.items() if text != old_text[name]}


def main(argv=None, today=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", default=str(cwv.REPO_ROOT), help="directory holding the files")
    args = parser.parse_args(argv)
    today = today or datetime.datetime.now(datetime.UTC).date()
    for name, text in run(args.root, today).items():
        (Path(args.root) / name).write_text(text, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
