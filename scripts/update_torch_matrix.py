#!/usr/bin/env python3
"""Rebuild PhilTorch's PyTorch-dependent configuration from upstream.

Rewrites ``torch_matrix.json``, ``pyproject.toml`` (requires-python, the
Python classifiers and ruff's target) and the CUDA index table of ``README.md``
under ``--root`` (default: the repository root), from PyPI, download.pytorch.org,
pytorch/pytorch's release branches and the CPython release cycle.
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
import torch_matrix as tm  # noqa: E402

sys.path.insert(0, str(tm.REPO_ROOT))
from build_support import MIN_TORCH  # noqa: E402

PYPI_URL = "https://pypi.org/pypi/torch/json"
RELEASE_CYCLE_URL = "https://peps.python.org/api/release-cycle.json"
ARCH_TABLE_URL = (
    "https://raw.githubusercontent.com/pytorch/pytorch/release/{minor}"
    "/.ci/wheel/linux/build_env_setup.py"
)
PAGES_URL = "https://yoyolicoris.github.io/philtorch/"
TABLE_START = "<!-- cuda-index-table:start -->"
TABLE_END = "<!-- cuda-index-table:end -->"
MATRIX_FILE, PYPROJECT_FILE, README_FILE = "torch_matrix.json", "pyproject.toml", "README.md"


def fetch(url):
    """Return the body of ``url`` as text."""
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read().decode("utf-8")


def parse_pypi_torch(data):
    """Return ``{version: CPython minors}`` of the stable releases.

    A file counts if it isn't yanked and is a Linux x86_64 wheel; its
    ``python_version`` is the CPython tag, ``cp314`` for free-threaded too.
    """
    result = {}
    for version, files in data["releases"].items():
        if tm.TORCH_RE.fullmatch(version):
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
    as ``cc * 10`` integers per host. PyTorch's release wheels carry no PTX;
    ours do, for the highest architecture.
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
            if not archs:
                return None
            return " ".join(f"{cc // 10}.{cc % 10}" for cc in sorted(archs)) + "+PTX"
    return None


def end_of_life(entry):
    """The date an entry of the release cycle ends; a bare month ends on its last day."""
    text = entry["end_of_life"]
    if re.fullmatch(r"\d{4}-\d{2}", text):
        year, month = map(int, text.split("-"))
        return datetime.date(year, month, calendar.monthrange(year, month)[1])
    return datetime.date.fromisoformat(text)


def compute_torch(oldest):
    """The torch entries: the latest patch of the floor minor (MIN_TORCH) and of the
    three newest minors, each with its CUDA versions and its newest CPython."""
    pypi = parse_pypi_torch(json.loads(fetch(PYPI_URL)))
    latest = {}
    for version in pypi:
        if tm.torch_minor(version) not in latest or tm.version_tuple(version) > tm.version_tuple(
            latest[tm.torch_minor(version)]
        ):
            latest[tm.torch_minor(version)] = version
    minors = [
        minor
        for minor in sorted(latest, key=tm.version_tuple)
        if tm.version_tuple(minor) >= tm.version_tuple(MIN_TORCH)
    ]
    tags = {
        tag: cuda
        for tag in sorted(set(re.findall(r'href="(cu\d{3})/"', fetch(tm.INDEX_URL))))
        if tm.version_tuple(cuda := f"{tag[2:-1]}.{tag[-1]}") >= tm.MIN_CUDA
    }
    listed = {tag: parse_index_versions(fetch(f"{tm.INDEX_URL}{tag}/torch/"), tag) for tag in tags}
    entries = []
    for minor in dict.fromkeys(minors[:1] + minors[-3:]):
        version = latest[minor]
        cuda = sorted(
            (cuda for tag, cuda in tags.items() if version in listed[tag]), key=tm.version_tuple
        )
        if not cuda:
            raise RuntimeError(f"torch {version} is on no CUDA index of {tm.INDEX_URL}")
        pythons = [p for p in pypi[version] if tm.version_tuple(p)[1] >= oldest]
        if not pythons:
            raise RuntimeError(f"torch {version} has no wheel for CPython 3.{oldest} or newer")
        entries.append(
            {"version": version, "cuda": cuda, "python": max(pythons, key=tm.version_tuple)}
        )
    return entries


def compute_arch_lists(torch, old):
    """The arch list of each CUDA major, from the release branch of the torch that builds it."""
    lists = {}
    for entry, cuda in tm.build_pairs(torch):
        minor, major = tm.torch_minor(entry["version"]), tm.cuda_major(cuda)
        try:
            source = fetch(ARCH_TABLE_URL.format(minor=minor))
        except urllib.error.HTTPError as error:
            if error.code != 404:
                raise
            source = ""  # no table on that branch: keep the existing list
        lists[major] = parse_arch_table(source, cuda) or old["cuda_arch_lists"].get(major)
        if lists[major] is None:
            raise RuntimeError(
                f"no x86_64 arch list for CUDA {cuda} in release/{minor} of pytorch/pytorch, "
                "and no existing list to keep"
            )
    return lists


def render_matrix(data):
    """The matrix file's text: ``json.dumps`` with indent 2, but a torch entry on one line."""

    def field(key):
        if key == "torch":
            body = ",\n".join(f"    {json.dumps(entry)}" for entry in data[key])
            return f'  "torch": [\n{body}\n  ]'
        return f"  {json.dumps(key)}: " + json.dumps(data[key], indent=2).replace("\n", "\n  ")

    return "{\n" + ",\n".join(field(key) for key in data) + "\n}\n"


def _sub_once(pattern, replacement, text, what, flags=re.MULTILINE):
    new, count = re.subn(pattern, replacement, text, flags=flags)
    if count != 1:
        raise RuntimeError(f"expected one {what}, found {count}")
    return new


def update_pyproject(text, oldest, newest):
    """Set requires-python, the classifiers for 3.oldest to 3.newest, and ruff's target."""
    text = _sub_once(
        r'^(requires-python = ">=3\.)\d+(")$', rf"\g<1>{oldest}\g<2>", text, "requires-python"
    )
    classifiers = "".join(
        f'    "Programming Language :: Python :: 3.{p}",\n' for p in range(oldest, newest + 1)
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


def render_index_table(torch):
    """The README's CUDA index table: a row per CUDA major, newest first."""
    majors = sorted({int(tm.cuda_major(c)) for e in torch for c in e["cuda"]}, reverse=True)
    lines = ["| CUDA | Index |", "| --- | --- |"]
    for major in majors:
        path = f"whl/cu{major}/"
        lines.append(f"| {major} | [`{path}`]({PAGES_URL}{path}) |")
    return "\n".join(lines)


def update_readme(text, torch):
    return _sub_once(
        rf"({re.escape(TABLE_START)}\n).*?(\n{re.escape(TABLE_END)})",
        lambda m: m[1] + render_index_table(torch) + m[2],
        text,
        f"{TABLE_START} block in README.md",
        flags=re.DOTALL,
    )


def run(root, today):
    """Compute the new text of the files under ``root`` that change."""
    names = (MATRIX_FILE, PYPROJECT_FILE, README_FILE)
    old_text = {name: (Path(root) / name).read_text(encoding="utf-8") for name in names}
    oldest = tm.oldest_python(old_text[PYPROJECT_FILE])
    old = tm.validate_matrix(json.loads(old_text[MATRIX_FILE]), oldest, MATRIX_FILE)
    cycle = json.loads(fetch(RELEASE_CYCLE_URL))
    while end_of_life(cycle[f"3.{oldest}"]) <= today:
        oldest += 1
    torch = compute_torch(oldest)
    new = dict(old, torch=torch, cuda_arch_lists=compute_arch_lists(torch, old))
    tm.validate_matrix(new, oldest, "the updated matrix")
    newest = max(tm.version_tuple(entry["python"])[1] for entry in torch)
    new_text = {
        MATRIX_FILE: render_matrix(new),
        PYPROJECT_FILE: update_pyproject(old_text[PYPROJECT_FILE], oldest, newest),
        README_FILE: update_readme(old_text[README_FILE], torch),
    }
    return {name: text for name, text in new_text.items() if text != old_text[name]}


def main(argv=None, today=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--root",
        default=str(tm.REPO_ROOT),
        help="directory holding the files; MIN_TORCH comes from this checkout, not here",
    )
    args = parser.parse_args(argv)
    today = today or datetime.datetime.now(datetime.UTC).date()
    for name, text in run(args.root, today).items():
        (Path(args.root) / name).write_text(text, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
