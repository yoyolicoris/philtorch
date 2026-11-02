"""Tests for scripts/update_torch_matrix.py, with recorded-style upstream data.

No test touches the network: ``fetch`` is replaced by a table of responses
that ``upstream`` derives from a list of torch entries, the way PyPI,
download.pytorch.org, pytorch/pytorch and the Python release cycle would
answer for them, plus decoys the script must ignore (pre-releases, yanked
releases, aarch64 and Windows wheels, free-threaded wheels, CUDA 12.6,
Python 3.10).
"""

import datetime
import importlib.util
import json
import urllib.error
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).parents[1] / "scripts"
REPO_ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location(
    "update_torch_matrix", SCRIPTS / "update_torch_matrix.py"
)
assert SPEC is not None and SPEC.loader is not None
UPDATE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(UPDATE)
TM = UPDATE.tm

TODAY = datetime.date(2026, 10, 29)
ARCHS = "7.5 8.0 8.6 9.0 10.0 12.0"
FILES = ("torch_matrix.json", "pyproject.toml", "README.md")
EOL = {f"3.{n}": f"{2026 + n - 10}-10" for n in range(10, 18)}  # 3.10 ends 2026-10, ...


def entry(version, cuda, python="3.14"):
    return {"version": version, "cuda": cuda, "python": python}


BASE = [
    entry("2.12.1", ["12.9", "13.0"]),
    entry("2.13.0", ["12.9", "13.0"]),
    entry("2.14.1", ["13.0", "13.2"]),
    entry("2.15.0", ["13.2", "13.4"], "3.15"),
]


def upstream(torch, older=("2.11.0",), arch_404=(), tables=None):
    """The responses upstream gives for ``torch``, by URL; exceptions are failures.

    ``tables`` maps a minor to its ``TORCH_CUDA_ARCH_LIST_TABLE``; by default,
    the one every CUDA of ``torch`` has, with ``ARCHS`` for x86_64.
    """

    # Every release has wheels for the CPythons from 3.11 to its own, a 3.10
    # one (older than requires-python), a 3.16 one for aarch64 only and a
    # free-threaded one (cp316t) that is newer than the others.
    def wheel(python, platform="manylinux_2_28_x86_64", yanked=False):
        filename = f"torch-0-cp3{python}-cp3{python}-{platform}.whl"
        return {"python_version": f"cp3{python}", "filename": filename, "yanked": yanked}

    releases = {
        e["version"]: [wheel(p) for p in range(10, int(e["python"][2:]) + 1)]
        + [wheel(16, "manylinux_2_28_aarch64"), wheel("16t")]
        for e in torch
    }
    releases |= {v: [wheel(11)] for v in older}
    # A pre-release and a yanked release are newer than anything stable.
    releases |= {"2.99.0rc1": [wheel(11)], "2.98.0": [wheel(11, yanked=True)]}
    tags = sorted(
        {"cu" + c.replace(".", "") for e in torch for c in e["cuda"]} | {"cu126", "cu118"}
    )
    root = "".join(f'<a href="{t}/">{t}</a><br/>' for t in [*tags, "cpu", "nvidia-cublas-cu12"])
    urls = {UPDATE.PYPI_URL: json.dumps({"releases": releases}), UPDATE.INDEX_ROOT_URL: root}
    for tag in set(tags) - {"cu126", "cu118"}:
        # other platforms list every version for every CUDA
        wheels = [f"torch-{v}+{tag}-cp311-cp311-manylinux_2_28_aarch64.whl" for v in releases]
        wheels += [f"torch-{v}+{tag}-cp311-cp311-win_amd64.whl" for v in releases]
        wheels += [
            f"torch-{e['version']}+{tag}-cp311-cp311-manylinux_2_28_x86_64.whl"
            for e in torch
            if any("cu" + c.replace(".", "") == tag for c in e["cuda"])
        ]
        urls[f"{UPDATE.INDEX_ROOT_URL}{tag}/torch/"] = "".join(
            f'<a href="https://x/{w}#sha256=0">{w}</a><br/>' for w in wheels
        )
    for e in torch:
        minor = TM.torch_minor(e["version"])
        table = (tables or {}).get(minor) or {
            c: {"x86_64": {round(float(a) * 10) for a in ARCHS.split()}} for c in e["cuda"]
        }
        url = UPDATE.ARCH_TABLE_URL.format(minor=minor)
        urls[url] = (
            urllib.error.HTTPError(url, 404, "Not Found", {}, None)
            if minor in arch_404
            else f"_PTX_ARCHES = {{120}}\nTORCH_CUDA_ARCH_LIST_TABLE: dict = {table!r}\n"
        )
    cycle = {v: {"end_of_life": ends} for v, ends in EOL.items()}
    return urls | {UPDATE.RELEASE_CYCLE_URL: json.dumps(cycle)}


PYPROJECT = """\
# kept
[project]
# kept too
requires-python = ">=3.{requires}"
classifiers = [
    "Development Status :: 3 - Alpha",
    "Programming Language :: Python :: 3",
{classifiers}]

[tool.ruff]
line-length = 100
target-version = "py3{target}"
"""
README = "Prose.\n\n{start}\n{table}\n{end}\n\nMore prose.\n"
ARCH_LISTS = {"12": ARCHS + "+PTX", "13": ARCHS + "+PTX"}


def pyproject(oldest, newest):
    classifiers = "".join(
        f'    "Programming Language :: Python :: 3.{p}",\n' for p in range(oldest, newest + 1)
    )
    return PYPROJECT.format(requires=oldest, classifiers=classifiers, target=oldest)


def readme(torch):
    return README.format(
        start=UPDATE.TABLE_START, table=UPDATE.render_index_table(torch), end=UPDATE.TABLE_END
    )


def write_repo(root, torch, oldest=11, arch_lists=ARCH_LISTS):
    matrix = {
        "schema_version": 4,
        "policy": "kept",
        "platform": "manylinux_2_28_x86_64",
        "build_image": "quay.io/pypa/manylinux_2_28_x86_64:2026.10.26-1",
        "cuda_arch_lists": arch_lists,
        "torch": torch,
    }
    newest = max(int(e["python"][2:]) for e in torch)
    (root / FILES[0]).write_text(UPDATE.render_matrix(matrix), encoding="utf-8")
    (root / FILES[1]).write_text(pyproject(oldest, newest), encoding="utf-8")
    (root / FILES[2]).write_text(readme(torch), encoding="utf-8")
    return matrix


def run(root, urls, monkeypatch, today=TODAY, overrides=None):
    table = urls | (overrides or {})

    def fake_fetch(url):
        if isinstance(table[url], Exception):
            raise table[url]
        return table[url]

    monkeypatch.setattr(UPDATE, "fetch", fake_fetch)
    UPDATE.main(["--root", str(root)], today=today)


def read(root):
    return {name: (root / name).read_text(encoding="utf-8") for name in FILES}


def test_current_upstream_reproduces_the_repository_files(tmp_path, monkeypatch):
    # Upstream as the checked-in matrix describes it: nothing to change, now
    # and on a second run, and the README's table is the one the matrix
    # generates. The floor's branch has no arch table.
    for name in FILES:
        (tmp_path / name).write_text(
            (REPO_ROOT / name).read_text(encoding="utf-8"), encoding="utf-8"
        )
    torch = json.loads((tmp_path / FILES[0]).read_text(encoding="utf-8"))["torch"]
    urls = upstream(torch, arch_404={TM.torch_minor(torch[0]["version"])})
    before = read(tmp_path)
    for _ in range(2):
        run(tmp_path, urls, monkeypatch)
        assert read(tmp_path) == before
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(FILES)


@pytest.mark.parametrize(
    "torch, older, dropped",
    [
        # The floor (2.12) and the three newest stay: 2.13 drops, and the
        # older 2.11 is left out.
        pytest.param(
            BASE + [entry("2.16.0", ["13.2", "13.4", "13.6"], "3.15")],
            ("2.11.0",),
            "2.13.0",
            id="new-minor-keeps-floor",
        ),
        pytest.param(
            BASE[:3] + [entry("2.15.1", ["13.2", "13.4"], "3.15")],
            (),
            "2.15.0",
            id="new-patch",
        ),
        # A new CPython adds a classifier.
        pytest.param(
            [*BASE[:3], entry("2.15.0", ["13.2", "13.4"], "3.16")],
            (),
            None,
            id="new-python",
        ),
    ],
)
def test_new_release_updates_matrix(tmp_path, monkeypatch, torch, older, dropped):
    write_repo(tmp_path, BASE)
    urls = upstream(torch, older=older, arch_404={"2.12"})
    run(tmp_path, urls, monkeypatch)
    files = read(tmp_path)
    expected = [e for e in torch if e["version"] != dropped]
    matrix = json.loads(files[FILES[0]])
    assert matrix["torch"] == expected
    # The floor's arch lists are kept; the rest of the file is as it was.
    assert matrix["cuda_arch_lists"] == ARCH_LISTS
    assert matrix["policy"] == "kept"
    assert files[FILES[0]] == UPDATE.render_matrix(matrix)
    assert files[FILES[1]] == pyproject(11, max(int(e["python"][2:]) for e in expected))
    assert files[FILES[2]] == readme(expected)
    # A second run, now that upstream and the files agree, changes nothing.
    run(tmp_path, urls, monkeypatch)
    assert read(tmp_path) == files


def test_python_end_of_life_raises_requires_python(tmp_path, monkeypatch):
    write_repo(tmp_path, BASE)
    run(tmp_path, upstream(BASE, arch_404={"2.12"}), monkeypatch, today=datetime.date(2027, 11, 1))
    files = read(tmp_path)
    # Only the lines of requires-python, the classifiers and ruff's target change.
    assert files[FILES[1]] == pyproject(12, 15)
    assert json.loads(files[FILES[0]])["torch"] == BASE


def test_arch_list_comes_from_the_branch_of_the_torch_that_builds_the_wheel(tmp_path, monkeypatch):
    # The branch of 2.12, which builds both wheels, lists CUDA 12.9 only: the
    # CUDA 13 list is kept, and 2.13's table is not consulted.
    write_repo(tmp_path, BASE)
    tables = {"2.12": {"12.9": {"x86_64": {80, 90}}}, "2.13": {"13.0": {"x86_64": {120}}}}
    run(tmp_path, upstream(BASE, tables=tables), monkeypatch)
    assert json.loads(read(tmp_path)[FILES[0]])["cuda_arch_lists"] == {
        "12": "8.0 9.0+PTX",
        "13": ARCH_LISTS["13"],
    }


def test_arch_table_puts_ptx_on_the_highest_architecture_and_skips_other_hosts():
    source = """
_PTX_ARCHES = {80}
TORCH_CUDA_ARCH_LIST_TABLE: dict[str, dict[str, set[int]]] = {
    "13.0": {"x86_64": {120, 75, 100, 80}, "aarch64": {80}},
}
"""
    assert UPDATE.parse_arch_table(source, "13.0") == "7.5 8.0 10.0 12.0+PTX"
    assert UPDATE.parse_arch_table(source, "13.2") is None
    assert UPDATE.parse_arch_table("x = 1", "13.0") is None


ARCH_2_12 = UPDATE.ARCH_TABLE_URL.format(minor="2.12")


@pytest.mark.parametrize(
    "url, response, error, message",
    [
        pytest.param(
            UPDATE.PYPI_URL,
            urllib.error.URLError("down"),
            urllib.error.URLError,
            "down",
            id="fetch",
        ),
        # Only a missing page is no table: other errors fail.
        pytest.param(
            ARCH_2_12,
            urllib.error.HTTPError(ARCH_2_12, 500, "Server Error", {}, None),
            urllib.error.HTTPError,
            "Server Error",
            id="arch-server-error",
        ),
        # The result must pass the matrix validation: CUDA 13 has no sm_70.
        pytest.param(
            ARCH_2_12,
            'TORCH_CUDA_ARCH_LIST_TABLE = {"12.9": {"x86_64": {80}}, "13.0": {"x86_64": {70}}}',
            RuntimeError,
            "CUDA 13 dropped 7.0",
            id="invalid-result",
        ),
    ],
)
def test_failure_raises_and_writes_nothing(tmp_path, monkeypatch, url, response, error, message):
    write_repo(tmp_path, BASE)
    before = read(tmp_path)
    with pytest.raises(error, match=message):
        run(tmp_path, upstream(BASE), monkeypatch, overrides={url: response})
    assert read(tmp_path) == before


def test_a_new_cuda_major_without_an_arch_table_fails(tmp_path, monkeypatch):
    # The file has CUDA 12 only; the update adds 13, whose list no branch gives.
    old = [entry(e["version"], ["12.9"], e["python"]) for e in BASE]
    write_repo(tmp_path, old, arch_lists={"12": ARCH_LISTS["12"]})
    before = read(tmp_path)
    with pytest.raises(RuntimeError, match="no existing list to keep"):
        run(tmp_path, upstream(BASE, arch_404={"2.12"}), monkeypatch)
    assert read(tmp_path) == before


def test_index_table_has_a_row_per_cuda_major():
    pages = UPDATE.PAGES_URL
    assert UPDATE.render_index_table(BASE) == (
        "| CUDA | Index |\n"
        "| --- | --- |\n"
        f"| 13 | [`whl/cu13/`]({pages}whl/cu13/) |\n"
        f"| 12 | [`whl/cu12/`]({pages}whl/cu12/) |"
    )
