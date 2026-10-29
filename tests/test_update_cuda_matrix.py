"""Tests for scripts/update_cuda_matrix.py, with recorded-style upstream data.

No test touches the network: ``fetch`` is replaced by a table of responses
that ``upstream`` derives from a list of builds, the way PyPI,
download.pytorch.org, pytorch/pytorch, the Python release cycle and quay.io
would answer for them, plus decoys the script must ignore (pre-releases,
yanked releases, aarch64 and Windows wheels, CUDA 12.6, Python 3.10).
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
    "update_cuda_matrix", SCRIPTS / "update_cuda_matrix.py"
)
assert SPEC is not None and SPEC.loader is not None
UPDATE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(UPDATE)
CWV = UPDATE.cwv

TODAY = datetime.date(2026, 10, 29)
ARCHS = "7.5 8.0 8.6 9.0 10.0 12.0"
IMAGE = "quay.io/pypa/manylinux_2_28_x86_64:2026.10.26-1"
NEW_IMAGE = "quay.io/pypa/manylinux_2_28_x86_64:2026.11.02-1"
FILES = ("cuda_wheel_matrix.json", "pyproject.toml", "README.md")
EOL = {f"3.{n}": f"{2026 + n - 10}-10" for n in range(10, 18)}  # 3.10 ends 2026-10, ...


def build(torch, cuda, pythons):
    return {
        "torch": torch,
        "cuda": cuda,
        "torch_index_url": UPDATE.INDEX_ROOT_URL + CWV._cuda_tag(cuda),
        "cuda_arch_list": ARCHS,
        "python_versions": pythons,
    }


OLD, NEW = ["3.11", "3.12", "3.13", "3.14"], ["3.11", "3.12", "3.13", "3.14", "3.15"]
BASE = [
    *(build("2.13.0", cuda, OLD) for cuda in ("12.9", "13.0")),
    *(build("2.14.1", cuda, OLD) for cuda in ("13.0", "13.2")),
    *(build("2.15.0", cuda, NEW) for cuda in ("13.2", "13.4")),
]


def upstream(builds, older=("2.12.1",), arch_404=()):
    """The responses upstream gives for ``builds``, by URL; exceptions are failures."""
    pythons = {}
    for b in builds:
        pythons.setdefault(b["torch"], set()).update(b["python_versions"])
    pythons.update({version: {"3.11"} for version in older})

    # A pre-release and a yanked release are newer than anything stable. Every
    # release also has a CPython 3.10 wheel (older than requires-python) and a
    # 3.16 wheel that is for aarch64 only.
    def wheel(python, platform="manylinux_2_28_x86_64", yanked=False):
        filename = f"torch-0-cp3{python}-cp3{python}-{platform}.whl"
        return {"python_version": f"cp3{python}", "filename": filename, "yanked": yanked}

    releases = {
        version: [wheel(p[2:]) for p in sorted(py)]
        + [wheel(10), wheel(16, "manylinux_2_28_aarch64")]
        for version, py in pythons.items()
    }
    releases |= {"2.99.0rc1": [wheel(11)], "2.98.0": [wheel(11, yanked=True)]}
    pypi = {"releases": releases}
    tags = sorted({CWV._cuda_tag(b["cuda"]) for b in builds} | {"cu126", "cu118"})
    root = "".join(f'<a href="{t}/">{t}</a><br/>' for t in [*tags, "cpu", "nvidia-cublas-cu12"])
    urls = {UPDATE.PYPI_URL: json.dumps(pypi), UPDATE.INDEX_ROOT_URL: root}
    for tag in set(tags) - {"cu126", "cu118"}:
        wheels = [  # other platforms list every version for every CUDA
            f"torch-{v}+{tag}-cp311-cp311-manylinux_2_28_aarch64.whl" for v in pythons
        ]
        wheels += [f"torch-{v}+{tag}-cp311-cp311-win_amd64.whl" for v in pythons]
        wheels += [
            f"torch-{b['torch']}+{tag}-cp{p[2:]}-cp{p[2:]}-manylinux_2_28_x86_64.whl"
            for b in builds
            if CWV._cuda_tag(b["cuda"]) == tag
            for p in b["python_versions"]
        ]
        urls[f"{UPDATE.INDEX_ROOT_URL}{tag}/torch/"] = "".join(
            f'<a href="https://x/{w}#sha256=0">{w}</a><br/>' for w in wheels
        )
    for minor in {UPDATE._minor(b["torch"]) for b in builds}:
        table = {
            b["cuda"]: {"x86_64": {round(float(a) * 10) for a in b["cuda_arch_list"].split()}}
            for b in builds
            if UPDATE._minor(b["torch"]) == minor
        }
        url = UPDATE.ARCH_TABLE_URL.format(minor=minor)
        urls[url] = (
            urllib.error.HTTPError(url, 404, "Not Found", {}, None)
            if minor in arch_404
            else f"_PTX_ARCHES = {{120}}\nTORCH_CUDA_ARCH_LIST_TABLE: dict = {table!r}\n"
        )
    cycle = {v: {"end_of_life": ends} for v, ends in EOL.items()}
    tags_json = {
        "tags": [
            {"name": n}
            for n in (
                "latest",
                "2026.10.26-1",
                NEW_IMAGE.split(":")[1],
                "2026.10.30-0",
                "cp313-cp313",
            )
        ]
    }
    return urls | {
        UPDATE.RELEASE_CYCLE_URL: json.dumps(cycle),
        UPDATE.QUAY_TAGS_URL: json.dumps(tags_json),
    }


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


def pyproject(oldest, pythons):
    classifiers = "".join(f'    "Programming Language :: Python :: {p}",\n' for p in pythons)
    return PYPROJECT.format(requires=oldest, classifiers=classifiers, target=oldest)


def write_repo(root, builds, oldest=11, image=IMAGE):
    pythons = sorted({p for b in builds for p in b["python_versions"]}, key=UPDATE._key)
    matrix = {
        "builds": builds,
        "build_image": image,
        "platform": "manylinux_2_28_x86_64",
        "policy": "kept",
        "schema_version": 3,
    }
    table = UPDATE.render_index_table(builds)
    (root / "cuda_wheel_matrix.json").write_text(json.dumps(matrix, indent=2) + "\n")
    (root / "pyproject.toml").write_text(pyproject(oldest, pythons))
    (root / "README.md").write_text(
        README.format(start=UPDATE.TABLE_START, table=table, end=UPDATE.TABLE_END)
    )
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
    return {name: (root / name).read_text() for name in FILES}


def test_current_upstream_reproduces_the_repository_files(tmp_path, monkeypatch):
    # Upstream as the checked-in matrix describes it: nothing to change, now
    # and on a second run, and the README's table is the one the matrix
    # generates. The newest minor has an arch table, older ones 404.
    for name in FILES:
        (tmp_path / name).write_text((REPO_ROOT / name).read_text())
    matrix = json.loads((tmp_path / FILES[0]).read_text())
    minors = sorted({UPDATE._minor(b["torch"]) for b in matrix["builds"]}, key=UPDATE._key)
    urls = upstream(matrix["builds"], arch_404=minors[:-1])
    before = read(tmp_path)
    for _ in range(2):
        run(tmp_path, urls, monkeypatch)
        assert read(tmp_path) == before
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(FILES)


@pytest.mark.parametrize(
    "builds, older, image",
    [
        pytest.param(
            [b for b in BASE if not b["torch"].startswith("2.13")]
            + [build("2.16.0", c, NEW) for c in ("13.2", "13.4", "13.6")],
            ("2.13.0",),
            IMAGE,
            id="new-minor-drops-oldest",
        ),
        pytest.param(
            [b for b in BASE if not b["torch"].startswith("2.13")]
            + [build("2.16.0", c, [*NEW, "3.16"]) for c in ("13.2", "13.4")],
            ("2.13.0",),
            NEW_IMAGE,
            id="new-python-moves-image",
        ),
        pytest.param(
            [dict(b, torch="2.15.1") if b["torch"] == "2.15.0" else b for b in BASE],
            (),
            IMAGE,
            id="new-patch-keeps-image",
        ),
    ],
)
def test_new_release_updates_matrix(tmp_path, monkeypatch, builds, older, image):
    write_repo(tmp_path, BASE)
    urls = upstream(builds, older=older)
    run(tmp_path, urls, monkeypatch)
    files = read(tmp_path)
    matrix = CWV.validate_matrix(json.loads(files[FILES[0]]))
    assert matrix["build_image"] == image
    assert matrix["builds"] == sorted(
        builds, key=lambda b: (UPDATE._key(b["torch"]), UPDATE._key(b["cuda"]))
    )
    assert files[FILES[0]] == json.dumps(matrix, indent=2) + "\n"
    pythons = sorted({p for b in builds for p in b["python_versions"]}, key=UPDATE._key)
    assert files[FILES[1]] == pyproject(11, pythons)
    assert files[FILES[2]] == README.format(
        start=UPDATE.TABLE_START,
        table=UPDATE.render_index_table(matrix["builds"]),
        end=UPDATE.TABLE_END,
    )
    # A second run, now that upstream and the files agree, changes nothing.
    run(tmp_path, urls, monkeypatch)
    assert read(tmp_path) == files


def test_python_end_of_life_raises_requires_python(tmp_path, monkeypatch):
    write_repo(tmp_path, BASE)
    run(tmp_path, upstream(BASE), monkeypatch, today=datetime.date(2027, 11, 1))
    files = read(tmp_path)
    # Only the lines of requires-python, the classifiers and ruff's target change.
    assert files[FILES[1]] == pyproject(12, ["3.12", "3.13", "3.14", "3.15"])
    assert json.loads(files[FILES[0]])["builds"] == [
        dict(b, python_versions=b["python_versions"][1:]) for b in BASE
    ]


def test_arch_list_comes_from_pytorch_and_ignores_ptx_and_other_hosts():
    source = """
_PTX_ARCHES = {120}
TORCH_CUDA_ARCH_LIST_TABLE: dict[str, dict[str, set[int]]] = {
    "13.0": {"x86_64": {120, 75, 100, 80}, "aarch64": {80}},
}
"""
    assert UPDATE.parse_arch_table(source, "13.0") == "7.5 8.0 10.0 12.0"
    assert UPDATE.parse_arch_table(source, "13.2") is None
    assert UPDATE.parse_arch_table("x = 1", "13.0") is None


ARCH_2_16 = UPDATE.ARCH_TABLE_URL.format(minor="2.16")


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
        # A new build has no existing entry whose list could be kept.
        pytest.param(
            ARCH_2_16,
            urllib.error.HTTPError(ARCH_2_16, 404, "Not Found", {}, None),
            RuntimeError,
            "no existing build to keep",
            id="arch-missing-for-new-build",
        ),
        # The result must pass the matrix validation: CUDA 13 has no sm_70.
        pytest.param(
            ARCH_2_16,
            'TORCH_CUDA_ARCH_LIST_TABLE = {"13.2": {"x86_64": {70}}, "13.4": {"x86_64": {70}}}',
            RuntimeError,
            "CUDA 13 dropped 7.0",
            id="invalid-result",
        ),
    ],
)
def test_failure_raises_and_writes_nothing(tmp_path, monkeypatch, url, response, error, message):
    write_repo(tmp_path, BASE)
    # A new minor and a new Python, so that every source is read.
    builds = BASE[2:] + [build("2.16.0", c, [*NEW, "3.16"]) for c in ("13.2", "13.4")]
    before = read(tmp_path)
    with pytest.raises(error, match=message):
        run(tmp_path, upstream(builds, older=("2.13.0",)), monkeypatch, overrides={url: response})
    assert read(tmp_path) == before
