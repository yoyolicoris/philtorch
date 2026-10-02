"""Tests for scripts/generate_simple_index.py."""

import hashlib
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import quote

import pytest

SCRIPTS = Path(__file__).parents[1] / "scripts"
SCRIPT = SCRIPTS / "generate_simple_index.py"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


INDEX = _load("generate_simple_index")

BASE_URL = "https://example.github.io/philtorch"
CU130 = "philtorch-0.6.0+torch2.14.1.cu130-cp313-cp313-manylinux_2_28_x86_64.whl"


def asset(name, content=b"wheel", tag="v0.6.0"):
    return {
        "name": name,
        "browser_download_url": (
            f"https://github.com/example/philtorch/releases/download/{tag}/{quote(name)}"
        ),
        "digest": "sha256:" + hashlib.sha256(content).hexdigest(),
    }


def release(tag, *assets, draft=False):
    return {"tag_name": tag, "draft": draft, "prerelease": False, "assets": list(assets)}


def link(entry):
    sha256 = entry["digest"].removeprefix("sha256:")
    return f'<a href="{entry["browser_download_url"]}#sha256={sha256}">{entry["name"]}</a>'


def test_writes_one_index_per_torch_minor_and_cuda(tmp_path):
    cp312 = asset(CU130.replace("cp313", "cp312"), b"cp312")
    cp313 = asset(CU130, b"cp313")
    cu132 = asset(CU130.replace("cu130", "cu132"), b"cu132")
    older = asset("philtorch-0.6.0+torch2.13.0.cu129-cp313-cp313-manylinux_2_28_x86_64.whl")
    releases = [release("v0.6.0", cp313, older, cu132, cp312)]

    counts = INDEX.build_site(releases, tmp_path / "site", BASE_URL)

    assert counts == {"torch2.14-cu132": 1, "torch2.14-cu130": 2, "torch2.13-cu129": 1}
    page = (tmp_path / "site/whl/torch2.14-cu130/philtorch/index.html").read_text()
    assert page.index(link(cp312)) < page.index(link(cp313))
    assert cu132["name"] not in page
    root = (tmp_path / "site/whl/torch2.14-cu130/index.html").read_text()
    assert '<a href="philtorch/">philtorch</a>' in root
    assert '<meta name="pypi:repository-version" content="1.0">' in root
    landing = (tmp_path / "site/index.html").read_text()
    assert f"pip install philtorch --index-url {BASE_URL}/whl/torch2.13-cu129/" in landing
    assert landing.index("torch2.14-cu132") < landing.index("torch2.14-cu130")

    # The same releases always give the same site.
    INDEX.build_site(list(reversed(releases)), tmp_path / "again", BASE_URL + "/")
    for path in sorted((tmp_path / "site").rglob("*.html")):
        again = tmp_path / "again" / path.relative_to(tmp_path / "site")
        assert again.read_text() == path.read_text()


def test_keeps_wheels_from_older_releases(tmp_path):
    new = asset(CU130.replace("0.6.0+torch2.14.1", "0.6.1+torch2.14.2"), b"new", "v0.6.1")
    old = asset(CU130, b"old")

    INDEX.build_site([release("v0.6.1", new), release("v0.6.0", old)], tmp_path, BASE_URL)

    # A build against another patch of the same torch minor shares its index.
    page = (tmp_path / "whl/torch2.14-cu130/philtorch/index.html").read_text()
    assert page.index(link(old)) < page.index(link(new))


@pytest.mark.parametrize(
    "releases",
    [
        pytest.param([release("v0.6.0", asset(CU130), draft=True)], id="draft"),
        pytest.param([release("v0.6.0", asset("philtorch-0.6.0.tar.gz"))], id="sdist"),
        pytest.param(
            [release("v0.6.0", asset("philtorch-0.6.0-cp313-cp313-manylinux_2_28_x86_64.whl"))],
            id="cpu-wheel",
        ),
        pytest.param(
            [release("v0.6.0", asset(CU130.replace("philtorch", "torchlpc")))],
            id="other-project",
        ),
        pytest.param([release("v0.6.0")], id="no-assets"),
    ],
)
def test_skips_assets_that_do_not_belong_in_an_index(tmp_path, releases):
    assert INDEX.build_site(releases, tmp_path, BASE_URL) == {}

    assert not (tmp_path / "whl").exists()
    assert "No CUDA wheels have been published yet." in (tmp_path / "index.html").read_text()


def test_accepts_identical_duplicate_in_two_releases(tmp_path):
    first = asset(CU130, tag="v0.6.0")
    second = asset(CU130, tag="v0.6.0-copy")
    releases = [release("v0.6.0", first), release("v0.6.0-copy", second)]
    kept = min(first, second, key=lambda entry: entry["browser_download_url"])

    # Whichever release the API lists first, the page links the same copy.
    for site, order in [(tmp_path / "a", releases), (tmp_path / "b", releases[::-1])]:
        INDEX.build_site(order, site, BASE_URL)
        page = (site / "whl/torch2.14-cu130/philtorch/index.html").read_text()
        assert page.count(f">{CU130}</a>") == 1
        assert link(kept) in page


def test_rejects_conflicting_duplicate_filename(tmp_path):
    releases = [release("v0.6.0", asset(CU130, b"one")), release("v0.6.1", asset(CU130, b"two"))]

    with pytest.raises(
        ValueError, match=re.escape(f"conflicting duplicate wheel filename: {CU130}")
    ):
        INDEX.build_site(releases, tmp_path / "site", BASE_URL)

    assert not (tmp_path / "site").exists()


@pytest.mark.parametrize("digest", [None, "", "md5:0123", "sha256:" + "A" * 64])
def test_rejects_asset_without_sha256_digest(tmp_path, digest):
    entry = {**asset(CU130), "digest": digest}

    with pytest.raises(ValueError, match="v0.6.0: asset .* has no SHA-256 digest"):
        INDEX.build_site([release("v0.6.0", entry)], tmp_path, BASE_URL)


def test_rejects_asset_not_served_over_https(tmp_path):
    entry = {**asset(CU130), "browser_download_url": "http://example.com/" + CU130}

    with pytest.raises(ValueError, match="not served over HTTPS"):
        INDEX.build_site([release("v0.6.0", entry)], tmp_path, BASE_URL)


@pytest.mark.parametrize(
    "filename",
    [
        "philtorch-garbage.whl",
        "philtorch-0.6.0-x-py3-none-any.whl",
        "philtorch--py3-none-any.whl",
    ],
)
def test_rejects_malformed_wheel_filename(tmp_path, filename):
    with pytest.raises(ValueError, match="not a wheel filename"):
        INDEX.build_site([release("v0.6.0", asset(filename))], tmp_path, BASE_URL)


def test_accepts_wheel_with_build_tag(tmp_path):
    entry = asset(CU130.replace("cu130-", "cu130-1-"))

    INDEX.build_site([release("v0.6.0", entry)], tmp_path, BASE_URL)

    assert link(entry) in (tmp_path / "whl/torch2.14-cu130/philtorch/index.html").read_text()


def test_rejects_non_empty_output(tmp_path):
    (tmp_path / "stale.html").write_text("old site")

    with pytest.raises(ValueError, match="output directory is not empty"):
        INDEX.build_site([release("v0.6.0", asset(CU130))], tmp_path, BASE_URL)


@pytest.mark.parametrize("project", ["/tmp/example", "..", "philtorch\n", ""])
def test_rejects_unsafe_project_name(tmp_path, project):
    with pytest.raises(ValueError, match="invalid project name"):
        INDEX.build_site([], tmp_path / "site", BASE_URL, project)

    assert not (tmp_path / "site").exists()


def test_index_names_match_the_cuda_wheel_labels():
    version_script = _load("cuda_wheel_version")
    data = version_script.load_matrix(version_script.DEFAULT_MATRIX)
    for build in data["builds"]:
        version = version_script.cuda_wheel_version("0.6.0", build["torch"], build["cuda"])
        filename = f"philtorch-{version}-cp313-cp313-manylinux_2_28_x86_64.whl"
        minor = build["torch"].rsplit(".", 1)[0]
        cuda = build["cuda"].replace(".", "")
        assert INDEX.index_name(filename) == f"torch{minor}-cu{cuda}"


def test_cli_reads_paginated_release_list(tmp_path, monkeypatch, capsys):
    # gh api --paginate --slurp wraps each page of releases in its own list.
    pages = [[release("v0.6.1")], [release("v0.6.0", asset(CU130))]]
    (tmp_path / "releases.json").write_text(json.dumps(pages))
    argv = ["prog", str(tmp_path / "releases.json"), "--output", str(tmp_path / "site")]
    monkeypatch.setattr("sys.argv", [*argv, "--base-url", BASE_URL])

    INDEX.main()

    assert capsys.readouterr().out == "torch2.14-cu130: 1 wheels\n"
    assert (tmp_path / "site/whl/torch2.14-cu130/philtorch/index.html").exists()


@pytest.mark.parametrize(
    ("releases", "message"),
    [
        pytest.param({"message": "Not Found"}, "expected a JSON list of releases", id="not-list"),
        pytest.param(
            [release("v0.6.0", asset(CU130, b"one"), asset(CU130, b"two"))],
            "conflicting duplicate wheel filename",
            id="conflict",
        ),
    ],
)
def test_cli_reports_errors(tmp_path, monkeypatch, capsys, releases, message):
    (tmp_path / "releases.json").write_text(json.dumps(releases))
    argv = ["prog", str(tmp_path / "releases.json"), "--output", str(tmp_path / "site")]
    monkeypatch.setattr("sys.argv", [*argv, "--base-url", BASE_URL])

    with pytest.raises(SystemExit) as exit_info:
        INDEX.main()

    assert exit_info.value.code == 2
    assert message in capsys.readouterr().err


def test_script_runs_as_a_command(tmp_path):
    (tmp_path / "releases.json").write_text(json.dumps([release("v0.6.0", asset(CU130))]))

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(tmp_path / "releases.json"),
            "--output",
            str(tmp_path / "site"),
            "--base-url",
            BASE_URL,
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout == "torch2.14-cu130: 1 wheels\n"
