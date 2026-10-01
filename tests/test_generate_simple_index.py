import hashlib
import importlib.util
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import quote

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "generate_simple_index.py"
SPEC = importlib.util.spec_from_file_location("generate_simple_index", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
INDEX = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(INDEX)


def wheel(directory, filename, content):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    path.write_bytes(content)
    return path


def test_generates_sorted_hash_links_and_root_page(tmp_path):
    output = tmp_path / "site"
    first_name = "philtorch-0.6.0+torch2.8.cu128-cp310-cp310-linux_x86_64.whl"
    second_name = "philtorch-0.6.0+torch2.9.cu130-cp313-cp313-linux_x86_64.whl"
    first = wheel(tmp_path / "wheels", first_name, b"first wheel")
    second = wheel(tmp_path / "wheels", second_name, b"second wheel")

    INDEX.generate_index("PhilTorch", [second, first], output)

    package_dir = output / "simple" / "philtorch"
    page = (package_dir / "index.html").read_text()
    first_hash = hashlib.sha256(b"first wheel").hexdigest()
    second_hash = hashlib.sha256(b"second wheel").hexdigest()
    assert page.index(first_name) < page.index(second_name)
    assert f"{quote(first_name)}#sha256={first_hash}" in page
    assert f"{quote(second_name)}#sha256={second_hash}" in page
    assert (package_dir / first_name).read_bytes() == b"first wheel"
    assert (package_dir / second_name).read_bytes() == b"second wheel"
    root = (output / "simple/index.html").read_text()
    assert 'href="philtorch/"' in root
    assert '<meta name="pypi:repository-version" content="1.0">' in root

    mtime = (package_dir / "index.html").stat().st_mtime_ns
    INDEX.generate_index("philtorch", [first, second], output)
    assert (package_dir / "index.html").read_text() == page
    assert (package_dir / "index.html").stat().st_mtime_ns == mtime


def test_accepts_identical_duplicate_filename(tmp_path):
    output = tmp_path / "site"
    filename = "philtorch-0.6.0+torch2.9.cu130-cp312-cp312-linux_x86_64.whl"
    first = wheel(tmp_path / "one", filename, b"same wheel")
    second = wheel(tmp_path / "two", filename, b"same wheel")

    INDEX.generate_index("philtorch", [first, second], output)

    page = (output / "simple/philtorch/index.html").read_text()
    assert page.count(f">{filename}</a>") == 1


def test_rejects_conflicting_duplicate_input_filename(tmp_path):
    output = tmp_path / "site"
    filename = "philtorch-0.6.0+torch2.9.cu130-cp312-cp312-linux_x86_64.whl"
    first = wheel(tmp_path / "one", filename, b"first wheel")
    second = wheel(tmp_path / "two", filename, b"different wheel")

    with pytest.raises(
        ValueError, match=re.escape(f"conflicting duplicate wheel filename: {filename}")
    ):
        INDEX.generate_index("philtorch", [first, second], output)

    assert not output.exists()


def test_rejects_conflict_with_existing_published_wheel(tmp_path):
    output = tmp_path / "site"
    filename = "philtorch-0.6.0+torch2.9.cu130-cp312-cp312-linux_x86_64.whl"
    published = wheel(output / "simple/philtorch", filename, b"published wheel")
    incoming = wheel(tmp_path / "incoming", filename, b"changed wheel")

    with pytest.raises(
        ValueError, match=re.escape(f"conflicting duplicate wheel filename: {filename}")
    ):
        INDEX.generate_index("philtorch", [incoming], output)

    assert published.read_bytes() == b"published wheel"


@pytest.mark.parametrize("project", ["/tmp/example", "..", "philtorch\n", ""])
def test_rejects_unsafe_project_name(tmp_path, project):
    output = tmp_path / "site"

    with pytest.raises(ValueError, match="invalid project name"):
        INDEX.generate_index(project, [], output)

    assert not output.exists()


def test_incremental_run_keeps_previously_published_wheels(tmp_path):
    output = tmp_path / "site"
    first_name = "philtorch-0.6.0+torch2.8.cu128-cp310-cp310-linux_x86_64.whl"
    second_name = "philtorch-0.6.0+torch2.9.cu130-cp313-cp313-linux_x86_64.whl"
    first = wheel(tmp_path / "wheels", first_name, b"first wheel")
    second = wheel(tmp_path / "wheels", second_name, b"second wheel")

    INDEX.generate_index("philtorch", [first], output)
    INDEX.generate_index("philtorch", [second], output)

    package_dir = output / "simple" / "philtorch"
    page = (package_dir / "index.html").read_text()
    first_hash = hashlib.sha256(b"first wheel").hexdigest()
    second_hash = hashlib.sha256(b"second wheel").hexdigest()
    assert f"{quote(first_name)}#sha256={first_hash}" in page
    assert f"{quote(second_name)}#sha256={second_hash}" in page
    assert (package_dir / first_name).read_bytes() == b"first wheel"
    assert (package_dir / second_name).read_bytes() == b"second wheel"


def test_accepts_wheel_with_build_tag(tmp_path):
    output = tmp_path / "site"
    filename = "philtorch-0.6.0-1-py3-none-any.whl"

    INDEX.generate_index("philtorch", [wheel(tmp_path, filename, b"wheel")], output)

    assert (output / "simple/philtorch" / filename).read_bytes() == b"wheel"


@pytest.mark.parametrize(
    ("filename", "message"),
    [
        ("philtorch-garbage.whl", "not a wheel filename"),
        ("philtorch-0.6.0-x-py3-none-any.whl", "not a wheel filename"),
        ("philtorch--py3-none-any.whl", "not a wheel filename"),
        ("philtorch-0.6.0-py3-none-any.zip", "not a wheel filename"),
        ("torch-2.9.0-cp312-cp312-linux_x86_64.whl", "does not belong to philtorch"),
    ],
)
def test_rejects_malformed_or_foreign_wheel(tmp_path, filename, message):
    output = tmp_path / "site"

    with pytest.raises(ValueError, match=message):
        INDEX.generate_index("philtorch", [wheel(tmp_path, filename, b"x")], output)

    assert not output.exists()


def test_cli_expands_wheel_directories(tmp_path, monkeypatch):
    output = tmp_path / "site"
    filename = "philtorch-0.6.0+torch2.9.cu130-cp312-cp312-linux_x86_64.whl"
    wheel(tmp_path / "wheels", filename, b"wheel")
    wheel(tmp_path / "wheels", "notes.txt", b"ignored")
    monkeypatch.setattr(
        "sys.argv", ["prog", str(tmp_path / "wheels"), "--output", str(output)]
    )

    INDEX.main()

    assert sorted(p.name for p in (output / "simple/philtorch").iterdir()) == [
        "index.html",
        filename,
    ]


def test_cli_rejects_inputs_without_wheels(tmp_path, monkeypatch, capsys):
    (tmp_path / "empty").mkdir()
    output = tmp_path / "site"
    monkeypatch.setattr(
        "sys.argv", ["prog", str(tmp_path / "empty"), "--output", str(output)]
    )

    with pytest.raises(SystemExit) as exit_info:
        INDEX.main()

    assert exit_info.value.code == 2
    assert "no wheel files found" in capsys.readouterr().err
    assert not output.exists()


def test_cli_reports_generator_errors(tmp_path, monkeypatch, capsys):
    output = tmp_path / "site"
    filename = "philtorch-0.6.0+torch2.9.cu130-cp312-cp312-linux_x86_64.whl"
    wheel(output / "simple/philtorch", filename, b"published wheel")
    incoming = wheel(tmp_path / "incoming", filename, b"changed wheel")
    monkeypatch.setattr("sys.argv", ["prog", str(incoming), "--output", str(output)])

    with pytest.raises(SystemExit) as exit_info:
        INDEX.main()

    assert exit_info.value.code == 2
    assert "conflicting duplicate wheel filename" in capsys.readouterr().err


def test_rejects_missing_wheel_path(tmp_path):
    missing = tmp_path / "philtorch-0.6.0-py3-none-any.whl"

    with pytest.raises(ValueError, match="wheel path is not a file"):
        INDEX.generate_index("philtorch", [missing], tmp_path / "site")


def test_failed_copy_leaves_no_partial_wheel(tmp_path, monkeypatch):
    output = tmp_path / "site"
    filename = "philtorch-0.6.0+torch2.9.cu130-cp312-cp312-linux_x86_64.whl"
    source = wheel(tmp_path / "wheels", filename, b"complete wheel")

    def truncated_copy(src, dst):
        Path(dst).write_bytes(b"partial")

    monkeypatch.setattr(INDEX.shutil, "copyfile", truncated_copy)
    with pytest.raises(ValueError, match="copied wheel failed digest check"):
        INDEX.generate_index("philtorch", [source], output)

    assert list((output / "simple/philtorch").iterdir()) == []

    monkeypatch.undo()
    INDEX.generate_index("philtorch", [source], output)
    assert (output / "simple/philtorch" / filename).read_bytes() == b"complete wheel"


def test_script_runs_as_a_command(tmp_path):
    output = tmp_path / "site"
    filename = "philtorch-0.6.0+torch2.9.cu130-cp312-cp312-linux_x86_64.whl"
    wheel(tmp_path / "wheels", filename, b"wheel")

    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(tmp_path / "wheels"),
            "--output",
            str(output),
        ],
        check=True,
    )

    assert (output / "simple/philtorch" / filename).read_bytes() == b"wheel"
