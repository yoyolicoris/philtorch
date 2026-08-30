import hashlib
import importlib.util
import re
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
    assert 'href="philtorch/"' in (output / "simple/index.html").read_text()

    INDEX.generate_index("philtorch", [first, second], output)
    assert (package_dir / "index.html").read_text() == page


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
