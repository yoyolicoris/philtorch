import os
from pathlib import Path
from runpy import run_path

import pytest

ROOT = Path(__file__).resolve().parents[1]
build_support = run_path(str(ROOT / "build_support.py"))
metal_header = build_support["metal_header"]
target_version_header = build_support["target_version_header"]
write_metal_headers = build_support["write_metal_headers"]
MIN_TORCH_MACOS = build_support["MIN_TORCH_MACOS"]


def test_header_holds_the_source_as_a_raw_string(tmp_path):
    source = tmp_path / "scalar_recur_mps.metal"
    text = '#include <metal_stdlib>\nkernel void f() { /* ")" */ }\n'
    source.write_text(text)
    header = metal_header(str(source))
    assert 'constexpr const char *SCALAR_RECUR_MPS_METAL = R"METAL(' + text + ')METAL";' in header
    assert "#pragma once" in header


def test_header_rejects_the_raw_string_delimiter(tmp_path):
    source = tmp_path / "bad.metal"
    source.write_text('const char *s = ")METAL";\n')
    with pytest.raises(RuntimeError, match="delimiter"):
        metal_header(str(source))


def test_write_metal_headers_writes_one_header_per_metal_file(tmp_path):
    src, include = tmp_path / "csrc", tmp_path / "build" / "include"
    src.mkdir()
    (src / "a.metal").write_text("// a\n")
    (src / "b_c.metal").write_text("// b\n")
    (src / "ignored.mm").write_text("// not a shader\n")
    (src / "ignored.metal.txt").write_text("// not a shader\n")
    write_metal_headers(str(src), str(include))  # include doesn't exist yet
    assert sorted(os.listdir(include)) == ["a.metal.h", "b_c.metal.h", "torch_target_version.h"]
    assert "A_METAL" in (include / "a.metal.h").read_text()
    assert "B_C_METAL" in (include / "b_c.metal.h").read_text()


def test_write_metal_headers_rewrites_only_changed_sources(tmp_path):
    src, include = tmp_path / "csrc", tmp_path / "include"
    src.mkdir()
    source = src / "a.metal"
    source.write_text("// one\n")
    write_metal_headers(str(src), str(include))
    header = include / "a.metal.h"
    old = 1_000_000_000
    os.utime(header, (old, old))
    write_metal_headers(str(src), str(include))
    assert header.stat().st_mtime == old  # unchanged, so not rebuilt

    source.write_text("// two\n")
    write_metal_headers(str(src), str(include))
    assert header.stat().st_mtime != old
    assert "// two\n" in header.read_text()


def test_target_version_header_packs_the_version_as_torch_does():
    # torch's TORCH_TARGET_VERSION is major << 56 | minor << 48.
    assert "#define TORCH_TARGET_VERSION (((0ULL + 2) << 56) | ((0ULL + 10) << 48))" in (
        target_version_header("2.10")
    )


def test_written_target_version_is_the_macos_floor(tmp_path):
    write_metal_headers(str(tmp_path), str(tmp_path / "include"))
    header = (tmp_path / "include" / "torch_target_version.h").read_text()
    assert header == target_version_header(MIN_TORCH_MACOS)
