import os
from pathlib import Path
from runpy import run_path

ROOT = Path(__file__).resolve().parents[1]
build_support = run_path(str(ROOT / "build_support.py"))
metal_header = build_support["metal_header"]
write_metal_headers = build_support["write_metal_headers"]


def test_header_holds_the_source_as_a_raw_string(tmp_path):
    source = tmp_path / "scalar_recur_mps.metal"
    text = '#include <metal_stdlib>\nkernel void f() { /* ")" */ }\n'
    source.write_text(text)
    header = metal_header(str(source))
    assert 'constexpr const char *SCALAR_RECUR_MPS_METAL = R"METAL(' + text + ')METAL";' in header
    assert "#pragma once" in header


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
