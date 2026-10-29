"""Tests for scripts/pytorch_release_impact.py: its deterministic parts, offline."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "pytorch_release_impact.py"
spec = importlib.util.spec_from_file_location("pytorch_release_impact", SCRIPT)
impact = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = impact  # dataclasses look their module up here
spec.loader.exec_module(impact)

NOTES = """\
# PyTorch 2.99.0 Release Notes

- [Highlights](#highlights)

# Highlights

<table><tr><td><strong>A highlight</strong></td></tr></table>

# Backwards Incompatible Changes

## Python Frontend

### `torch.frobnicate` is removed ([#101](https://github.com/pytorch/pytorch/pull/101))

```python
# 2.98
torch.frobnicate(x)
```

## Distributed FSDP2

### FSDP2 reduces in bf16 ([#102](https://github.com/pytorch/pytorch/pull/102))

# Improvements

## CUDA

- Use CUB for `cumsum` on CUDA ([#201](https://github.com/pytorch/pytorch/pull/201))
- Use cuBLAS for a bare reference (#204)
- Rename a private helper ([#202](https://github.com/pytorch/pytorch/pull/202))

## DTensor

- Faster CUDA all-gather in DTensor ([#203](https://github.com/pytorch/pytorch/pull/203))
"""


def release(tag, **kw):
    return {"tag_name": tag, "html_url": f"https://example.org/{tag}", **kw}


def test_newest_minor_release_skips_patches_drafts_and_prereleases():
    releases = [
        release("v2.99.1"),  # a patch release
        release("v2.99.0", draft=True),
        release("v2.99.0-rc1", prerelease=True),
        release("v2.98.0"),
        release("v2.100.0"),  # newer than 2.99 by number, not by string
        release("v2.97.0"),
    ]
    assert impact.newest_minor_release(releases)["tag_name"] == "v2.100.0"
    assert impact.newest_minor_release([release("v1.2.1")]) is None
    assert impact.issue_title("v2.99.0") == "PyTorch 2.99 release impact"


@pytest.mark.parametrize("exists", [True, False])
def test_prepare_skips_an_existing_issue(monkeypatch, tmp_path, exists):
    monkeypatch.setattr(impact, "fetch_releases", lambda token: [release("v2.99.0", body=NOTES)])
    monkeypatch.setattr(impact, "issue_exists", lambda repo, title, token: exists)
    out = tmp_path / "out"
    assert impact.main(["--prepare", str(out), "--repo", "o/r"]) == 0
    if exists:
        assert not out.exists()
    else:
        prompt = (out / "prompt.md").read_text()
        assert "Use only the text below" in prompt and "Use CUB for `cumsum`" in prompt
        assert "Native ops" in prompt  # the brief, from AGENTS.md
        meta = json.loads((out / "meta.json").read_text())
        assert meta["title"] == "PyTorch 2.99 release impact"
        assert meta["entries"][0]["links"] == [
            ["101", "https://github.com/pytorch/pytorch/pull/101"]
        ]


def test_filter_keeps_relevant_entries_and_drops_distributed_ones():
    f = impact.filter_notes(NOTES)
    assert "`torch.frobnicate` is removed" in f.text
    assert "# 2.98" in f.text  # a code comment is not a heading
    assert "A highlight" in f.text and "<strong>" not in f.text
    assert "Use CUB for `cumsum`" in f.text
    assert "[#204](https://github.com/pytorch/pytorch/pull/204)" in f.text  # a bare #204
    assert "Rename a private helper" not in f.text  # no keyword, not a kept section
    assert "FSDP2" not in f.text and "DTensor" not in f.text
    assert [e.area for e in f.entries] == ["Python Frontend", "CUDA", "CUDA"]  # no highlight
    assert f.total == 3 and not f.cut


def test_filter_caps_the_text_and_closes_cut_code_fences(monkeypatch):
    monkeypatch.setattr(impact, "MAX_NOTES_CHARS", 300)
    f = impact.filter_notes(NOTES)
    assert len(f.text) <= 300 and f.cut and len(f.entries) < f.total
    monkeypatch.setattr(impact, "MAX_ENTRY_CHARS", 100)
    monkeypatch.setattr(impact, "MAX_NOTES_CHARS", 30_000)
    long = "### Use `cuda`\n" + "text " * 18 + "\n```python\ncode\n```\n"
    cut = impact.filter_notes("# Deprecations\n\n## A\n\n" + long).text
    assert "```" not in cut and cut.endswith("[...]")


def test_issue_body_labels_the_ai_section_and_sanitizes_it(monkeypatch):
    monkeypatch.delenv("GITHUB_RUN_ID", raising=False)
    meta = {
        "title": "PyTorch 2.99 release impact",
        "url": "https://example.org/v2.99.0",
        "published": "2026-01-02",
        "total": 3,
        "cut": True,
        "entries": [{"where": "CUDA", "summary": "S", "links": [["201", "https://x/pull/201"]]}],
    }
    reply = "Ping @someone about #123, mail a@b.c.\n" + "x\n" * 20_000
    body = impact.build_body(meta, reply)
    heading = "## AI-generated summary (GitHub Copilot): verify before acting"
    assert body.index(heading) < body.index("## Matched")
    assert "https://example.org/v2.99.0 (published 2026-01-02)" in body
    assert "1 of 3 entries were used" in body
    assert "@\u200bsomeone" in body and "a@b.c" in body
    assert "[#123](https://github.com/pytorch/pytorch/pull/123)" in body
    assert "was cut at 20,000" in body and "- CUDA: S ([#201](https://x/pull/201))" in body

    monkeypatch.setenv("GITHUB_RUN_ID", "9")
    monkeypatch.setenv("GITHUB_SERVER_URL", "https://github.com")
    monkeypatch.setenv("GITHUB_REPOSITORY", "o/r")
    plain = impact.build_body(meta, None)
    assert "AI-generated" not in plain
    assert "[workflow run](https://github.com/o/r/actions/runs/9)" in plain


@pytest.mark.parametrize("reply", ["A reply.", "  \n", None])
def test_post_dry_run_uses_a_non_empty_summary_file(tmp_path, capsys, reply):
    meta = {"title": "T", "url": "u", "published": "", "total": 0, "cut": False, "entries": []}
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    if reply is not None:
        (tmp_path / "summary.md").write_text(reply)
    args = ["--post", str(tmp_path), "--dry-run", "--summary-file", str(tmp_path / "summary.md")]
    assert impact.main(args) == 0
    out = capsys.readouterr().out
    assert out.startswith("Title: T")
    assert ("A reply." in out) == (reply == "A reply.")
    assert ("No AI summary" in out) == (reply != "A reply.")
