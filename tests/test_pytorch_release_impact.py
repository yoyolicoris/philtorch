"""Tests for scripts/pytorch_release_impact.py: its deterministic parts, offline."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "pytorch_release_impact.py"
spec = importlib.util.spec_from_file_location("pytorch_release_impact", SCRIPT)
impact = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = impact
spec.loader.exec_module(impact)


def release(tag, **kw):
    return {"tag_name": tag, "html_url": f"https://example.org/{tag}", **kw}


def test_newest_minor_release_skips_patches_drafts_and_prereleases():
    releases = [
        release("v2.99.1"),  # a patch release
        release("v2.99.0", draft=True),
        release("v2.99.0-rc1", prerelease=True),
        release("v2.98.0"),
        release("v2.100.0"),  # newer than 2.99 by number, not by string
    ]
    assert impact.newest_minor_release(releases)["tag_name"] == "v2.100.0"
    assert impact.newest_minor_release([release("v1.2.1")]) is None
    assert impact.issue_title("v2.99.0") == "PyTorch 2.99 release impact"


@pytest.mark.parametrize("exists", [True, False])
def test_prepare_skips_an_existing_issue(monkeypatch, tmp_path, exists):
    notes = "- Use CUB for `cumsum` on CUDA (#201)\n" + "x" * impact.MAX_NOTES_CHARS
    monkeypatch.setattr(impact, "fetch_releases", lambda token: [release("v2.99.0", body=notes)])
    monkeypatch.setattr(impact, "issue_exists", lambda repo, title, token: exists)
    out = tmp_path / "out"
    assert impact.main(["--prepare", str(out), "--repo", "o/r"]) == 0
    if exists:
        assert not out.exists()
    else:
        prompt = (out / "prompt.md").read_text(encoding="utf-8")
        assert "Use CUB for `cumsum`" in prompt and "Use only the notes below" in prompt
        assert len(prompt) < len(impact.PROMPT) + impact.MAX_NOTES_CHARS  # the notes are cut
        assert json.loads((out / "meta.json").read_text(encoding="utf-8")) == {
            "title": "PyTorch 2.99 release impact",
            "tag": "v2.99.0",
            "url": "https://example.org/v2.99.0",
        }


def test_issue_body_links_the_notes_and_sanitizes_the_summary(monkeypatch):
    monkeypatch.delenv("GITHUB_RUN_ID", raising=False)
    meta = {"title": "T", "tag": "v2.99.0", "url": "https://example.org/v2.99.0"}
    body = impact.build_body(meta, "Ping @someone about #123, mail a@b.c.")
    assert body.startswith(
        "[PyTorch 2.99.0 release notes](https://example.org/v2.99.0). "
        "Summary by GitHub Copilot: verify before acting."
    )
    assert "@​someone" in body and "a@b.c" in body
    assert "[#123](https://github.com/pytorch/pytorch/pull/123)" in body

    monkeypatch.setenv("GITHUB_RUN_ID", "9")
    monkeypatch.setenv("GITHUB_SERVER_URL", "https://github.com")
    monkeypatch.setenv("GITHUB_REPOSITORY", "o/r")
    plain = impact.build_body(meta, None)
    assert "Copilot isn't set up" in plain
    assert "[the workflow run](https://github.com/o/r/actions/runs/9)" in plain


@pytest.mark.parametrize("reply", ["A reply.", "  \n", None])
def test_post_dry_run_uses_a_non_empty_summary_file(tmp_path, capsys, reply):
    meta = {"title": "T", "tag": "v2.99.0", "url": "u"}
    (tmp_path / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    if reply is not None:
        (tmp_path / "summary.md").write_text(reply, encoding="utf-8")
    args = ["--post", str(tmp_path), "--dry-run", "--summary-file", str(tmp_path / "summary.md")]
    assert impact.main(args) == 0
    out = capsys.readouterr().out
    assert out.startswith("Title: T")
    assert ("A reply." in out) == (reply == "A reply.")
    assert ("No summary" in out) == (reply != "A reply.")
