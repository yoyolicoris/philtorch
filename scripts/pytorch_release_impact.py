"""Open an issue on how a new PyTorch release affects PhilTorch.

Finds the newest published minor release of pytorch/pytorch (``vX.Y.0``) and,
unless an issue titled ``PyTorch X.Y release impact`` already exists (open or
closed), opens one: a link to the release notes, then a brief summary by a
language model of what changed upstream, how it affects PhilTorch and what to
do now.

It runs in two steps around a model call it does not make itself:
``--prepare DIR`` writes ``DIR/prompt.md`` and ``DIR/meta.json``, or nothing
when there is nothing to do; the workflow runs GitHub Copilot CLI on the
prompt with no tools; ``--post DIR`` opens the issue with the reply
(``--summary-file``), or says there is none. Standard library only.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

UPSTREAM = "pytorch/pytorch"
LABEL = "build"
# The prompt goes to Copilot as one argument, which Linux caps at 128 KiB.
MAX_NOTES_BYTES = 100_000
# The model may cite pull requests as bare "#123", which GitHub would link to
# this repository's issue 123, and write "@name", which would notify a user.
# Code and Markdown links are left alone.
PROTECTED = re.compile(r"```.*?```|`[^`\n]*`|\[[^\]\n]*\]\([^)\s]*\)", re.S)
BARE_REF = re.compile(r"(?<![\w/&])#(\d+)\b")
MENTION = re.compile(r"(?<![\w/.])@(?=\w)")

PROMPT = """\
Summarise these PyTorch release notes, briefly, for the maintainer of \
PhilTorch: a PyTorch library of differentiable filters and recurrences, HMM \
and Kalman estimation and DTW/CTC alignment. Its native C++/CUDA ops are \
torch.library custom ops with registered autograd and torch.compile support; \
it has Triton kernels on CUDA and Metal kernels on MPS, and ships CPU wheels \
on PyPI and CUDA wheels for each PyTorch minor and CUDA version.

Reply with Markdown only, in exactly these three sections, each a short list \
of one-line items:

**What changed upstream**
The few changes that matter to PhilTorch, at most 6.

**How it affects PhilTorch**
What they mean for its code, tests, wheels or users, at most 6.

**To do now**
Immediate actions, as a task list (`- [ ] ...`), at most 5.

Rules: about 15 lines in all. Link a pull request \
(https://github.com/pytorch/pytorch/pull/N) only where it helps, at most 5 \
links in all. Use only the notes below; never invent an entry, link or claim. \
Do not use tools or read files. Write "None." for an empty section.

## Release notes

{notes}
"""


def release_version(tag: str) -> tuple[int, int] | None:
    """``(X, Y)`` of a minor release tag ``vX.Y.0``, else None."""
    m = re.fullmatch(r"v(\d+)\.(\d+)\.0", tag)
    return (int(m.group(1)), int(m.group(2))) if m else None


def newest_minor_release(releases: list[dict]) -> dict | None:
    """The newest published, non-prerelease ``vX.Y.0`` release."""
    minors = [
        r
        for r in releases
        if not r.get("draft") and not r.get("prerelease") and release_version(r["tag_name"])
    ]
    return max(minors, key=lambda r: release_version(r["tag_name"]), default=None)


def issue_title(tag: str) -> str:
    x, y = release_version(tag)
    return f"PyTorch {x}.{y} release impact"


def api(path: str, token: str | None, data: dict | None = None):
    """A GitHub REST call: GET, or POST with ``data``."""
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "philtorch-impact"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    body = json.dumps(data).encode() if data is not None else None
    req = urllib.request.Request(f"https://api.github.com{path}", body, headers)
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        raise SystemExit(f"GitHub API {path}: {e.code} {e.read().decode()[:500]}") from e


def fetch_releases(token: str | None) -> list[dict]:
    return api(f"/repos/{UPSTREAM}/releases?per_page=100", token)


def issue_exists(repo: str, title: str, token: str | None) -> bool:
    """Whether an issue of ours, open or closed, has exactly this title."""
    for page in range(1, 11):
        path = f"/repos/{repo}/issues?state=all&per_page=100&page={page}"
        issues = api(path, token)
        if any(i["title"] == title for i in issues):
            return True
        if len(issues) < 100:
            break
    return False


def prepare(out: Path, repo: str | None, token: str | None) -> int:
    """Select the release; write prompt.md and meta.json to ``out``, or nothing."""
    release = newest_minor_release(fetch_releases(token))
    if release is None:
        print("No published minor release found.")
        return 0
    title = issue_title(release["tag_name"])
    if repo and issue_exists(repo, title, token):
        print(f"Issue {title!r} already exists.")
        return 0
    notes = (release.get("body") or "").encode()[:MAX_NOTES_BYTES].decode(errors="ignore")
    meta = {"title": title, "tag": release["tag_name"], "url": release["html_url"]}
    out.mkdir(parents=True, exist_ok=True)
    (out / "prompt.md").write_text(PROMPT.format(notes=notes), encoding="utf-8")
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {out / 'prompt.md'} for {title!r}.")
    return 0


def sanitize(reply: str) -> str:
    """Make a model's reply safe to post: pull request links, no mentions."""

    def fix(text: str) -> str:
        text = BARE_REF.sub(rf"[#\1](https://github.com/{UPSTREAM}/pull/\1)", text)
        return MENTION.sub("@\u200b", text)

    reply, parts, end = reply.strip(), [], 0
    for m in PROTECTED.finditer(reply):
        parts += [fix(reply[end : m.start()]), m.group()]
        end = m.end()
    return "".join(parts + [fix(reply[end:])])


def build_body(meta: dict, summary: str | None) -> str:
    """The issue body. ``summary`` is Copilot's reply, or None."""
    notes = f"[PyTorch {meta['tag'].removeprefix('v')} release notes]({meta['url']})"
    if summary:
        return f"{notes}. Summary by GitHub Copilot: verify before acting.\n\n{sanitize(summary)}"
    return f"{notes}. No summary: the COPILOT_GITHUB_TOKEN secret isn't set."


def post(out: Path, summary_file: str | None, repo: str | None, token: str | None, dry_run: bool):
    """Open the issue described by ``out/meta.json``."""
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    summary = None
    if summary_file and Path(summary_file).is_file():
        summary = Path(summary_file).read_text(encoding="utf-8").strip() or None
    body = build_body(meta, summary)
    if dry_run:
        print(f"Title: {meta['title']}\n\n{body}")
        return 0
    data = {"title": meta["title"], "body": body, "labels": [LABEL]}
    print(f"Opened {api(f'/repos/{repo}/issues', token, data)['html_url']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", metavar="DIR", help="select the release, write the prompt")
    mode.add_argument("--post", metavar="DIR", help="open the issue from DIR/meta.json")
    p.add_argument("--summary-file", help="--post: Copilot's reply")
    p.add_argument("--dry-run", action="store_true", help="--post: print the issue, don't open it")
    p.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY"), help="issue repository")
    args = p.parse_args(argv)
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if args.prepare:
        return prepare(Path(args.prepare), args.repo, token)
    if not args.dry_run and not (args.repo and token):
        p.error("opening an issue needs --repo (GITHUB_REPOSITORY) and GH_TOKEN")
    return post(Path(args.post), args.summary_file, args.repo, token, args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
