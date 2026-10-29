"""Open an issue on how a new PyTorch release affects PhilTorch.

Finds the newest published minor release of pytorch/pytorch (``vX.Y.0``) and,
unless an issue titled ``PyTorch X.Y release impact`` already exists (open or
closed), opens one: a link to the release notes, then a brief summary by a
language model, labelled as such, of what changed upstream, how it affects
PhilTorch and what to do now. It links few pull requests, as each one linked
in an issue gets an event on its timeline.

The release notes of a minor release run to 100 KB or more, so they are
filtered first: the highlights, backwards-incompatible changes and deprecations
stay, and so do the entries that mention something PhilTorch depends on (CUDA,
``torch.compile``, ``torch.library``, ...). Sections on distributed training,
ONNX, XPU and the like are dropped.

The script runs in two steps around a model call it does not make itself:
``--prepare DIR`` selects the release, skips it if the issue exists, and writes
``DIR/prompt.md`` (instructions, a brief of PhilTorch from AGENTS.md, the
filtered notes) and ``DIR/meta.json``; nothing is written when there is nothing
to do. The workflow runs GitHub Copilot CLI on ``prompt.md`` with no tools.
``--post DIR`` then opens the issue with the reply (``--summary-file``), or
says there is none.

Standard library only.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = "pytorch/pytorch"
LABEL = "build"
MAX_NOTES_CHARS = 30_000
MAX_ENTRY_CHARS = 500
MAX_SUMMARY_CHARS = 20_000

# Sections kept whole, bar the dropped areas; elsewhere an entry needs a keyword.
KEEP_SECTIONS = ("highlights", "backwards incompatible changes", "deprecations")
DROP_AREAS = re.compile(
    r"distributed|c10d|fsdp|dtensor|devicemesh|ddp|pipelining|elastic|rpc|onnx|xpu|rocm"
    r"|quantiz|optim|mobile|caffe2|sparse|jit",
    re.IGNORECASE,
)
KEYWORDS = re.compile(
    r"python 3\.\d+|free-thread|nogil|cpython"
    r"|cuda|cublas|cub\b|cufft|cudnn|nvcc|nvrtc|blackwell|hopper|\bsm_?\d+"
    r"|\bmps\b|metal|macos|apple"
    r"|triton"
    r"|torch\.compile|dynamo|inductor|higher-order|\bhops?\b|functionaliz|fake tensor"
    r"|custom_op|custom op|torch\.library|register_autograd|register_fake|opcheck"
    r"|torch\.ops|dispatcher|autograd\.function|\bjvp\b|vmap|torch\.func|forward-mode"
    r"|cpp_extension|c\+\+|\babi\b|openmp|parallel_for|libtorch|\baten\b|opmath"
    r"|complex|\bwheels?\b|manylinux|pypi|cmake|release engineering"
    r"|cumsum|cumprod|associative_scan|float16|bfloat16",
    re.IGNORECASE,
)
# Sections cut first when the notes are too long.
LOW_PRIORITY = ("bug fixes", "documentation", "developers", "security")
# Published notes cite pull requests as bare "#123", which GitHub would link to
# this repository's issue 123 in our issue.
BARE_REF = re.compile(r"(?<![\w\[/&])#(\d+)\b")
MENTION = re.compile(r"(?<![\w`])@(?=\w)")
HTML_TAG = re.compile(r"<[^>]+>")


@dataclass
class Entry:
    section: str
    area: str
    text: str  # the entry's lines, first line a bullet or a ### heading

    @property
    def where(self) -> str:
        return f"{self.section} / {self.area}" if self.area else self.section


@dataclass
class Filtered:
    text: str
    used: int = 0  # entries in the text, highlights apart
    total: int = 0  # entries kept before the cap, highlights apart


def parse_entries(notes: str) -> list[Entry]:
    """Split release notes into bullets and ``###`` entries, tracking sections."""
    entries: list[Entry] = []
    section = area = ""
    block: list[str] = []
    fenced = False

    def flush() -> None:
        while block and not block[-1].strip():
            block.pop()
        if block and section:
            entries.append(Entry(section, area, "\n".join(block)))
        block.clear()

    for line in notes.replace("\r\n", "\n").split("\n"):
        if line.startswith("```"):
            fenced = not fenced
        heading = None if fenced else re.match(r"(#{1,3}) +(.*)", line)
        if heading:
            flush()
            level = len(heading.group(1))
            if level == 1:
                section, area = heading.group(2).strip(), ""
            elif level == 2:
                area = heading.group(2).strip()
            else:
                block.append(line)
        elif not fenced and line.startswith("- ") and not (block and block[0].startswith("###")):
            flush()  # A bullet starts an entry, except inside a ### entry's body.
            block.append(line)
        elif block or section.lower() == "highlights":
            block.append(line)
    flush()
    return entries


def link_refs(text: str) -> str:
    """Link bare ``#123`` to the pull request, not to an issue of this repository."""
    return BARE_REF.sub(rf"[#\1](https://github.com/{UPSTREAM}/pull/\1)", text)


def filter_notes(notes: str) -> Filtered:
    """Keep the entries PhilTorch cares about, up to ``MAX_NOTES_CHARS``."""
    kept = []
    for entry in parse_entries(link_refs(notes)):
        if DROP_AREAS.search(entry.area):
            continue
        if entry.section.lower() in KEEP_SECTIONS or KEYWORDS.search(entry.text):
            kept.append(entry)
    kept.sort(key=lambda e: e.section.lower() in LOW_PRIORITY)  # stable
    parts: list[str] = []
    size = used = 0
    last = None
    for entry in kept:
        highlight = entry.section.lower() == "highlights"
        text = HTML_TAG.sub("", entry.text) if highlight else entry.text
        if not highlight and len(text) > MAX_ENTRY_CHARS:
            text = text[:MAX_ENTRY_CHARS]
            if "```" in text:  # no code fence stays open
                text = text[: text.index("```")]
            text = text.rstrip() + " [...]"
        head = ""
        if (entry.section, entry.area) != last:
            head = f"\n# {entry.where}\n\n"
            last = (entry.section, entry.area)
        if size + len(head) + len(text) + 1 > MAX_NOTES_CHARS:
            break
        parts.append(head + text)
        size += len(head) + len(text) + 1
        used += not highlight
    total = sum(e.section.lower() != "highlights" for e in kept)
    return Filtered("\n".join(parts), used, total)


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
    for page in range(1, 6):
        path = f"/repos/{repo}/issues?labels={LABEL}&state=all&per_page=100&page={page}"
        issues = api(path, token)
        if any(i["title"] == title for i in issues):
            return True
        if len(issues) < 100:
            break
    return False


def project_brief() -> str:
    """The Architecture and Native ops sections of AGENTS.md."""
    text = (REPO_ROOT / "AGENTS.md").read_text()
    parts = re.findall(r"^## (?:Architecture|Native ops)\n.*?(?=^## )", text, re.S | re.M)
    if len(parts) != 2:
        raise SystemExit("AGENTS.md: no Architecture and Native ops sections")
    return "PhilTorch is a PyTorch library of native ops and kernels. From its AGENTS.md:\n\n" + (
        "\n".join(parts)
    )


PROMPT = """\
Summarise these PyTorch release notes, briefly, for the maintainer of the \
library described below. The notes are filtered: entries irrelevant to it were \
dropped. Reply with Markdown only, in exactly these three sections, each a \
short list of one-line items:

**What changed upstream**
The few changes that matter to PhilTorch, at most 6.

**How it affects PhilTorch**
What they mean for its code, tests, wheels or users, at most 6.

**To do now**
Immediate actions, as a task list (`- [ ] ...`), at most 5.

Rules: about 15 lines in all. Link a pull request \
(https://github.com/pytorch/pytorch/pull/N, as written in the notes) only \
where it helps, at most 5 links in all. Use only the text below; never invent \
an entry, link or claim. Do not use tools or read files. Write "None." for an \
empty section.

{brief}

## Release notes (filtered)

{notes}
"""


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
    f = filter_notes(release.get("body") or "")
    meta = {"title": title, "tag": release["tag_name"], "url": release["html_url"]}
    out.mkdir(parents=True, exist_ok=True)
    (out / "prompt.md").write_text(PROMPT.format(brief=project_brief(), notes=f.text))
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Wrote {out / 'prompt.md'} for {title!r}: {f.used} of {f.total} entries.")
    return 0


def sanitize(reply: str) -> str:
    """Make a model's reply safe to post: PR links, no mentions, capped."""
    reply = MENTION.sub("@\u200b", link_refs(reply.strip()))
    if len(reply) > MAX_SUMMARY_CHARS:
        reply = reply[:MAX_SUMMARY_CHARS].rsplit("\n", 1)[0]
        reply += f"\n\n*(The summary was cut at {MAX_SUMMARY_CHARS:,} characters.)*"
    return reply


def build_body(meta: dict, summary: str | None) -> str:
    """The issue body. ``summary`` is Copilot's reply, or None."""
    notes = f"[PyTorch {meta['tag'].removeprefix('v')} release notes]({meta['url']})"
    if summary:
        return f"{notes}. Summary by GitHub Copilot: verify before acting.\n\n{sanitize(summary)}"
    env = os.environ
    run = "the workflow run"
    if "GITHUB_RUN_ID" in env:
        run = (
            f"[the workflow run]({env['GITHUB_SERVER_URL']}/{env['GITHUB_REPOSITORY']}"
            f"/actions/runs/{env['GITHUB_RUN_ID']})"
        )
    return f"{notes}. No summary: Copilot isn't set up, or its step failed; see {run}."


def post(out: Path, summary_file: str | None, repo: str | None, token: str | None, dry_run: bool):
    """Open the issue described by ``out/meta.json``."""
    meta = json.loads((out / "meta.json").read_text())
    summary = None
    if summary_file and Path(summary_file).is_file():
        summary = Path(summary_file).read_text().strip() or None
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
    p.add_argument("--summary-file", help="--post: Copilot's reply, the issue's AI section")
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
