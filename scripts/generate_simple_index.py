#!/usr/bin/env python3
"""Generate PhilTorch's CUDA wheel package indexes from GitHub Release assets.

The CUDA wheels are stored as release assets; this script writes the static
PEP 503 pages that link to them, for GitHub Pages to serve. Each CUDA major
gets its own index, so ``pip install philtorch --index-url
<base-url>/whl/cu13/`` can only pick a build for CUDA 13::

    <output>/index.html                         landing page
    <output>/whl/cu13/index.html                index root
    <output>/whl/cu13/philtorch/index.html      links to the wheels

Wheels of earlier releases, labelled by PyTorch minor and CUDA version, stay
in their own indexes, e.g. ``whl/torch2.14-cu130/``.

The input is the repository's release list from the GitHub API::

    gh api --paginate --slurp repos/OWNER/REPO/releases > releases.json

The site is rebuilt from every published release on each run, so wheels from
older releases stay listed for as long as their release assets exist.
"""

from __future__ import annotations

import argparse
import html
import json
import re
from collections.abc import Iterable
from pathlib import Path

_NORMALIZE_PATTERN = re.compile(r"[-_.]+")
PROJECT = "philtorch"
# The local label scripts/torch_matrix.py gives CUDA wheels, e.g. "cu13".
# Earlier releases used "torch2.14.1.cu130", whose index is per torch minor,
# which those wheels pin.
_CUDA_LABEL = re.compile(r"cu(\d+)")
_OLD_CUDA_LABEL = re.compile(r"torch(\d+)\.(\d+)\.\d+\.cu(\d+)")
_INDEX_NAME = re.compile(r"cu(\d+)|torch(\d+)\.(\d+)-cu(\d+)")
# GitHub reports a digest for every release asset, so the pages can carry
# hashes without downloading the wheels.
_ASSET_DIGEST = re.compile(r"sha256:([0-9a-f]{64})")


def normalize_project_name(name: str) -> str:
    """Return the PEP 503 normalized form of a project name."""
    return _NORMALIZE_PATTERN.sub("-", name).lower()


def wheel_project_name(filename: str) -> str:
    """Read the escaped distribution component from a wheel filename."""
    # {name}-{version}(-{build})?-{python}-{abi}-{platform}.whl; a build tag
    # starts with a digit.
    parts = filename.removesuffix(".whl").split("-")
    if (
        not filename.endswith(".whl")
        or len(parts) not in (5, 6)
        or not all(parts)
        or (len(parts) == 6 and not parts[2][0].isdigit())
    ):
        raise ValueError(f"not a wheel filename: {filename}")
    return normalize_project_name(parts[0])


def index_name(filename: str) -> str | None:
    """Return the index a CUDA wheel belongs in, e.g. ``cu13`` or ``torch2.14-cu130``.

    Returns None for a wheel without a CUDA build label, such as a CPU wheel.
    """
    version = filename.removesuffix(".whl").split("-")[1]
    _, _, label = version.partition("+")
    if _CUDA_LABEL.fullmatch(label):
        return label
    match = _OLD_CUDA_LABEL.fullmatch(label)
    if match is None:
        return None
    major, minor, cuda = match.groups()
    return f"torch{major}.{minor}-cu{cuda}"


def release_wheels(releases: Iterable[dict]) -> dict[str, dict[str, tuple[str, str]]]:
    """Map each index name to ``{filename: (url, sha256)}`` for the CUDA wheels."""
    indexes: dict[str, dict[str, tuple[str, str]]] = {}
    for release in releases:
        # A draft's assets are not public, so pip could not download them.
        if release.get("draft"):
            continue
        for asset in release.get("assets", []):
            filename = asset["name"]
            if not filename.endswith(".whl"):
                continue
            if normalize_project_name(filename.split("-", 1)[0]) != PROJECT:
                continue
            wheel_project_name(filename)  # rejects a malformed filename
            index = index_name(filename)
            if index is None:
                continue
            digest = _ASSET_DIGEST.fullmatch(asset.get("digest") or "")
            if digest is None:
                raise ValueError(
                    f"release {release.get('tag_name')}: asset {filename} has no SHA-256 digest"
                )
            entry = (asset["browser_download_url"], digest[1])
            wheels = indexes.setdefault(index, {})
            previous = wheels.get(filename)
            if previous is not None and previous[1] != entry[1]:
                raise ValueError(f"conflicting duplicate wheel filename: {filename}")
            # Identical copies in two releases: keep one, whatever the release order.
            wheels[filename] = min(previous, entry) if previous else entry
    return indexes


def _sort_key(name: str) -> tuple[int, ...]:
    # The CUDA major indexes come before the per-torch-minor ones.
    parts = _INDEX_NAME.fullmatch(name).groups()
    return (1, int(parts[0])) if parts[0] else (0, *map(int, parts[1:]))


def _simple_page(title: str, body: str) -> str:
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n'
        "  <head>\n"
        '    <meta name="pypi:repository-version" content="1.0">\n'
        f"    <title>{html.escape(title)}</title>\n"
        "  </head>\n"
        "  <body>\n"
        f"{body}"
        "  </body>\n"
        "</html>\n"
    )


def _project_page(wheels: dict[str, tuple[str, str]]) -> str:
    links = "".join(
        f'    <a href="{html.escape(url)}#sha256={digest}">{html.escape(filename)}</a><br>\n'
        for filename, (url, digest) in sorted(wheels.items())
    )
    return _simple_page(f"Links for {PROJECT}", f"    <h1>Links for {PROJECT}</h1>\n{links}")


def _root_page() -> str:
    return _simple_page("Simple index", f'    <a href="{PROJECT}/">{PROJECT}</a>\n')


def _landing_page(base_url: str, names: list[str]) -> str:
    def items(group: list[str]) -> str:
        return "".join(
            f'      <li><a href="whl/{name}/">{name}</a>: '
            f"<code>pip install {PROJECT} --index-url {html.escape(base_url)}/whl/{name}/</code>"
            "</li>\n"
            for name in group
        )

    current = [name for name in names if _CUDA_LABEL.fullmatch(name)]
    earlier = [name for name in names if not _CUDA_LABEL.fullmatch(name)]
    if names:
        body = ""
        if current:
            body += (
                "    <p>One package index per CUDA major version. Install a PyTorch build\n"
                "    with that CUDA version first, then install from its index:</p>\n"
                f"    <ul>\n{items(current)}    </ul>\n"
            )
        if earlier:
            body += (
                "    <p>Earlier releases have one index per PyTorch minor and CUDA version.\n"
                "    Install the matching PyTorch build first:</p>\n"
                f"    <ul>\n{items(earlier)}    </ul>\n"
            )
    else:
        body = "    <p>No CUDA wheels have been published yet.</p>\n"
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n'
        "  <head>\n"
        '    <meta charset="utf-8">\n'
        f"    <title>{PROJECT} CUDA wheels</title>\n"
        "  </head>\n"
        "  <body>\n"
        f"    <h1>{PROJECT} CUDA wheels</h1>\n"
        f"{body}"
        "  </body>\n"
        "</html>\n"
    )


def build_site(releases: Iterable[dict], output: Path, base_url: str) -> dict[str, int]:
    """Write the site into an empty ``output`` and return each index's wheel count."""
    # Rebuilding into an old site would keep indexes whose releases are gone.
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"output directory is not empty: {output}")
    indexes = release_wheels(releases)
    names = sorted(indexes, key=_sort_key, reverse=True)

    for name in names:
        root = output / "whl" / name
        (root / PROJECT).mkdir(parents=True)
        (root / "index.html").write_text(_root_page(), encoding="utf-8")
        (root / PROJECT / "index.html").write_text(_project_page(indexes[name]), encoding="utf-8")
    output.mkdir(parents=True, exist_ok=True)
    (output / "index.html").write_text(_landing_page(base_url.rstrip("/"), names), encoding="utf-8")
    return {name: len(indexes[name]) for name in names}


def load_releases(path: Path) -> list[dict]:
    """Read a release list, flattening the pages of ``gh api --paginate --slurp``."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a JSON list of releases")
    if all(isinstance(page, list) for page in data):
        data = [release for page in data for release in page]
    return data


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("releases", type=Path, help="release list from the GitHub API")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--base-url", required=True, help="URL the site is served from, for the landing page"
    )
    arguments = parser.parse_args()
    try:
        counts = build_site(load_releases(arguments.releases), arguments.output, arguments.base_url)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    for name, count in counts.items():
        print(f"{name}: {count} wheels")
    if not counts:
        print("no CUDA wheels found in any published release")


if __name__ == "__main__":
    main()
