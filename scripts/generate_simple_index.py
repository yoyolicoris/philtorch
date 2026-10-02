#!/usr/bin/env python3
"""Generate PhilTorch's CUDA wheel package indexes from GitHub Release assets.

The CUDA wheels are stored as release assets; this script writes the static
PEP 503 pages that link to them, for GitHub Pages to serve. Each PyTorch
minor/CUDA pair gets its own index, the way download.pytorch.org has one per
CUDA version. Every index then holds one build per PhilTorch version, so
``pip install philtorch --index-url <base-url>/whl/torch2.14-cu130/`` can
only pick a build for that torch and CUDA::

    <output>/index.html                                landing page
    <output>/whl/torch2.14-cu130/index.html            index root
    <output>/whl/torch2.14-cu130/philtorch/index.html  links to the wheels

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
_VALID_PROJECT_NAME = re.compile(r"^[a-z0-9]([a-z0-9-]*[a-z0-9])?$")
# The local label scripts/cuda_wheel_version.py gives CUDA wheels, e.g.
# "torch2.14.1.cu130". Indexes are per torch minor, which the wheels pin.
_CUDA_LABEL = re.compile(r"torch(\d+)\.(\d+)\.\d+\.cu(\d+)")
_INDEX_NAME = re.compile(r"torch(\d+)\.(\d+)-cu(\d+)")
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
    """Return the index a CUDA wheel belongs in, e.g. ``torch2.14-cu130``.

    Returns None for a wheel without a CUDA build label, such as a CPU wheel.
    """
    version = filename.removesuffix(".whl").split("-")[1]
    _, _, label = version.partition("+")
    match = _CUDA_LABEL.fullmatch(label)
    if match is None:
        return None
    major, minor, cuda = match.groups()
    return f"torch{major}.{minor}-cu{cuda}"


def release_wheels(releases: Iterable[dict], project: str) -> dict[str, dict[str, tuple[str, str]]]:
    """Map each index name to ``{filename: (url, sha256)}`` for the project's CUDA wheels."""
    indexes: dict[str, dict[str, tuple[str, str]]] = {}
    for release in releases:
        # A draft's assets are not public, so pip could not download them.
        if release.get("draft"):
            continue
        for asset in release.get("assets", []):
            filename = asset["name"]
            if not filename.endswith(".whl"):
                continue
            if normalize_project_name(filename.split("-", 1)[0]) != project:
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
            url = asset["browser_download_url"]
            if not url.startswith("https://"):
                raise ValueError(f"asset {filename} is not served over HTTPS: {url}")
            entry = (url, digest[1])
            wheels = indexes.setdefault(index, {})
            previous = wheels.get(filename)
            if previous is not None and previous[1] != entry[1]:
                raise ValueError(f"conflicting duplicate wheel filename: {filename}")
            # Identical copies in two releases: keep one, whatever the release order.
            wheels[filename] = min(previous, entry) if previous else entry
    return indexes


def _sort_key(name: str) -> tuple[int, ...]:
    return tuple(int(part) for part in _INDEX_NAME.fullmatch(name).groups())


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


def _project_page(project: str, wheels: dict[str, tuple[str, str]]) -> str:
    links = "".join(
        f'    <a href="{html.escape(url)}#sha256={digest}">{html.escape(filename)}</a><br>\n'
        for filename, (url, digest) in sorted(wheels.items())
    )
    return _simple_page(f"Links for {project}", f"    <h1>Links for {project}</h1>\n{links}")


def _root_page(project: str) -> str:
    return _simple_page("Simple index", f'    <a href="{project}/">{project}</a>\n')


def _landing_page(project: str, base_url: str, names: list[str]) -> str:
    if names:
        items = "".join(
            f'      <li><a href="whl/{name}/">{name}</a>: '
            f"<code>pip install {project} --index-url {html.escape(base_url)}/whl/{name}/</code>"
            "</li>\n"
            for name in names
        )
        body = (
            "    <p>One package index per PyTorch minor and CUDA version. Install the\n"
            "    matching PyTorch build first, then install from its index:</p>\n"
            f"    <ul>\n{items}    </ul>\n"
        )
    else:
        body = "    <p>No CUDA wheels have been published yet.</p>\n"
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n'
        "  <head>\n"
        '    <meta charset="utf-8">\n'
        f"    <title>{project} CUDA wheels</title>\n"
        "  </head>\n"
        "  <body>\n"
        f"    <h1>{project} CUDA wheels</h1>\n"
        f"{body}"
        "  </body>\n"
        "</html>\n"
    )


def build_site(
    releases: Iterable[dict], output: Path, base_url: str, project: str = "philtorch"
) -> dict[str, int]:
    """Write the site into an empty ``output`` and return each index's wheel count."""
    project = normalize_project_name(project)
    if not _VALID_PROJECT_NAME.fullmatch(project):
        raise ValueError(f"invalid project name: {project}")
    # Rebuilding into an old site would keep indexes whose releases are gone.
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"output directory is not empty: {output}")
    indexes = release_wheels(releases, project)
    names = sorted(indexes, key=_sort_key, reverse=True)

    for name in names:
        root = output / "whl" / name
        (root / project).mkdir(parents=True)
        (root / "index.html").write_text(_root_page(project), encoding="utf-8")
        (root / project / "index.html").write_text(
            _project_page(project, indexes[name]), encoding="utf-8"
        )
    output.mkdir(parents=True, exist_ok=True)
    (output / "index.html").write_text(
        _landing_page(project, base_url.rstrip("/"), names), encoding="utf-8"
    )
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
    parser.add_argument("--project", default="philtorch")
    arguments = parser.parse_args()
    try:
        counts = build_site(
            load_releases(arguments.releases),
            arguments.output,
            arguments.base_url,
            arguments.project,
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    for name, count in counts.items():
        print(f"{name}: {count} wheels")
    if not counts:
        print("no CUDA wheels found in any published release")


if __name__ == "__main__":
    main()
