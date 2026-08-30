#!/usr/bin/env python3
"""Generate a deterministic PEP 503 index from local PhilTorch wheels."""

from __future__ import annotations

import argparse
import hashlib
import html
import re
import shutil
from pathlib import Path
from typing import Iterable
from urllib.parse import quote

_NORMALIZE_PATTERN = re.compile(r"[-_.]+")


def normalize_project_name(name: str) -> str:
    """Return the PEP 503 normalized form of a project name."""
    return _NORMALIZE_PATTERN.sub("-", name).lower()


def file_sha256(path: Path) -> str:
    """Calculate the SHA-256 digest for a file without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as wheel_file:
        for chunk in iter(lambda: wheel_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def wheel_project_name(filename: str) -> str:
    """Read the escaped distribution component from a wheel filename."""
    if not filename.endswith(".whl") or "-" not in filename:
        raise ValueError(f"not a wheel filename: {filename}")
    return normalize_project_name(filename.split("-", 1)[0])


def _add_wheel(wheels: dict[str, tuple[Path, str]], path: Path, project: str) -> None:
    path = path.resolve()
    if not path.is_file():
        raise ValueError(f"wheel path is not a file: {path}")
    if wheel_project_name(path.name) != project:
        raise ValueError(f"wheel does not belong to {project}: {path.name}")

    digest = file_sha256(path)
    previous = wheels.get(path.name)
    if previous is not None and previous[1] != digest:
        raise ValueError(f"conflicting duplicate wheel filename: {path.name}")
    wheels[path.name] = (path, digest)


def _write_if_changed(path: Path, content: str) -> None:
    if path.exists() and path.read_text(encoding="utf-8") == content:
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _project_page(project: str, wheels: dict[str, tuple[Path, str]]) -> str:
    links = "\n".join(
        f'    <a href="{quote(filename)}#sha256={digest}">{html.escape(filename)}</a><br>'
        for filename, (_, digest) in sorted(wheels.items())
    )
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n'
        "  <head>\n"
        '    <meta name="pypi:repository-version" content="1.0">\n'
        f"    <title>Links for {html.escape(project)}</title>\n"
        "  </head>\n"
        "  <body>\n"
        f"    <h1>Links for {html.escape(project)}</h1>\n"
        f"{links}\n"
        "  </body>\n"
        "</html>\n"
    )


def _root_page(project: str) -> str:
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n'
        "  <body>\n"
        f'    <a href="{quote(project)}/">{html.escape(project)}</a>\n'
        "  </body>\n"
        "</html>\n"
    )


def generate_index(project: str, wheel_paths: Iterable[Path], output: Path) -> None:
    """Copy wheels into ``output/simple`` and generate deterministic index pages."""
    project = normalize_project_name(project)
    package_dir = output.resolve() / "simple" / project
    wheels: dict[str, tuple[Path, str]] = {}

    for wheel_path in wheel_paths:
        _add_wheel(wheels, wheel_path, project)

    if package_dir.exists():
        for wheel_path in sorted(package_dir.glob("*.whl")):
            _add_wheel(wheels, wheel_path, project)

    package_dir.mkdir(parents=True, exist_ok=True)
    for filename, (source, digest) in sorted(wheels.items()):
        destination = package_dir / filename
        if destination.exists():
            if file_sha256(destination) != digest:
                raise ValueError(f"conflicting duplicate wheel filename: {filename}")
            continue
        shutil.copyfile(source, destination)

    _write_if_changed(package_dir / "index.html", _project_page(project, wheels))
    _write_if_changed(package_dir.parent / "index.html", _root_page(project))


def _expand_inputs(paths: Iterable[Path]) -> list[Path]:
    wheels: list[Path] = []
    for path in paths:
        if path.is_dir():
            wheels.extend(sorted(path.glob("*.whl")))
        else:
            wheels.append(path)
    return wheels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheels", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--project", default="philtorch")
    arguments = parser.parse_args()
    try:
        generate_index(
            arguments.project, _expand_inputs(arguments.wheels), arguments.output
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
