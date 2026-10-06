"""Sphinx configuration for the PhilTorch documentation.

The docs import the real package, so philtorch must be installed with its
compiled extension: an editable install builds it in place, which also makes
``import philtorch`` work from the repository root (see .readthedocs.yaml).
"""

from sphinx.ext.intersphinx import missing_reference

import philtorch

project = "PhilTorch"
author = "Chin-Yun Yu"
copyright = "2025-2026, Chin-Yun Yu"
release = philtorch.__version__
version = ".".join(release.split(".")[:2])

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

exclude_patterns = ["_build"]

# -- API reference -----------------------------------------------------------

autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_typehints = "none"  # the docstrings give every argument's type

napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_preprocess_types = True
napoleon_type_aliases = {
    "Tensor": "torch.Tensor",
    "Callable": "collections.abc.Callable",
    "Sequence": "collections.abc.Sequence",
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://docs.pytorch.org/docs/stable", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
}

# Every reference must resolve, except :attr:, which the docstrings use for
# argument names, as PyTorch's do.
nitpicky = True
nitpick_ignore_regex = [("py:attr", ".*")]
nitpick_ignore = [
    # Raised by torch.linalg, but missing from PyTorch's intersphinx inventory.
    ("py:exc", "torch.linalg.LinAlgError"),
]

# PyTorch's pages create their anchors with JavaScript, which linkcheck
# doesn't run, so check only that those pages exist.
linkcheck_anchors_ignore_for_url = [r"https://docs\.pytorch\.org/.*"]

# -- Markdown pages ----------------------------------------------------------

myst_heading_anchors = 3

# -- HTML --------------------------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_title = "PhilTorch"
html_theme_options = {
    "navigation_with_keys": False,
    "show_toc_level": 2,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/yoyolicoris/philtorch",
            "icon": "fa-brands fa-github",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/philtorch",
            "icon": "fa-brands fa-python",
        },
    ],
    "footer_start": ["copyright"],
    "footer_center": ["sphinx-version"],
    "footer_end": ["theme-version"],
}
html_show_sourcelink = False


# -- Short type names --------------------------------------------------------

# The docstrings name types as PyTorch's do, e.g. "Tensor" or "Sequence[int]".
# napoleon_type_aliases only covers argument types, so resolve the rest, such
# as return types, through intersphinx too.
_TYPE_ALIASES = {"Tensor": "torch.Tensor", "Callable": "collections.abc.Callable"}


def _resolve_type_alias(app, env, node, contnode):
    if node.get("refdomain") != "py":
        return None
    target = node.get("reftarget", "")
    alias = _TYPE_ALIASES.get(target)
    if alias is None and target.startswith("Sequence["):
        alias = "collections.abc.Sequence"
    if alias is None:
        return None
    node["reftarget"] = alias
    return missing_reference(app, env, node, contnode)


# -- README sections ---------------------------------------------------------

# A page built from a README section gives it the page's own level-1 title in
# place of the section's level-2 heading, so lift the section's subsections
# from level 3 to level 2. Lines in fenced code blocks are left alone.


def _lift_readme_headings(app, relative_path, parent_docname, content):
    if relative_path.name != "README.md":
        return
    lines, fenced = [], False
    for line in content[0].splitlines(keepends=True):
        if line.startswith("```"):
            fenced = not fenced
        elif not fenced and line.startswith("##"):
            line = line[1:]
        lines.append(line)
    content[0] = "".join(lines)


def setup(app):
    app.connect("missing-reference", _resolve_type_alias)
    app.connect("include-read", _lift_readme_headings)
