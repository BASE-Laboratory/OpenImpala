# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

from __future__ import annotations

import os
from pathlib import Path
import shutil
import sys

# -- Path setup ---------------------------------------------------------------
_DOCS_DIR = Path(__file__).resolve().parent
_PYTHON_DIR = str((_DOCS_DIR.parent / "python").resolve())
sys.path.insert(0, _PYTHON_DIR)
os.environ["PYTHONPATH"] = (
    f"{_PYTHON_DIR}{os.pathsep}{os.environ['PYTHONPATH']}"
    if os.environ.get("PYTHONPATH")
    else _PYTHON_DIR
)

# -- Project information ------------------------------------------------------
project = "OpenImpala"
copyright = "2024-2026, BASE Laboratory, University of Greenwich"
author = "James Le Houx"

try:
    from openimpala import __version__ as _oi_version

    release = _oi_version if _oi_version != "unknown" else "4.7.0"
except Exception:
    release = "4.7.0"
version = ".".join(release.split(".")[:2])

_DOXYGEN_XML_DIR = (_DOCS_DIR / "doxygen" / "xml").resolve()
_HAS_DOXYGEN_XML = (_DOXYGEN_XML_DIR / "index.xml").is_file()
if os.environ.get("OPENIMPALA_REQUIRE_DOXYGEN") == "1" and not _HAS_DOXYGEN_XML:
    raise RuntimeError(
        f"OPENIMPALA_REQUIRE_DOXYGEN=1, but Doxygen XML was not found at {_DOXYGEN_XML_DIR}"
    )

# -- General configuration ----------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "myst_nb",
    "sphinx_design",
    "sphinx_copybutton",
]
if _HAS_DOXYGEN_XML:
    extensions.append("breathe")

templates_path = ["_templates"] if (_DOCS_DIR / "_templates").is_dir() else []
exclude_patterns = [
    "_build",
    "doxygen",
    "jupyter_execute",
    "mainpage.md",
    "Thumbs.db",
    ".DS_Store",
]

# -- MyST & MyST-NB settings --------------------------------------------------
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
]

nb_execution_mode = "cache"
nb_execution_timeout = 180
nb_execution_raise_on_error = True
sd_card_legacy_separators = False

# -- Copybutton settings ------------------------------------------------------
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True

# -- Breathe (Doxygen bridge) -------------------------------------------------
breathe_projects = {"OpenImpala": str(_DOXYGEN_XML_DIR)}
breathe_default_project = "OpenImpala"

# -- Autodoc settings ---------------------------------------------------------
autodoc_mock_imports = ["openimpala._core", "mpi4py"]
autodoc_member_order = "bysource"
autodoc_typehints = "description"

# -- Napoleon (NumPy docstrings) ----------------------------------------------
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_rtype = False

# -- Intersphinx --------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

# -- HTML output (PyData Sphinx Theme) ----------------------------------------
html_theme = "pydata_sphinx_theme"
html_title = f"OpenImpala {version}"
html_static_path = ["_static"] if (_DOCS_DIR / "_static").is_dir() else []

html_theme_options = {
    "github_url": "https://github.com/BASE-Laboratory/OpenImpala",
    "icon_links": [
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/openimpala/",
            "icon": "fa-brands fa-python",
        },
    ],
    "navbar_align": "left",
    "header_links_before_dropdown": 6,
    "show_toc_level": 2,
    "navigation_with_keys": False,
}

# -- Autosummary --------------------------------------------------------------
autosummary_generate = True


def _sync_tutorials(_app) -> None:
    """Copy canonical tutorials/*.ipynb into docs/tutorials/ before Sphinx reads sources."""
    src_dir = _DOCS_DIR.parent / "tutorials"
    dst_dir = _DOCS_DIR / "tutorials"
    dst_dir.mkdir(parents=True, exist_ok=True)
    for old_nb in dst_dir.glob("*.ipynb"):
        old_nb.unlink()
    for nb_path in sorted(src_dir.glob("0*.ipynb")):
        shutil.copy2(nb_path, dst_dir / nb_path.name)


def setup(app) -> None:
    app.connect("builder-inited", _sync_tutorials)
    if not _HAS_DOXYGEN_XML:
        from docutils import nodes
        from docutils.parsers.rst import Directive, directives

        class _DoxygenPlaceholder(Directive):
            required_arguments = 1
            optional_arguments = 0
            final_argument_whitespace = True
            has_content = True
            option_spec = {
                "members": directives.flag,
                "outline": directives.flag,
                "project": directives.unchanged,
            }

            def run(self):
                symbol = self.arguments[0]
                para = nodes.paragraph()
                para += nodes.literal(text=symbol)
                para += nodes.Text(
                    " — run 'doxygen Doxyfile' before 'sphinx-build' to render full C++ signatures."
                )
                return [para]

        app.add_directive("doxygenclass", _DoxygenPlaceholder)
        app.add_directive("doxygenstruct", _DoxygenPlaceholder)
