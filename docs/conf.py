# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# -- Path setup ---------------------------------------------------------------
# Add the Python package to sys.path so autodoc can find it.
sys.path.insert(0, os.path.abspath(os.path.join("..", "python")))

# -- Project information ------------------------------------------------------
project = "OpenImpala"
copyright = "2024-2026, BASE Laboratory, University of Greenwich"
author = "James Le Houx"

# Version is read from pyproject.toml at build time; fallback for local builds.
try:
    from importlib.metadata import version as _version

    release = _version("openimpala")
except Exception:
    release = "4.0.0"
version = ".".join(release.split(".")[:2])

# -- General configuration ----------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "breathe",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "doxygen", "Thumbs.db", ".DS_Store"]

# -- MyST (Markdown) settings -------------------------------------------------
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
]
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# -- Breathe (Doxygen bridge) -------------------------------------------------
breathe_projects = {"OpenImpala": os.path.abspath("doxygen/xml")}
breathe_default_project = "OpenImpala"

# -- Autodoc settings ---------------------------------------------------------
autodoc_mock_imports = ["openimpala._core", "mpi4py"]
autodoc_member_order = "bysource"
autodoc_typehints = "description"

# -- Napoleon (Google/NumPy docstrings) ----------------------------------------
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_rtype = False

# -- Intersphinx (cross-project links) ----------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

# -- HTML output ---------------------------------------------------------------
html_theme = "furo"
html_title = "OpenImpala"
html_static_path = ["_static"]

html_theme_options = {
    "source_repository": "https://github.com/BASE-Laboratory/OpenImpala",
    "source_branch": "master",
    "source_directory": "docs/",
    "light_css_variables": {
        "color-brand-primary": "#2962FF",
        "color-brand-content": "#2962FF",
    },
}

# -- Autosummary ---------------------------------------------------------------
autosummary_generate = True
