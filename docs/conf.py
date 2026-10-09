"""Sphinx configuration: API reference (autodoc) and the tutorial notebook (myst-nb).

Build: pip install ".[docs]"; sphinx-build -W --keep-going docs docs/_build/html
"""

import pysurfaceevolver

project = "pySurfaceEvolver"
author = "Gaute Linga"
release = pysurfaceevolver.__version__
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "myst_nb",
]
exclude_patterns = ["_build", "PLAN.md", "**.ipynb_checkpoints"]

autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_default_options = {"members": True, "undoc-members": False}
napoleon_numpy_docstring = True
napoleon_google_docstring = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}

# the tutorial runs at build time; PyVista draws static images
nb_execution_mode = "force"
nb_execution_timeout = 600
nb_execution_raise_on_error = True
myst_enable_extensions = ["colon_fence", "dollarmath"]

html_theme = "furo"
html_static_path = ["_static"]     # the drainage movies and stored runs
html_title = f"pySurfaceEvolver {release}"
