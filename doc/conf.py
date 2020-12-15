# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('../examples'))

# -- Project information -----------------------------------------------------

project = 'TensorLy-Torch'
from datetime import datetime
year = datetime.now().year
copyright = f'{year}, Jean Kossaifi'
author = 'Jean Kossaifi'

# The full version, including alpha/beta/rc tags
import tltorch
release = tltorch.__version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    "myst_nb",
    # 'jupyter_sphinx',
    # 'matplotlib.sphinxext.plot_directive',
    'sphinx.ext.imgmath', #'sphinx.ext.mathjax',
    'numpydoc.numpydoc',
]

html4_writer=True

# Remove the permalinks ("¶" symbols)
html_add_permalinks = ""

# NumPy 
numpydoc_class_members_toctree = False
numpydoc_show_class_members = True
numpydoc_show_inherited_class_members = False

# generate autosummary even if no references
autosummary_generate = True
autodoc_member_order = 'bysource'
autodoc_default_flags = ['members']

# Napoleon
napoleon_google_docstring = False
napoleon_use_rtype = False

# imgmath/mathjax
imgmath_image_format = 'svg'

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
# source_suffix = ['.rst', '.md']
source_suffix = '.rst'


# The master toctree document.
master_doc = 'index'

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


primary_domain = 'py'

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme_path = ['themes']
html_theme = 'tensorly'
html_logo = '_static/logos/tensorly-torch-logo.png'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

html_theme_options = {
    'github_url': 'https://github.com/tensorly/tensorly-torch'
}

# -- Options for LaTeX output ---------------------------------------------

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, 'tensorly.tex', 'Tensor operations in Python',
     'Jean Kossaifi', 'manual'),
]

latex_preamble = r"""
\usepackage{amsmath}\usepackage{amsfonts}
\setcounter{MaxMatrixCols}{20}
"""

imgmath_latex_preamble = latex_preamble

latex_elements = {
    'classoptions': ',oneside',
    'babel': '\\usepackage[english]{babel}',
    # Get completely rid of index
    'printindex': '',
    'preamble': latex_preamble,
}
