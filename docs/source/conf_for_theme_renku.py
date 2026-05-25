# Copyright 2026 Raul Mohedano and Erik Velasco-Salido (Vision Modeling Group, CSIC)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://apache.org
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'project-rf-comparison'
copyright = '2024, Raul Mohedano'
author = 'Raul Mohedano'
release = '0.5'


# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path info -----------------------------------------------------

import os
import sys
sys.path.insert(0, os.path.abspath('../../src/'))

# -- Modification for compatibility -----------------------------------------------------

from sphinx_math_dollar import NODE_BLACKLIST
from docutils.nodes import header

math_dollar_node_blacklist = NODE_BLACKLIST + (header,)




# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
	'sphinx.ext.autodoc',
	'sphinx.ext.autosummary',
	'sphinx.ext.napoleon',
	'sphinx.ext.intersphinx',
	'sphinx.ext.inheritance_diagram',
	'sphinx_math_dollar', 'sphinx.ext.mathjax',
]

intersphinx_mapping = {
	'python': ('https://docs.python.org/3', None),
	'torch': ('https://pytorch.org/docs/stable/', None),
	'tensordict': ('https://pytorch.github.io/tensordict/', None),
	'torchrl': ('https://pytorch.org/rl/', None),
	'torchaudio': ('https://pytorch.org/audio/stable/', None),
	'torchtext': ('https://pytorch.org/text/stable/', None),
	'torchvision': ('https://pytorch.org/vision/stable/', None),
	'pandas': ('http://pandas.pydata.org/pandas-docs/dev', None)
}

mathjax_config = {
    'tex2jax': {
        'inlineMath': [ ["\\(","\\)"] ],
        'displayMath': [["\\[","\\]"] ],
    },
}

mathjax3_config = {
  "tex": {
    "inlineMath": [['\\(', '\\)']],
    "displayMath": [["\\[", "\\]"]],
  }
}


templates_path = ['_templates']
exclude_patterns = []


# -- Certain formatting options ---------------------------------------------------
add_module_names = False


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'renku'
html_static_path = ['_static']
