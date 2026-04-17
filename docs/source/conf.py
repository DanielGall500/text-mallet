import os
import sys
sys.path.insert(0, os.path.abspath('../../src')) 

project = 'text-mallet'
copyright = '2026, InfAI e.V.'
author = 'Daniel Gallagher'
release = '0.0.1'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode', 
]

templates_path = ['_templates']
exclude_patterns = []


html_theme = 'shibuya'
html_static_path = ['_static']
