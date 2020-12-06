"""Pymathics Graph - Working with Graphs (Vertices and Edgies)

This module provides functions and variables for workting with
graphs.
"""


from pymathics.graph.version import __version__
from pymathics.graph.__main__ import *
from pymathics.graph.tree import *
from pymathics.graph.generators import *
from pymathics.graph.algorithms import *

pymathics_version_data = {
    "author": "The Mathics Team",
    "version": __version__,
    "name": "Graph",
    "requires": ["networkx"],
}
