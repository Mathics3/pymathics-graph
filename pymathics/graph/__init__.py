"""Pymathics Graph - Working with Graphs (Vertices and Edges)

This module provides functions and variables for workting with
graphs.

Example:
   In[1]:= LoadModule["pymathics.graph"]
   Out[1]= pymathics.graph
   In[2]:= BinomialTree[3]
   In[3]:= BinomialTree[6]
   In[4]:= CompleteKaryTree[3, VertexLabels->True]
"""

from pymathics.graph.__main__ import *  # noqa
from pymathics.graph.algorithms import *  # noqa
from pymathics.graph.generators import *  # noqa
from pymathics.graph.tree import *  # noqa
from pymathics.graph.version import __version__  # noqa

pymathics_version_data = {
    "author": "The Mathics Team",
    "version": __version__,
    "name": "Graph",
    "requires": ["networkx"],
}
