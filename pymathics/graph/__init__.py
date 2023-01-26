"""Pymathics Graph - Working with Graphs (Vertices and Edges)

This module provides functions and variables for working with
graphs.

Example:
   In[1]:= LoadModule["pymathics.graph"]
   Out[1]= pymathics.graph
   In[2]:= BinomialTree[3]
   In[3]:= BinomialTree[6]
   In[4]:= CompleteKaryTree[3, VertexLabels->True]
"""

from pymathics.graph.main import (
    AcyclicGraphQ,
    BetweennessCentrality,
    ClosenessCentrality,
    ConnectedGraphQ,
    DegreeCentrality,
    DirectedEdge,
    DirectedGraphQ,
    EdgeConnectivity,
    EdgeCount,
    EdgeIndex,
    EdgeList,
    EdgeRules,
    EigenvectorCentrality,
    FindShortestPath,
    FindVertexCut,
    Graph,
    GraphBox,
    HITSCentrality,
    HighlightGraph,
    KatzCentrality,
    LoopFreeGraphQ,
    MixedGraphQ,
    MultigraphQ,
    PageRankCentrality,
    PlanarGraphQ,
    PathGraphQ,
    Property,
    PropertyValue,
    SimpleGraphQ,
    UndirectedEdge,
    VertexAdd,
    VertexConnectivity,
    VertexCount,
    VertexDegree,
    VertexDelete,
    VertexIndex,
    VertexList,
)
from pymathics.graph.algorithms import *  # noqa
from pymathics.graph.generators import *  # noqa
from pymathics.graph.tree import *  # noqa
from pymathics.graph.version import __version__  # noqa

pymathics_version_data = {
    "author": "The Mathics3 Team",
    "version": __version__,
    "name": "Graph",
    "requires": ["networkx"],
}
