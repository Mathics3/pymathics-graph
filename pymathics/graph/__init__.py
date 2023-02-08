"""
Graphs - Vertices and Edges

A Pymathics module that provides functions and variables for working with graphs.

Example:
   In[1]:= LoadModule["pymathics.graph"]
   Out[1]= pymathics.graph
   In[2]:= BinomialTree[3]
   In[3]:= BinomialTree[6]
   In[4]:= CompleteKaryTree[3, VertexLabels->True]

Networkx does the heavy lifting here.
"""

from pymathics.graph.base import (
    AcyclicGraphQ,
    BetweennessCentrality,
    ClosenessCentrality,
    ConnectedGraphQ,
    DegreeCentrality,
    DirectedEdge,
    DirectedGraphQ,
    EdgeConnectivity,
    EdgeIndex,
    EdgeList,
    EdgeRules,
    EigenvectorCentrality,
    FindShortestPath,
    FindVertexCut,
    Graph,
    GraphAtom,
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
    VertexDelete,
    VertexIndex,
    VertexList,
)

from pymathics.graph.measures_and_metrics import EdgeCount, VertexCount, VertexDegree

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
