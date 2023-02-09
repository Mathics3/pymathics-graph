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

from pymathics.graph.algorithms import (
    ConnectedComponents,
    GraphDistance,
    FindSpanningTree,
    PlanarGraphQ,
    WeaklyConnectedComponents,
)

from pymathics.graph.generators import (
    BalancedTree,
    BarbellGraph,
    BinomialTree,
    CompleteGraph,
    CompleteKaryTree,
    CycleGraph,
    FullRAryTree,
    GraphAtlas,
    HknHararyGraph,
    HmnHararyGraph,
    KaryTree,
    LadderGraph,
    PathGraph,
    RandomGraph,
    RandomTree,
    StarGraph,
)
from pymathics.graph.tree import TreeGraphAtom, TreeGraph, TreeGraphQ
from pymathics.graph.version import __version__

pymathics_version_data = {
    "author": "The Mathics3 Team",
    "version": __version__,
    "name": "Graph",
    "requires": ["networkx"],
}

# Thsee are the publicly exported names
__all__ = [
    "AcyclicGraphQ",
    "BalancedTree",
    "BarbellGraph",
    "BetweennessCentrality",
    "BinomialTree",
    "ClosenessCentrality",
    "CompleteGraph",
    "CompleteKaryTree",
    "ConnectedComponents",
    "ConnectedGraphQ",
    "CycleGraph",
    "DegreeCentrality",
    "DirectedEdge",
    "DirectedGraphQ",
    "EdgeConnectivity",
    "EdgeCount",
    "EdgeIndex",
    "EdgeList",
    "EdgeRules",
    "EigenvectorCentrality",
    "FindShortestPath",
    "FindSpanningTree",
    "FindVertexCut",
    "FullRAryTree",
    "Graph",
    "GraphAtlas",
    "GraphAtom",
    "GraphBox",
    "GraphDistance",
    "HITSCentrality",
    "HighlightGraph",
    "HknHararyGraph",
    "HmnHararyGraph",
    "KaryTree",
    "KatzCentrality",
    "LadderGraph",
    "LoopFreeGraphQ",
    "MixedGraphQ",
    "MultigraphQ",
    "PageRankCentrality",
    "PathGraph",
    "PathGraphQ",
    "PlanarGraphQ",
    "Property",
    "PropertyValue",
    "RandomGraph",
    "RandomTree",
    "SimpleGraphQ",
    "StarGraph",
    "TreeGraph",
    "TreeGraphAtom",
    "TreeGraphQ",
    "UndirectedEdge",
    "VertexAdd",
    "VertexConnectivity",
    "VertexCount",
    "VertexDegree",
    "VertexDelete",
    "VertexIndex",
    "VertexList",
    "WeaklyConnectedComponents",
    "__version__",
    "pymathics_version_data",
]
