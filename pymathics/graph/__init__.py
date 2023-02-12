"""
Graphs - Vertices and Edges

Mathics3 Module that provides functions and variables for working with graphs.

Example:
   In[1]:= LoadModule["pymathics.graph"]
   Out[1]= pymathics.graph
   In[2]:= BinomialTree[3]
   In[3]:= BinomialTree[6]
   In[4]:= CompleteKaryTree[3, VertexLabels->True]

Networkx does the heavy lifting here.
"""

from pymathics.graph.base import (
    AdjacencyList,
    BetweennessCentrality,
    ClosenessCentrality,
    DegreeCentrality,
    DirectedEdge,
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
    HighlightGraph,
    HITSCentrality,
    KatzCentrality,
    PageRankCentrality,
    Property,
    PropertyValue,
    UndirectedEdge,
    VertexAdd,
    VertexConnectivity,
    VertexDelete,
    VertexIndex,
    VertexList,
)
from pymathics.graph.components import ConnectedComponents, WeaklyConnectedComponents
from pymathics.graph.curated import GraphData
from pymathics.graph.measures_and_metrics import (
    EdgeCount,
    GraphDistance,
    VertexCount,
    VertexDegree,
)
from pymathics.graph.operations import FindSpanningTree
from pymathics.graph.parametric import (
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
    RandomTree,
    StarGraph,
)
from pymathics.graph.properties import (
    AcyclicGraphQ,
    ConnectedGraphQ,
    DirectedGraphQ,
    LoopFreeGraphQ,
    MixedGraphQ,
    MultigraphQ,
    PathGraphQ,
    PlanarGraphQ,
    SimpleGraphQ,
)
from pymathics.graph.random import RandomGraph
from pymathics.graph.structured import PathGraph, TreeGraph
from pymathics.graph.tree import TreeGraphAtom, TreeGraphQ
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
    "AdjacencyList",
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
    "GraphData",
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
