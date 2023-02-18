"""
Graphs - Vertices and Edges


A Graph is a tuple of a set of Nodes and Edges.

Mathics3 Module that provides functions and variables for working with graphs.

Examples:

   >> LoadModule["pymathics.graph"]
    = pymathics.graph

   >> BinomialTree[3, DirectedEdges->True]
     = -Graph-

   >> BalancedTree[3, 3]
     = -Graph-

   >> g = Graph[{1 -> 2, 2 -> 3, 3 <-> 4}, VertexLabels->True]
    = -Graph-

   >> ConnectedComponents[g]
    = {{3, 4}, {2}, {1}}

   >> WeaklyConnectedComponents[g]
    = {{1, 2, 3, 4}}

   >> GraphDistance[g, 1, 4]
    = 3

   >> GraphDistance[g, 3, 2]
    = Infinity

<url>
:NetworkX:
https://networkx.org</url> does the heavy lifting here.
"""

from pymathics.graph.base import (
    AdjacencyList,
    DirectedEdge,
    EdgeConnectivity,
    EdgeDelete,
    EdgeIndex,
    EdgeList,
    EdgeRules,
    FindShortestPath,
    FindVertexCut,
    Graph,
    GraphAtom,
    GraphBox,
    HighlightGraph,
    Property,
    PropertyValue,
    UndirectedEdge,
    VertexAdd,
    VertexConnectivity,
    VertexDelete,
    VertexIndex,
    VertexList,
)

from pymathics.graph.centralities import (
    BetweennessCentrality,
    ClosenessCentrality,
    DegreeCentrality,
    EigenvectorCentrality,
    HITSCentrality,
    KatzCentrality,
    PageRankCentrality,
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
    "EdgeDelete",
    "EdgeIndex",
    "EdgeList",
    "EdgeRules",
    "EigenvectorCentrality",
    "FindShortestPath",
    "FindSpanningTree",
    "FindVertexCut",
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
