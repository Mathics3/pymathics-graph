"""
Graph Operations and Modifications
"""

import networkx as nx
from mathics.core.convert.python import from_python
from mathics.core.evaluation import Evaluation
from mathics.core.expression import Expression

from pymathics.graph.base import (
    DEFAULT_GRAPH_OPTIONS,
    SymbolDirectedEdge,
    SymbolUndirectedEdge,
    _create_graph,
    _NetworkXBuiltin,
)


class FindSpanningTree(_NetworkXBuiltin):
    """
    <dl>
      <dt>'FindSpanningTree'[$g$]
      <dd>finds a spanning tree of the graph $g$.
    </dl>

    >> FindSpanningTree[CycleGraph[4]]
     = -Graph-
    """

    options = DEFAULT_GRAPH_OPTIONS

    def eval(self, graph, expression, evaluation: Evaluation, options: dict):
        "%(name)s[graph_, OptionsPattern[%(name)s]]"
        graph = self._build_graph(graph, evaluation, options, expression)
        if graph:
            graph.update_weights(evaluation)
            SymbolDirectedEdge if graph.G.is_directed() else SymbolUndirectedEdge
            # FIXME: put in edge to Graph conversion function?
            edges = [
                Expression(SymbolUndirectedEdge, from_python(u), from_python(v))
                for u, v in nx.minimum_spanning_edges(graph.G, data=False)
            ]
            g = _create_graph(edges, [None] * len(edges), options)
            if not hasattr(g.G, "graph_layout"):
                if hasattr(graph.G, "graph_layout"):
                    g.G.graph_layout = graph.G.graph_layout
                else:
                    g.G.graph_layout = "tree"
            return g


# TODO: Subgraph, NeighborhoodGraph + many others
