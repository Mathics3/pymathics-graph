import networkx as nx
from mathics.core.evaluation import Evaluation
from mathics.core.expression import Expression

from pymathics.graph.base import (
    Graph,
    SymbolUndirectedEdge,
    _graph_from_list,
    _NetworkXBuiltin,
)
from pymathics.graph.tree import DEFAULT_TREE_OPTIONS


class PathGraph(_NetworkXBuiltin):
    """
    <url>
    :WMA:https://reference.wolfram.com/language/ref/PathGraph.html
    </url>

    <dl>
      <dt>'PathGraph[{$v_1$, $v_2$, ...}]'
      <dd>Returns a Graph with a path with vertices $v_i$ and edges between $v-i$ and $v_i+1$ .
    </dl>

    >> PathGraph[{1, 2, 3}]
     = -Graph-
    """

    summary_text = "create a graph from a path"

    def eval(self, e, evaluation: Evaluation, options: dict) -> Graph:
        "PathGraph[e_List, OptionsPattern[PathGraph]]"
        elements = e.elements

        def edges():
            for u, v in zip(elements, elements[1:]):
                yield Expression(SymbolUndirectedEdge, u, v)

        g = _graph_from_list(edges(), options)
        g.G.graph_layout = (
            options["System`GraphLayout"].get_string_value() or "spiral_equidistant"
        )
        return g


class TreeGraph(Graph):
    """
    <url>
    :WMA:https://reference.wolfram.com/language/ref/TreeGraph.html
    </url>

    <dl>
      <dt>'TreeGraph[{$edge_1$, $edge_2$, ...}]'
      <dd>create a tree-like from a list of edges.
    </dl>


    >> TreeGraph[{1->2, 2->3, 2->4}]
     = -Graph-

    """

    options = DEFAULT_TREE_OPTIONS

    messages = {
        "notree": "Graph is not a tree.",
    }

    summary_text = "build a tree graph"

    def __init__(self, G, **kwargs):
        super(Graph, self).__init__()
        if not nx.is_tree(G):
            raise ValueError
        self.G = G


# TODO: PlanarGraph
