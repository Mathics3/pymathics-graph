import networkx as nx

from mathics.core.evaluation import Evaluation
from mathics.core.expression import Expression
from pymathics.graph.base import (
    SymbolUndirectedEdge,
    Graph,
    _graph_from_list,
    _NetworkXBuiltin,
)

from pymathics.graph.tree import DEFAULT_TREE_OPTIONS


class PathGraph(_NetworkXBuiltin):
    """
    <dl>
      <dt>'PathGraph[{$v_1$, $v_2$, ...}]'
      <dd>Returns a Graph with a path with vertices $v_i$ and edges between $v-i$ and $v_i+1$ .
    </dl>
    >> PathGraph[{1, 2, 3}]
     = -Graph-
    """

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
    >> TreeGraph[{1->2, 2->3, 2->4}]
     = -Graph-

    """

    options = DEFAULT_TREE_OPTIONS

    messages = {
        "notree": "Graph is not a tree.",
    }

    def __init__(self, G, **kwargs):
        super(Graph, self).__init__()
        if not nx.is_tree(G):
            raise ValueError
        self.G = G


# TODO: PlanarGraph
