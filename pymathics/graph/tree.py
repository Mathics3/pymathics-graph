import networkx as nx
from pymathics.graph.__main__ import Graph, _graph_from_list, DEFAULT_GRAPH_OPTIONS
from mathics.core.expression import String

DEFAULT_TREE_OPTIONS = {
    **DEFAULT_GRAPH_OPTIONS,
    **{"Directed": "False", "GraphLayout": '"tree"'},
}

from mathics.builtin.base import Builtin, AtomBuiltin


class TreeGraphAtom(AtomBuiltin):
    """
    >> TreeGraph[{1->2, 2->3, 3->1}]
     = -Graph-

    """

    options = DEFAULT_TREE_OPTIONS

    messages = {
        "v": "Expected first parameter vertices to be a list of vertices",
        "notree": "Graph is not a tree.",
    }

    def apply(self, rules, evaluation, options):
        "TreeGraph[rules_List, OptionsPattern[%(name)s]]"
        g = _graph_from_list(rules.leaves, options)
        if not nx.is_tree(g.G):
            evaluation.message(self.get_name(), "notree")

        g.G.graph_layout = String("tree")
        # Compute/check/set for root?
        return g

    def apply_1(self, vertices, edges, evaluation, options):
        "TreeGraph[vertices_List, edges_List, OptionsPattern[%(name)s]]"
        if not all(isinstance(v, Atom) for v in vertices.leaves):
            evaluation.message(self.get_name(), "v")

        g = _graph_from_list(
            edges.leaves, options=options, new_vertices=vertices.leaves
        )
        if not nx.is_tree(g.G):
            evaluation.message(self.get_name(), "notree")

        g.G.graph_layout = String("tree")
        # Compute/check/set for root?
        return g


class TreeGraph(Graph):
    def __init__(self, G, **kwargs):
        super(Graph, self).__init__()
        self.G = G
