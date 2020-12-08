import networkx as nx
from pymathics.graph.__main__ import Graph, _graph_from_list, DEFAULT_GRAPH_OPTIONS, _NetworkXBuiltin, WL_MARKER_TO_NETWORKX
from mathics.core.expression import String, Symbol

DEFAULT_TREE_OPTIONS = {
    **DEFAULT_GRAPH_OPTIONS,
    **{"GraphLayout": '"tree"'},
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

        g.G.graph_layout = "tree"
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

        g.G.graph_layout = "tree"
        # Compute/check/set for root?
        return g


class TreeGraph(Graph):
    options = DEFAULT_TREE_OPTIONS

    messages = {
        "notree": "Graph is not a tree.",
    }

    def __init__(self, G, **kwargs):
        super(Graph, self).__init__()
        if not nx.is_tree(G):
            evaluation.message(self.get_name(), "notree")
        self.G = G


class TreeGraphQ(_NetworkXBuiltin):
    """
    <dl>
      <dt>'TreeGraphQ[$g$]'
      <dd>returns $True$ if the graph $g$ is a tree and $False$ otherwise.
    </dl>

    >> TreeGraphQ[StarGraph[3]]
     = True
    >> TreeGraphQ[CompleteGraph[2]]
     = True
    >> TreeGraphQ[CompleteGraph[3]]
     = False
    """

    def apply(self, g, expression, evaluation, options):
        "TreeGraphQ[g_, OptionsPattern[%(name)s]]"
        if not isinstance(g, Graph):
            return Symbol("False")
        return Symbol("True" if nx.is_tree(g.G) else "False")
