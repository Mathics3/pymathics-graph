import networkx as nx
from mathics.core.atoms import Atom
from mathics.core.evaluation import Evaluation
from mathics.core.symbols import SymbolConstant, SymbolFalse, SymbolTrue

from pymathics.graph.__main__ import (
    DEFAULT_GRAPH_OPTIONS,
    Graph,
    _graph_from_list,
    _NetworkXBuiltin,
)

DEFAULT_TREE_OPTIONS = {
    **DEFAULT_GRAPH_OPTIONS,
    **{"GraphLayout": '"tree"'},
}

from mathics.builtin.base import AtomBuiltin


def eval_TreeGraphQ(g: Graph) -> SymbolConstant:
    """
    Returns SymbolTrue if g is a (networkx) tree and SymbolFalse
    otherwise.
    """
    if not isinstance(g, Graph):
        return SymbolFalse
    return SymbolTrue if nx.is_tree(g.G) else SymbolFalse


# FIXME: do we need to have TreeGraphAtom and TreeGraph?
# Can't these be combined into one?
class TreeGraphAtom(AtomBuiltin):
    options = DEFAULT_TREE_OPTIONS

    messages = {
        "v": "Expected first parameter vertices to be a list of vertices",
        "notree": "Graph is not a tree.",
    }

    def eval(self, rules, evaluation: Evaluation, options: dict):
        "TreeGraph[rules_List, OptionsPattern[%(name)s]]"
        g = _graph_from_list(rules.elements, options)
        if not nx.is_tree(g.G):
            evaluation.message(self.get_name(), "notree")

        g.G.graph_layout = "tree"
        # Compute/check/set for root?
        return g

    def eval_with_v_e(self, vertices, edges, evaluation: Evaluation, options: dict):
        "TreeGraph[vertices_List, edges_List, OptionsPattern[%(name)s]]"
        if not all(isinstance(v, Atom) for v in vertices.elements):
            evaluation.message(self.get_name(), "v")

        g = _graph_from_list(
            edges.elements, options=options, new_vertices=vertices.elements
        )
        if not nx.is_tree(g.G):
            evaluation.message(self.get_name(), "notree")

        g.G.graph_layout = "tree"
        # Compute/check/set for root?
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

    def eval(
        self, g, expression, evaluation: Evaluation, options: dict
    ) -> SymbolConstant:
        "TreeGraphQ[g_, OptionsPattern[%(name)s]]"
        return eval_TreeGraphQ(g)
