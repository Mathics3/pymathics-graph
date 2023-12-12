"""
Trees
"""
import networkx as nx
from mathics.core.atoms import Atom
from mathics.core.evaluation import Evaluation
from mathics.core.symbols import SymbolConstant

from pymathics.graph.base import (
    DEFAULT_GRAPH_OPTIONS,
    _graph_from_list,
    _NetworkXBuiltin,
)
from pymathics.graph.eval.tree import eval_TreeGraphQ

DEFAULT_TREE_OPTIONS = {
    **DEFAULT_GRAPH_OPTIONS,
    **{"GraphLayout": '"tree"'},
}

from mathics.core.builtin import AtomBuiltin


# FIXME: do we need to have TreeGraphAtom and TreeGraph?
# Can't these be combined into one?
class TreeGraphAtom(AtomBuiltin):
    """
    <url>:Tree Graph:https://en.wikipedia.org/wiki/Tree_(graph_theory)</url>
    (<url>:WMA:https://reference.wolfram.com/language/ref/TreeGraph.html</url>)
    <dl>
      <dt>'TreeGraph'[$edges$]
      <dd>Build a Tree-like graph from the list of edges $edges$.
      <dt>'TreeGraph'[$vert$, $edges$]
      <dd>build a Tree-like graph from the list of vertices $vert$ and  edges $edges$.
    </dl>


    >> TreeGraph[{1->2, 2->3, 2->4}]
     = -Graph-

    If the $edges$ does not match with a tree-like pattern, the evaluation fails:
    >> TreeGraph[{1->2, 2->3, 3->1}]
     : Graph is not a tree.
     = TreeGraph[{1 -> 2, 2 -> 3, 3 -> 1}]
    """

    options = DEFAULT_TREE_OPTIONS

    messages = {
        "v": "Expected first parameter vertices to be a list of vertices",
        "notree": "Graph is not a tree.",
    }
    summary_text = "build a tree graph"

    def eval(self, rules, evaluation: Evaluation, options: dict):
        "TreeGraph[rules_List, OptionsPattern[%(name)s]]"
        g = _graph_from_list(rules.elements, options)
        if not nx.is_tree(g.G):
            evaluation.message(self.get_name(), "notree")
            return

        g.G.graph_layout = "tree"
        # Compute/check/set for root?
        return g

    def eval_with_v_e(self, vertices, edges, evaluation: Evaluation, options: dict):
        "TreeGraph[vertices_List, edges_List, OptionsPattern[%(name)s]]"
        if not all(isinstance(v, Atom) for v in vertices.elements):
            evaluation.message(self.get_name(), "v")
            return

        g = _graph_from_list(
            edges.elements, options=options, new_vertices=vertices.elements
        )
        if not nx.is_tree(g.G):
            evaluation.message(self.get_name(), "notree")
            return

        g.G.graph_layout = "tree"
        # Compute/check/set for root?
        return g


class TreeGraphQ(_NetworkXBuiltin):
    """
    <url>:Tree Graph:https://en.wikipedia.org/wiki/Tree_(graph_theory)</url>
    (<url>:WMA:https://reference.wolfram.com/language/ref/TreeGraphQ.html</url>)

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

    summary_text = "test for a tree-like graph"

    def eval(
        self, g, expression, evaluation: Evaluation, options: dict
    ) -> SymbolConstant:
        "TreeGraphQ[g_, OptionsPattern[%(name)s]]"
        return eval_TreeGraphQ(g)
