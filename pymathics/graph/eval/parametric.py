"""
evaluation methods for Parametric Graphs
"""

from typing import Optional

import networkx as nx
from mathics.core.atoms import Integer
from mathics.core.evaluation import Evaluation

from pymathics.graph.base import Graph, graph_helper
from pymathics.graph.eval.harary import hkn_harary_graph


def eval_complete_graph(
    self, n: Integer, expression, evaluation: Evaluation, options: dict
) -> Optional[Graph]:
    py_n = n.value

    if py_n < 1:
        evaluation.message(self.get_name(), "ilsmp", expression)
        return

    args = (py_n,)
    g = graph_helper(
        nx.complete_graph, options, False, "circular", evaluation, None, *args
    )
    if not g:
        return None

    g.G.n = n
    return g


def eval_full_rary_tree(
    self, r: Integer, n: Integer, expression, evaluation: Evaluation, options: dict
) -> Optional[Graph]:
    """
    Call ``networkx.full_rary_tree()`` using parameters, ``r`` and ``t``.
    """
    py_r = r.value

    if py_r < 0:
        evaluation.message(self.get_name(), "ilsmp", expression)
        return

    py_n = n.value
    if py_n < 0:
        evaluation.message(self.get_name(), "ilsmp", expression)
        return

    args = (py_r, py_n)
    g = graph_helper(nx.full_rary_tree, options, True, "tree", evaluation, 0, *args)
    if not g:
        return None

    g.G.r = r
    g.G.n = n
    return g


def eval_hkn_harary(
    self, k: Integer, n: Integer, expression, evaluation: Evaluation, options: dict
) -> Optional[Graph]:
    py_k = k.value

    if py_k < 0:
        evaluation.message(self.get_name(), "ilsmp", expression)
        return

    py_n = n.value
    if py_n < 0:
        evaluation.message(self.get_name(), "ilsmp2", expression)
        return

    args = (py_k, py_n)
    g = graph_helper(
        hkn_harary_graph, options, False, "circular", evaluation, None, *args
    )
    if not g:
        return None
    g.k = py_k
    return g
