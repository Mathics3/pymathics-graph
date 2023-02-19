"""
Random Graphs
"""

import networkx as nx
from mathics.builtin.numbers.randomnumbers import RandomEnv
from mathics.core.atoms import Integer, Integer1
from mathics.core.evaluation import Evaluation
from mathics.core.list import ListExpression

from pymathics.graph.base import (
    Graph,
    _convert_networkx_graph,
    _NetworkXBuiltin,
    _process_graph_options,
    has_directed_option,
)


class RandomGraph(_NetworkXBuiltin):
    """
    <url>
    :WMA link:https://reference.wolfram.com/language/ref/RandomGraph.html
    </url>

    <dl>
      <dt>'RandomGraph[{$n$, $m$}]'
      <dd>Returns a pseudorandom graph with $n$ vertices and $m$ edges.

      <dt>'RandomGraph[{$n$, $m$}, $k$]'
      <dd>Returns list of $k$ RandomGraph[{$n$, $m$}].
    </dl>
    """

    summary_text = "build a random graph"

    def _generate(
        self, n: Integer, m: Integer, k: Integer, evaluation: Evaluation, options: dict
    ) -> Graph:
        py_n = n.value
        py_m = m.value
        py_k = k.value
        is_directed = has_directed_option(options)

        with RandomEnv(evaluation) as _:
            for _ in range(py_k):
                # seed = rand.randint(0, 2 ** 63 - 1) # 2**63 is too large
                G = nx.gnm_random_graph(py_n, py_m, directed=is_directed)
                yield _convert_networkx_graph(G, options)

    def eval_nm(
        self, n: Integer, m: Integer, expression, evaluation: Evaluation, options: dict
    ) -> Graph:
        "RandomGraph[{n_Integer, m_Integer}, OptionsPattern[RandomGraph]]"
        g = list(self._generate(n, m, Integer1, evaluation, options))[0]
        _process_graph_options(g, options)
        return g

    def eval_nmk(
        self,
        n: Integer,
        m: Integer,
        k: Integer,
        expression,
        evaluation: Evaluation,
        options: dict,
    ) -> Graph:
        "RandomGraph[{n_Integer, m_Integer}, k_Integer, OptionsPattern[RandomGraph]]"
        return ListExpression(*self._generate(n, m, k, evaluation, options))
