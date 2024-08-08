# -*- coding: utf-8 -*-
"""
Parametric Graphs
"""

from typing import Optional

import networkx as nx
from mathics.core.atoms import Integer, Integer2
from mathics.core.evaluation import Evaluation
from mathics.core.expression import Expression

from pymathics.graph.base import (
    Graph,
    SymbolUndirectedEdge,
    _graph_from_list,
    _NetworkXBuiltin,
    graph_helper,
)
from pymathics.graph.eval.harary import hnm_harary_graph
from pymathics.graph.eval.parametric import (
    eval_complete_graph,
    eval_full_rary_tree,
    eval_hkn_harary,
)
from pymathics.graph.tree import DEFAULT_TREE_OPTIONS

# TODO: Can this code can be DRY'd more?


class BalancedTree(_NetworkXBuiltin):
    """
    <url>
    :WMA:
    https://reference.wolfram.com/language/ref/BalancedTree.html</url>

    <dl>
      <dt>'BalancedTree[$r$, $h$]'
      <dd>Returns the perfectly balanced $r$-ary tree of height $h$.

      In this tree produced, all non-leaf nodes will have $r$ children and \
      the height of the path from root $r$ to any leaf will be $h$.
    </dl>

    >> BalancedTree[2, 3]
     = -Graph-

    """

    messages = {
        "ilsmp": "Expected a non-negative integer at position 1 in ``.",
        "ilsmp2": "Expected a non-negative integer at position 2 in ``.",
        "mem": "Out of memory",
    }

    options = DEFAULT_TREE_OPTIONS
    summary_text = "build a balanced tree graph"

    def eval(
        self, r: Integer, h: Integer, expression, evaluation: Evaluation, options: dict
    ) -> Optional[Graph]:
        "%(name)s[r_Integer, h_Integer, OptionsPattern[%(name)s]]"
        py_r = r.value

        if py_r < 0:
            evaluation.message(self.get_name(), "ilsmp", expression)
            return None

        py_h = h.value
        if py_h < 0:
            evaluation.message(self.get_name(), "ilsmp2", expression)
            return None

        args = (py_r, py_h)
        g = graph_helper(nx.balanced_tree, options, True, "tree", evaluation, 0, *args)
        if not g:
            return None
        g.G.r = r
        g.G.h = h
        return g


class BarbellGraph(_NetworkXBuiltin):
    """
    <url>
    :Barbell graph:https://en.wikipedia.org/wiki/Barbell_graph </url> (<url>
    :NetworkX:https://networkx.org/documentation/networkx-2.8.8/reference/\
generated/networkx.generators.classic.barbell_graph.html</url>, <url>
    :Wolfram MathWorld:
    https://mathworld.wolfram.com/BarbellGraph.html</url>)

    <dl>
      <dt>'BarbellGraph[$m1$, $m2$]'
      <dd>Barbell Graph: two complete graphs connected by a path.
    </dl>

    >> BarbellGraph[4, 1]
     = -Graph-

    """

    messages = {
        "ilsmp": "Expected a non-negative integer at position 1 in ``.",
        "ilsmp2": "Expected a non-negative integer at position 2 in ``.",
    }
    summary_text = "build a n-m Barbell graph"

    def eval(
        self,
        m1: Integer,
        m2: Integer,
        expression,
        evaluation: Evaluation,
        options: dict,
    ) -> Optional[Graph]:
        "BarbellGraph[m1_Integer, m2_Integer, OptionsPattern[BarbellGraph]]"
        py_m1 = m1.value

        if py_m1 < 0:
            evaluation.message(self.get_name(), "ilsmp", expression)
            return

        py_m2 = m2.value
        if py_m2 < 0:
            evaluation.message(self.get_name(), "ilsmp2", expression)
            return

        args = (py_m1, py_m2)
        g = graph_helper(
            nx.barbell_graph, options, False, "spring", evaluation, None, *args
        )
        if not g:
            return None

        g.G.m1 = py_m1
        g.G.m2 = py_m2
        return g


class BinomialTree(_NetworkXBuiltin):
    """
    <url>
    :Binomial tree:
    https://en.wikipedia.org/wiki/Binomial_heap</url> (<url>
    :NetworkX:
    https://networkx.org/documentation/networkx-2.8.8/reference/\
generated/networkx.generators.classic.binomial_tree.html</url>, <url>
    :WMA:https://reference.wolfram.com/language/ref/BinomialTree.html</url>)

    <dl>
      <dt>'BinomialTree[$n$]'
      <dd>Returns the Binomial Tree of order $n$.

      The binomial tree of order $n$ with root $R$ is defined as:

      If $k$=0, 'B[$k$]' = 'B[0]' = {$R$}. i.e., the binomial tree of order \
      zero consists of a single node, $R$.

      If $k$>0, 'B[$k$]' = {$R$, 'B[0]', 'B[1]' ... 'B[$k$]'}, i.e., the binomial tree \
      of order $k$>0 comprises the root $R$, and $k$ binomial subtrees, \
      'B[0]' to 'B[$k$]'.

      Binomial trees are the underlying data structure in <url>
      :Binomial heaps:
      https://en.wikipedia.org/wiki/Binomial_heap#Binomial_tree</url>.
    </dl>

    >> BinomialTree[3]
     = -Graph-

    """

    messages = {
        "ilsmp": "Expected a non-negative integer at position 1 in ``.",
        "mem": "Out of memory",
    }
    summary_text = "build a binomial tree"

    def eval(
        self, n: Integer, expression, evaluation: Evaluation, options: dict
    ) -> Optional[Graph]:
        "%(name)s[n_Integer, OptionsPattern[%(name)s]]"
        py_n = n.value

        if py_n < 0:
            evaluation.message(self.get_name(), "ilsmp", expression)
            return

        args = (py_n,)
        g = graph_helper(nx.binomial_tree, options, True, "tree", evaluation, 0, *args)
        if not g:
            return None
        g.G.n = n
        return g


class CompleteGraph(_NetworkXBuiltin):
    """
    <url>
    :Complete Multipartite Graph:
    https://en.wikipedia.org/wiki/Multipartite_graph</url> (<url>
    :NetworkX:
    https://networkx.org/documentation/networkx-2.8.8/reference/\
generated/networkx.generators.classic.complete_multipartite_graph.html</url>, <url>
    :WMA:
    https://reference.wolfram.com/language/ref/CompleteGraph.html</url>)

    <dl>
      <dt>'CompleteGraph[$n$]'
      <dd>Returns the complete graph with $n$ vertices, $K_n$.
    </dl>

    >> CompleteGraph[8]
     = -Graph-
    """

    messages = {
        "ilsmp": "Expected a positive integer at position 1 in ``.",
    }

    summary_text = "build a completely-connected graph"

    def eval(self, n: Integer, expression, evaluation: Evaluation, options: dict):
        "CompleteGraph[n_Integer, OptionsPattern[CompleteGraph]]"
        return eval_complete_graph(self, n, expression, evaluation, options)

    def eval_multipartite(
        self, n, evaluation: Evaluation, options: dict
    ) -> Optional[Graph]:
        "CompleteGraph[n_List, OptionsPattern[CompleteGraph]]"
        if all(isinstance(i, Integer) for i in n.elements):
            return Graph(nx.complete_multipartite_graph(*[i.value for i in n.elements]))


class CompleteKaryTree(_NetworkXBuiltin):
    """
    <url>
    :M-ary Tree:
    https://en.wikipedia.org/wiki/M-ary_tree</url> (<url>
    :NetworkX:
    https://networkx.org/documentation/networkx-2.8.8/reference/\
generated/networkx.generators.classic.full_rary_tree.html</url>, <url>
    :WMA:
    https://reference.wolfram.com/language/ref/CompleteKaryTree.html</url>)

    <dl>
      <dt>'CompleteKaryTree[$n$, $k$]'
      <dd>Creates a complete $k$-ary tree of $n$ levels.
    </dl>

    In the returned tree, with $n$ nodes, the from root $R$ to any \
    leaf be $k$.

    >> CompleteKaryTree[2, 3]
     = -Graph-

    >> CompleteKaryTree[3]
     = -Graph-

    """

    options = DEFAULT_TREE_OPTIONS
    summary_text = "build a complete k-ary tree"

    def eval(
        self, k: Integer, n: Integer, expression, evaluation: Evaluation, options: dict
    ):
        "CompleteKaryTree[n_Integer, k_Integer, OptionsPattern[CompleteKaryTree]]"

        n_int = n.value
        k_int = k.value

        new_n_int = int(((k_int**n_int) - 1) / (k_int - 1))
        return eval_full_rary_tree(
            self, k, Integer(new_n_int), expression, evaluation, options
        )

    # FIXME: can be done with rules?
    def eval_2(self, n: Integer, expression, evaluation: Evaluation, options: dict):
        "CompleteKaryTree[n_Integer, OptionsPattern[CompleteKaryTree]]"

        n_int = n.value

        new_n_int = int(2**n_int) - 1
        return eval_full_rary_tree(
            self, Integer2, Integer(new_n_int), expression, evaluation, options
        )


class CycleGraph(_NetworkXBuiltin):
    """
    <url>:Cycle Graph:
    https://en.wikipedia.org/wiki/Cycle_graph</url> (<url>
    :WMA:
    https://reference.wolfram.com/language/ref/CycleGraph.html</url>)

    <dl>
      <dt>'CycleGraph[$n$]'
      <dd>Returns the cycle graph with $n$ vertices $C_n$.
    </dl>

    >> CycleGraph[5, PlotLabel -> "C_i"]
     = -Graph-
    """

    summary_text = "build a cycle graph"

    def eval(
        self, n: Integer, expression, evaluation: Evaluation, options: dict
    ) -> Optional[Graph]:
        "CycleGraph[n_Integer, OptionsPattern[CycleGraph]]"
        n_int = n.value
        if n_int < 3:
            return eval_complete_graph(self, n, expression, evaluation, options)
        else:
            return eval_hkn_harary(self, Integer2, n, expression, evaluation, options)


class GraphAtlas(_NetworkXBuiltin):
    """
    <url>:NetworkX:
    https://networkx.org/documentation/networkx-2.8.8/reference/\
generated/networkx.generators.atlas.graph_atlas.html
    </url>

    <dl>
      <dt>'GraphAtlas[$n$]'
      <dd>Returns graph number $i$ from the NetworkX's Graph \
      Atlas. There are about 1200 of them and get large as $i$ \
      increases.
    </dl>

    >> GraphAtlas[1000]
     = -Graph-
    """

    messages = {
        "ilsmp": "Expected a positive integer at position 1 in ``.",
    }
    summary_text = "retrieve a graph by number from the NetworkX Atlas"

    def eval(
        self, n: Integer, expression, evaluation: Evaluation, options: dict
    ) -> Optional[Graph]:
        "GraphAtlas[n_Integer, OptionsPattern[GraphAtlas]]"
        py_n = n.value

        if py_n < 1:
            evaluation.message(self.get_name(), "ilsmp", expression)
            return

        args = (py_n,)
        g = graph_helper(
            nx.graph_atlas, options, False, "spring", evaluation, None, *args
        )
        if not g:
            return None
        g.n = py_n
        return g


class HknHararyGraph(_NetworkXBuiltin):
    """
    <url>:NetworkX:
    https://networkx.org/documentation/networkx-2.8.8/reference\
/generated/networkx.generators.harary_graph.hkn_harary_graph.html#hkn-harary-graph</url>, <url>
    :WMA:
    https://reference.wolfram.com/language/ref/HknHararyGraph.html</url>

    <dl>
      <dt>'HknHararyGraph[$k$, $n$]'
      <dd>Returns the Harary graph with given node connectivity and node number.

      This second generator gives the Harary graph that minimizes the \
      number of edges in the graph with given node connectivity and   \
      number of nodes.

      Harary, F.  The Maximum Connectivity of a Graph.  \
      Proc. Nat. Acad. Sci. USA 48, 1142-1146, 1962.
    </dl>

    >> HknHararyGraph[3, 10]
     = -Graph-

    """

    messages = {
        "ilsmp": "Expected a non-negative integer at position 1 in ``.",
        "ilsmp2": "Expected a non-negative integer at position 2 in ``.",
    }
    summary_text = "build a Hkn Harary graph"

    def eval(self, k, n, expression, evaluation: Evaluation, options: dict):
        "%(name)s[k_Integer, n_Integer, OptionsPattern[%(name)s]]"
        return eval_hkn_harary(self, k, n, expression, evaluation, options)


class HmnHararyGraph(_NetworkXBuiltin):
    """
    <url>
    :NetworkX:
    https://networkx.org/documentation/networkx-2.8.8/reference\
/generated/networkx.generators.harary_graph.hnm_harary_graph.html</url>, <url>
   :WMA:
    https://reference.wolfram.com/language/ref/HmnHararyGraph.html</url>

    <dl>
      <dt>'HmnHararyGraph[$m$, $n$]'
      <dd>Returns the Harary graph with given numbers of nodes and edges.

      This generator gives the Harary graph that maximizes the node \
      connectivity with given number of nodes and given number of \
      edges.

      Harary, F.  The Maximum Connectivity of a Graph.\
      Proc. Nat. Acad. Sci. USA 48, 1142-1146, 1962.
    </dl>

    >> HmnHararyGraph[5, 10]
     = -Graph-
    """

    messages = {
        "ilsmp": "Expected a non-negative integer at position 1 in ``.",
        "ilsmp2": "Expected a non-negative integer at position 2 in ``.",
    }

    summary_text = "build a Hmn Harary graph"

    def eval(
        self, n: Integer, m: Integer, expression, evaluation: Evaluation, options: dict
    ) -> Optional[Graph]:
        "%(name)s[n_Integer, m_Integer, OptionsPattern[%(name)s]]"
        py_n = n.value

        if py_n < 0:
            evaluation.message(self.get_name(), "ilsmp", expression)
            return

        py_m = m.value

        if py_m < 0:
            evaluation.message(self.get_name(), "ilsmp2", expression)
            return

        args = (py_n, py_m)
        g = graph_helper(
            hnm_harary_graph, options, False, "circular", evaluation, None, *args
        )
        if not g:
            return None
        g.m = py_m
        return g


class KaryTree(_NetworkXBuiltin):
    """
    <url>
    :M-ary Tree:https://en.wikipedia.org/wiki/M-ary_tree
    </url>


    <dl>
      <dt>'KaryTree[$r$, $n$]'
      <dd>Creates binary tree of $n$ vertices.
    </dl>

    <dl>
      <dt>'KaryTree[$n$, $k$]'
      <dd>Creates $k$-ary tree with $n$ vertices.
    </dl>

    >> KaryTree[10]
     = -Graph-

    >> KaryTree[3, 10]
     = -Graph-

    """

    messages = {
        "ilsmp": "Expected a non-negative integer at position 1 in ``.",
        "ilsmp2": "Expected a non-negative integer at position 2 in ``.",
        "mem": "Out of memory",
    }

    options = DEFAULT_TREE_OPTIONS
    summary_text = "build a k-ary tree"

    def eval(
        self, n: Integer, expression, evaluation: Evaluation, options: dict
    ) -> Optional[Graph]:
        "KaryTree[n_Integer, OptionsPattern[KaryTree]]"
        return eval_full_rary_tree(self, Integer2, n, expression, evaluation, options)

    def eval_with_k(
        self, n: Integer, k: Integer, expression, evaluation: Evaluation, options: dict
    ) -> Optional[Graph]:
        "KaryTree[n_Integer, k_Integer, OptionsPattern[KaryTree]]"
        return eval_full_rary_tree(self, k, n, expression, evaluation, options)


class LadderGraph(_NetworkXBuiltin):
    """
    <url>
    :Ladder graph:https://en.wikipedia.org/wiki/Ladder_graph</url> (<url>
    :NetworkX:
    https://networkx.org/documentation/networkx-2.8.8/reference\
/generated/networkx.generators.classic.ladder_graph.html</url>)

    <dl>
      <dt>'LadderGraph[$n$]'
      <dd>Returns the Ladder graph of length $n$.
    </dl>

    >> LadderGraph[8]
     = -Graph-
    """

    messages = {
        "ilsmp": "Expected a positive integer at position 1 in ``.",
    }
    summary_text = "build a ladder tree"

    def eval(
        self, n: Integer, expression, evaluation: Evaluation, options: dict
    ) -> Optional[Graph]:
        "LadderGraph[n_Integer, OptionsPattern[LadderGraph]]"
        py_n = n.value

        if py_n < 1:
            evaluation.message(self.get_name(), "ilsmp", expression)
            return

        args = (py_n,)
        g = graph_helper(
            nx.ladder_graph, options, False, "spring", evaluation, 0, *args
        )
        if not g:
            return None
        g.G.n = n
        return g


class PathGraph(_NetworkXBuiltin):
    """
    <url>
    :Path graph:https://en.wikipedia.org/wiki/Path_graph
    </url> (<url>:WMA:https://reference.wolfram.com/language/ref/PathGraph.html
    </url>)
    <dl>
      <dt>'PathGraph[{$v_1$, $v_2$, ...}]'
      <dd>Returns a Graph with a path with vertices $v_i$ and \
      edges between $v-i$ and $v_i+1$ .
    </dl>

    >> PathGraph[{1, 2, 3}]
     = -Graph-
    """

    summary_text = "build a path graph"

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


class RandomTree(_NetworkXBuiltin):
    """
    <url>:NetworkX:
    https://networkx.org/documentation/networkx-2.8.8/reference\
/generated/networkx.generators.trees.random_tree.html</url>, <url>
    :WMA:
    https://reference.wolfram.com/language/ref/RandomTree.html</url>

    <dl>
      <dt>'RandomTree[$n$]'
      <dd>Returns a uniformly random tree on $n$ nodes.
    </dl>

    >> RandomTree[3]
     = -Graph-

    """

    messages = {
        "ilsmp": "Expected a non-negative integer at position 1 in ``.",
    }

    summary_text = "build a random tree"

    def eval(
        self, n: Integer, expression, evaluation: Evaluation, options: dict
    ) -> Optional[Graph]:
        "RandomTree[n_Integer, OptionsPattern[RandomTree]]"
        py_n = n.value

        if py_n < 0:
            evaluation.message(self.get_name(), "ilsmp", expression)
            return

        args = (py_n,)
        g = graph_helper(nx.random_tree, options, False, "tree", evaluation, 0, *args)
        if not g:
            return None
        g.G.n = n
        return g


class StarGraph(_NetworkXBuiltin):
    """
    <url>
    :Star graph:https://en.wikipedia.org/wiki/Star_graph
    </url>(<url>:NetworkX:
    https://networkx.org/documentation/networkx-2.8.8/reference\
/generated/networkx.generators.classic.star_graph.html
    </url>, <url>:WMA:
    https://reference.wolfram.com/language/ref/StarGraph.html
    </url>)
    <dl>
      <dt>'StarGraph[$n$]'
      <dd>Returns a star graph with $n$ vertices.
    </dl>

    >> StarGraph[8]
     = -Graph-
    """

    messages = {
        "ilsmp": "Expected a positive integer at position 1 in ``.",
    }
    summary_text = "build a star graph"

    def eval(
        self, n: Integer, expression, evaluation: Evaluation, options: dict
    ) -> Optional[Graph]:
        "StarGraph[n_Integer, OptionsPattern[StarGraph]]"
        py_n = n.value

        if py_n < 1:
            evaluation.message(self.get_name(), "ilsmp", expression)
            return

        args = (py_n,)
        g = graph_helper(nx.star_graph, options, False, "spring", evaluation, 0, *args)
        if not g:
            return None
        g.G.n = py_n
        return g
