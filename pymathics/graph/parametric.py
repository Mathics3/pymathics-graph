# -*- coding: utf-8 -*-
"""
Parametric Graphs
"""

from typing import Optional

import networkx as nx
from mathics.core.atoms import Integer
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
    :WMA:https://reference.wolfram.com/language/ref/BalancedTree.html
    </url>

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
    :Barbell graph:https://en.wikipedia.org/wiki/Barbell_graph
    </url> (
    <url>
    :NetworkX:https://networkx.org/documentation/stable/reference/\
generated/networkx.generators.classic.barbell_graph.html</url>
)

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

        g.G.m1 = m1
        g.G.m2 = m2
        return g


class BinomialTree(_NetworkXBuiltin):
    """
    <url>
    :Binomial tree:https://en.wikipedia.org/wiki/Binomial_tree
    </url> (
    <url>
    :NetworkX:https://networkx.org/documentation/stable/reference/\
generated/networkx.generators.classic.binomial_tree.html</url>,
    <url>
    :WMA:https://reference.wolfram.com/language/ref/BinomialTree.html
    </url>
    )

    <dl>
      <dt>'BinomialTree[$n$]'
      <dd>Returns the Binomial Tree of order $n$.

      The binomial tree of order $n$ with root $R$ is defined as:

      If $k$=0,  $B[k]$ = $B[0]$ = {$R$}. i.e., the binomial tree of order \
      zero consists of a single node, $R$.

      If $k>0$, B[k] = {$R$, $B[0$], $B[1]$ .. $B[k]$, i.e., the binomial tree \
      of order $k$>0 comprises the root $R$, and $k$ binomial subtrees, \
      $B[0] to $B[k].

      Binomial trees are the underlying datastructre in Binomial Heaps.
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
    ) -> Graph:
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

    summary_text = "build a completely connected graph"

    def eval(self, n: Integer, expression, evaluation: Evaluation, options: dict):
        "%(name)s[n_Integer, OptionsPattern[%(name)s]]"
        return eval_complete_graph(self, n, expression, evaluation, options)

    def eval_multipartite(self, n, evaluation: Evaluation, options: dict):
        "%(name)s[n_List, OptionsPattern[%(name)s]]"
        if all(isinstance(i, Integer) for i in n.elements):
            return Graph(
                nx.complete_multipartite_graph(*[i.get_int_value() for i in n.elements])
            )


class CompleteKaryTree(_NetworkXBuiltin):
    """<dl>
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

    def eval(self, k, n, expression, evaluation: Evaluation, options: dict):
        "%(name)s[n_Integer, k_Integer, OptionsPattern[%(name)s]]"

        n_int = n.value
        k_int = k.value

        new_n_int = int(((k_int**n_int) - 1) / (k_int - 1))
        return eval_full_rary_tree(
            self, k, Integer(new_n_int), expression, evaluation, options
        )

    # FIXME: can be done with rules?
    def eval_2(self, n, expression, evaluation: Evaluation, options: dict):
        "%(name)s[n_Integer, OptionsPattern[%(name)s]]"

        n_int = n.get_int_value()

        new_n_int = int(2**n_int) - 1
        return eval_full_rary_tree(
            self, Integer(2), Integer(new_n_int), expression, evaluation, options
        )


class CycleGraph(_NetworkXBuiltin):
    """<dl>
        <dt>'CycleGraph[$n$]'
        <dd>Returns the cycle graph with $n$ vertices $C_n$.
      </dl>

    >> CycleGraph[5, PlotLabel -> "C_i"]
     = -Graph-
    """

    summary_text = "build a cycle graph"

    def eval(self, n: Integer, expression, evaluation: Evaluation, options: dict):
        "%(name)s[n_Integer, OptionsPattern[%(name)s]]"
        n_int = n.get_int_value()
        if n_int < 3:
            return eval_complete_graph(self, n, expression, evaluation, options)
        else:
            return eval_hkn_harary(self, Integer(2), n, expression, evaluation, options)


class FullRAryTree(_NetworkXBuiltin):
    """
    <dl>
      <dt>'FullRAryTree[$r$, $n$]'
      <dd>Creates a full $r$-ary tree of $n$ vertices.
    </dl>

    In the returned tree, with $n$ nodes, the from root $R$ to any
    leaf will differ by most 1, the height of the tree from any root
    to a leaf is O(log($n, $r$)).

    >> FullRAryTree[2, 10]
     = -Graph-

    """

    messages = {
        "ilsmp": "Expected a non-negative integer at position 1 in ``.",
        "ilsmp2": "Expected a non-negative integer at position 2 in ``.",
        "mem": "Out of memory",
    }

    options = DEFAULT_TREE_OPTIONS
    summary_text = "build a full r-ary tree"

    def eval(self, r, n, expression, evaluation: Evaluation, options: dict):
        "%(name)s[r_Integer, n_Integer, OptionsPattern[%(name)s]]"
        return eval_full_rary_tree(self, r, n, expression, evaluation, options)


class GraphAtlas(_NetworkXBuiltin):
    """<dl>
      <dt>'GraphAtlas[$n$]'
      <dd>Returns graph number $i$ from the Networkx's Graph \
      Atlas. There are about 1200 of them and get large as $i$ \
      increases.
    </dl>

    >> GraphAtlas[1000]
     = -Graph-
    """

    messages = {
        "ilsmp": "Expected a positive integer at position 1 in ``.",
    }
    summary_text = "build the i-esim graph from the Networkx atlas"

    def eval(self, n, expression, evaluation: Evaluation, options: dict):
        "%(name)s[n_Integer, OptionsPattern[%(name)s]]"
        py_n = n.get_int_value()

        if py_n < 1:
            evaluation.message(self.get_name(), "ilsmp", expression)
            return

        args = (py_n,)
        g = graph_helper(
            nx.graph_atlas, options, False, "spring", evaluation, None, *args
        )
        if not g:
            return None
        g.n = n
        return g


class HknHararyGraph(_NetworkXBuiltin):
    """
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

    def eval(self, n, m, expression, evaluation: Evaluation, options: dict):
        "%(name)s[n_Integer, m_Integer, OptionsPattern[%(name)s]]"
        py_n = n.value

        if py_n < 0:
            evaluation.message(self.get_name(), "ilsmp", expression)
            return

        py_m = m.get_int_value()

        if py_m < 0:
            evaluation.message(self.get_name(), "ilsmp2", expression)
            return

        args = (py_n, py_m)
        g = graph_helper(
            hnm_harary_graph, options, False, "circular", evaluation, None, *args
        )
        if not g:
            return None
        g.n = py_n
        g.m = py_m
        return g


class KaryTree(_NetworkXBuiltin):
    """
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
    summary_text = "build a Kary tree"

    def eval(
        self, n: Integer, expression, evaluation: Evaluation, options: dict
    ) -> Graph:
        "KaryTree[n_Integer, OptionsPattern[KaryTree]]"
        return eval_full_rary_tree(self, Integer(2), n, expression, evaluation, options)

    def eval_with_k(
        self, n: Integer, k: Integer, expression, evaluation: Evaluation, options: dict
    ) -> Graph:
        "KaryTree[n_Integer, k_Integer, OptionsPattern[KaryTree]]"
        return eval_full_rary_tree(self, k, n, expression, evaluation, options)


class LadderGraph(_NetworkXBuiltin):
    """
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
    ) -> Graph:
        "LadderGraph[n_Integer, OptionsPattern[%(name)s]]"
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
    ) -> Graph:
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
    ) -> Graph:
        "StarGraph[n_Integer, OptionsPattern[StarGraph]]"
        py_n = n.value

        if py_n < 1:
            evaluation.message(self.get_name(), "ilsmp", expression)
            return

        args = (py_n,)
        g = graph_helper(nx.star_graph, options, False, "spring", evaluation, 0, *args)
        if not g:
            return None
        g.G.n = n
        return g
