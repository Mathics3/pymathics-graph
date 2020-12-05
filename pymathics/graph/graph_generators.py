from pymathics.graph.__main__ import (
    Graph,
    _NetworkXBuiltin,
    nx,
)

from pymathics.graph.tree import DEFAULT_TREE_OPTIONS

from mathics.core.expression import Expression, Integer, String
from typing import Callable, Optional

# TODO: Can this code can be DRY'd more?


def graph_helper(
    graph_generator_func: Callable,
    options: dict,
    can_digraph: bool,
    graph_layout: str,
    root: Optional[int] = None,
    *args,
    **kwargs
) -> Optional[Callable]:
    should_digraph = can_digraph and options["System`DirectedEdges"].to_python()
    try:
        G = (
            graph_generator_func(*args, create_using=nx.DiGraph, **kwargs)
            if should_digraph
            else graph_generator_func(*args, **kwargs)
        )
    except MemoryError:
        evaluation.message(self.get_name(), "mem", expression)
        return None
    G.graph_layout = options["System`GraphLayout"].get_string_value() or String(
        graph_layout
    )
    G.vertex_labels = options["System`VertexLabels"]
    g = Graph(G)

    if root is not None:
        G.root = g.root = root
    G.title = g.title = options["System`PlotLabel"]
    return g


class BalancedTree(_NetworkXBuiltin):
    """
    <dl>
      <dt>'BalancedTree[$r$, $h$]'
      <dd>Returns the perfectly balanced $r$-ary tree of height $h$.

      In this tree produced, all non-leaf nodes will have $r$ children and the height of
      the path from root $r$ to any leaf will be $h$.
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

    def apply(self, r, h, expression, evaluation, options):
        "%(name)s[r_Integer, h_Integer, OptionsPattern[%(name)s]]"
        py_r = r.get_int_value()

        if py_r < 0:
            evaluation.message(self.get_name(), "ilsmp", expression)
            return None

        py_h = h.get_int_value()
        if py_h < 0:
            evaluation.message(self.get_name(), "ilsmp2", expression)
            return None

        args = (py_r, py_h)
        g = graph_helper(nx.balanced_tree, options, True, "tree", 0, *args)
        if not g:
            return None
        g.G.r = r
        g.G.h = h
        return g


class BarbellGraph(_NetworkXBuiltin):
    """
    <dl>
      <dt>'BarbellGraph[$m1$, $m2$]'
      <dd>Barbell Graph: two complete graphs connected by a path.
    </dl>

    >> BarBellGraph[4, 1]
     = -Graph-

    """

    messages = {
        "ilsmp": "Expected a non-negative integer at position 1 in ``.",
        "ilsmp2": "Expected a non-negative integer at position 2 in ``.",
    }

    def apply(self, m1, m2, expression, evaluation, options):
        "%(name)s[m1_Integer, m2_Integer, OptionsPattern[%(name)s]]"
        py_m1 = m1.get_int_value()

        if py_m1 < 0:
            evaluation.message(self.get_name(), "ilsmp", expression)
            return

        py_m2 = m2.get_int_value()
        if py_m2 < 0:
            evaluation.message(self.get_name(), "ilsmp", expression)
            return

        args = (py_m1, py_m2)
        g = graph_helper(nx.barbell_graph, options, False, "spring", None, *args)
        if not g:
            return None

        g.G.m1 = m1
        g.G.m2 = m2
        return g


class BinomialTree(_NetworkXBuiltin):
    """
    <dl>
      <dt>'BinomialTree[$n$]'
      <dd>Returns the Binomial Tree of order $n$.

      The binomial tree of order $n$ with root $R$ is defined as:

      If $k$=0,  $B[k]$ = $B[0]$ = {$R$}. i.e., the binomial tree of order zero consists of a single node, $R$.

      If $k>0$, B[k] = {$R$, $B[0$], $B[1]$ .. $B[k]$, i.e., the binomial tree of order $k$>0 comprises the root $R$, and $k$ binomial subtrees, $B[0] to $B[k].

      Binomial trees the underlying datastructre in Binomial Heaps.
    </dl>

    >> BinomialTree[3]
     = -Graph-

    """

    messages = {
        "ilsmp": "Expected a non-negative integer at position 1 in ``.",
        "mem": "Out of memory",
    }

    def apply(self, n, expression, evaluation, options):
        "%(name)s[n_Integer, OptionsPattern[%(name)s]]"
        py_n = n.get_int_value()

        if py_n < 0:
            evaluation.message(self.get_name(), "ilsmp", expression)
            return

        args = (py_n,)
        g = graph_helper(nx.binomial_tree, options, False, "tree", 0, *args)
        if not g:
            return None
        g.G.n = n
        return g


class CompleteGraph(_NetworkXBuiltin):
    """
    <dl>
      <dt>'CompleteGraph[$n$]'
      <dd>gives the complete graph with $n$ vertices, $K_n$
    </dl>

    >> CompleteGraph[8]
     = -Graph-

    #> CompleteGraph[0]
     : Expected a positive integer at position 1 in CompleteGraph[0].
     = CompleteGraph[0]
    """

    messages = {
        "ilsmp": "Expected a positive integer at position 1 in ``.",
    }

    def apply(self, n, expression, evaluation, options):
        "%(name)s[n_Integer, OptionsPattern[%(name)s]]"
        py_n = n.get_int_value()

        if py_n < 1:
            evaluation.message(self.get_name(), "ilsmp", expression)
            return

        args = (py_n,)
        g = graph_helper(nx.complete_graph, options, False, "circular", None, *args)
        if not g:
            return None

        g.G.n = n
        return g

    def apply_multipartite(self, n, evaluation, options):
        "%(name)s[n_List, OptionsPattern[%(name)s]]"
        if all(isinstance(i, Integer) for i in n.leaves):
            return Graph(
                nx.complete_multipartite_graph(*[i.get_int_value() for i in n.leaves])
            )

class CompleteKaryTree(_NetworkXBuiltin):
    """<dl>
      <dt>'CompleteKaryTree[$n$, $k$]'
      <dd>Creates a complete $k$-ary tree of $n$ levels.
    </dl>

    In the returned tree, with $n$ nodes, the from root $R$ to any
    leaf be $k.

    >> CompleteKaryTree[2, 3]
     = -Graph-

    >> CompleteKaryTree[3]
     = -Graph-

    """

    options = DEFAULT_TREE_OPTIONS

    def apply(self, k, n, expression, evaluation, options):
        "%(name)s[n_Integer, k_Integer, OptionsPattern[%(name)s]]"

        n_int = n.get_int_value()
        k_int = k.get_int_value()

        new_n_int = int(((k_int ** n_int) - 1) / (k_int - 1))
        return f_r_t_apply(self, k, Integer(new_n_int), expression, evaluation, options)


    # FIXME: can be done with rules?
    def apply_2(self, n, expression, evaluation, options):
        "%(name)s[n_Integer, OptionsPattern[%(name)s]]"

        n_int = n.get_int_value()

        new_n_int = int(2 ** n_int) - 1
        return f_r_t_apply(self, Integer(2), Integer(new_n_int), expression, evaluation, options)


class CycleGraph(_NetworkXBuiltin):
    """<dl>
        <dt>'CycleGraph[$n$]'
        <dd>Returns the cycle graph with $n$ vertices $C_n$.
      </dl>

    >> CycleGraph[3, PlotLabel -> "C_i"]
     = -Graph-
    """

    def apply(self, n, expression, evaluation, options):
        "%(name)s[n_Integer, OptionsPattern[%(name)s]]"
        return hkn_harary_apply(self, Integer(2), n, expression, evaluation, options)


def f_r_t_apply(self, r, n, expression, evaluation, options):
    py_r = r.get_int_value()

    if py_r < 0:
        evaluation.message(self.get_name(), "ilsmp", expression)
        return

    py_n = n.get_int_value()
    if py_n < 0:
        evaluation.message(self.get_name(), "ilsmp", expression)
        return

    args = (py_r, py_n)
    g = graph_helper(nx.full_rary_tree, options, True, "tree", 0, *args)
    if not g:
        return None

    g.G.r = r
    g.G.n = n
    return g

class FullRAryTree(_NetworkXBuiltin):
    """<dl>
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
    def apply(self, r, n, expression, evaluation, options):
        "%(name)s[r_Integer, n_Integer, OptionsPattern[%(name)s]]"
        return f_r_t_apply(self, r, n, expression, evaluation, options)


class GraphAtlas(_NetworkXBuiltin):
    """<dl>
      <dt>'GraphAtlas[$n$]'
      <dd>gives graph number $i$ from the Networkx's Graph
      Atlas. There are about 1200 of them and get large as $i$
      increases.
    </dl>

    >> GraphAtlas[1000]
     = -Graph-
    """

    messages = {
        "ilsmp": "Expected a positive integer at position 1 in ``.",
    }

    def apply(self, n, expression, evaluation, options):
        "%(name)s[n_Integer, OptionsPattern[%(name)s]]"
        py_n = n.get_int_value()

        if py_n < 1:
            evaluation.message(self.get_name(), "ilsmp", expression)
            return

        args = (py_n,)
        g = graph_helper(nx.graph_atlas, options, False, "spring", None, *args)
        if not g:
            return None
        g.n = n
        return g

def hkn_harary_apply(self, k, n, expression, evaluation, options):
    py_k = k.get_int_value()

    if py_k < 0:
        evaluation.message(self.get_name(), "ilsmp", expression)
        return

    py_n = n.get_int_value()
    if py_n < 0:
        evaluation.message(self.get_name(), "ilsmp2", expression)
        return

    from pymathics.graph.harary import hkn_harary_graph

    args = (py_k, py_n)
    g = graph_helper(hkn_harary_graph, options, False, "circular", None, *args)
    if not g:
        return None
    g.k = py_k
    g.n = py_n
    return g


class HknHararyGraph(_NetworkXBuiltin):
    """<dl>
        <dt>'HmnHararyGraph[$k$, $n$]'
        <dd>Returns the Harary graph with given node connectivity and node number.

      This second generator gives the Harary graph that minimizes the
      number of edges in the graph with given node connectivity and
      number of nodes.

      Harary, F.  The Maximum Connectivity of a Graph.  Proc. Nat. Acad. Sci. USA 48, 1142-1146, 1962.
    </dl>

    >> HknHararyGraph[3, 10]
     = -Graph-

    """

    messages = {
        "ilsmp": "Expected a non-negative integer at position 1 in ``.",
        "ilsmp2": "Expected a non-negative integer at position 2 in ``.",
    }

    def apply(self, k, n, expression, evaluation, options):
        "%(name)s[k_Integer, n_Integer, OptionsPattern[%(name)s]]"
        return hkn_harary_apply(self, k, n, expression, evaluation, options)


class HmnHararyGraph(_NetworkXBuiltin):
    """<dl>
      <dt>'HmnHararyGraph[$m$, $n$]'
      <dd>Returns the Harary graph with given numbers of nodes and edges.

      This generator gives the Harary graph that maximizes the node
      connectivity with given number of nodes and given number of
      edges.

      Harary, F.  The Maximum Connectivity of a Graph.  Proc. Nat. Acad. Sci. USA 48, 1142-1146, 1962.
    </dl>

    >> HmnHararyGraph[5, 10]
     = -Graph-
    """

    messages = {
        "ilsmp": "Expected a non-negative integer at position 1 in ``.",
        "ilsmp2": "Expected a non-negative integer at position 2 in ``.",
    }

    def apply(self, n, m, expression, evaluation, options):
        "%(name)s[n_Integer, m_Integer, OptionsPattern[%(name)s]]"
        py_n = n.get_int_value()

        if py_n < 0:
            evaluation.message(self.get_name(), "ilsmp", expression)
            return

        py_m = m.get_int_value()

        if py_m < 0:
            evaluation.message(self.get_name(), "ilsmp2", expression)
            return

        from pymathics.graph.harary import hnm_harary_graph

        args = (py_n, py_m)
        g = graph_helper(hmn_harary_graph, options, False, "circular", None, *args)
        if not g:
            return None
        g.n = py_n
        g.m = py_m
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

    def apply(self, n, expression, evaluation, options):
        "%(name)s[n_Integer, OptionsPattern[%(name)s]]"
        py_n = n.get_int_value()

        if py_n < 0:
            evaluation.message(self.get_name(), "ilsmp", expression)
            return

        args = (py_n,)
        g = graph_helper(nx.random_tree, options, False, "tree", 0, *args)
        if not g:
            return None
        g.G.n = n
        return g


class StarGraph(_NetworkXBuiltin):
    """
    <dl>
      <dt>'StarGraph[$n$]'
      <dd>gives a star graph with $n$ vertices
    </dl>

    >> StarGraph[8]
     = -Graph-
    """

    messages = {
        "ilsmp": "Expected a positive integer at position 1 in ``.",
    }

    def apply(self, n, expression, evaluation, options):
        "%(name)s[n_Integer, OptionsPattern[%(name)s]]"
        py_n = n.get_int_value()

        if py_n < 1:
            evaluation.message(self.get_name(), "ilsmp", expression)
            return

        args = (py_n,)
        g = graph_helper(nx.star_graph, options, False, "spring", 0, *args)
        if not g:
            return None
        g.G.n = n
        return g
