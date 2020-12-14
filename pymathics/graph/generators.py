# -*- coding: utf-8 -*-
"""
Routines for generating classes of Graphs.

networkx does all the heavy lifting.
"""

from mathics.builtin.randomnumbers import RandomEnv
from mathics.core.expression import String

from pymathics.graph.__main__ import (
    Graph,
    WL_MARKER_TO_NETWORKX,
    _NetworkXBuiltin,
    _convert_networkx_graph,
    _graph_from_list,
    has_directed_option,
    _process_graph_options,
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
    should_digraph = can_digraph and has_directed_option(options)
    try:
        G = (
            graph_generator_func(*args, create_using=nx.DiGraph, **kwargs)
            if should_digraph
            else graph_generator_func(*args, **kwargs)
        )
    except MemoryError:
        evaluation.message(self.get_name(), "mem", expression)
        return None
    if graph_layout and not options["System`GraphLayout"].get_string_value():
        options["System`GraphLayout"] = String(graph_layout)

    g = Graph(G)
    _process_graph_options(g, options)

    if root is not None:
        G.root = g.root = root
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


# Oddly, networkX doesn't allow the directed case.
def binomial_tree(n, create_using=None):
    """Returns the Binomial Tree of order n.

    The binomial tree of order 0 consists of a single vertex. A binomial tree of order k
    is defined recursively by linking two binomial trees of order k-1: the root of one is
    the leftmost child of the root of the other.

    Parameters
    ----------
    n : int
        Order of the binomial tree.

    Returns
    -------
    G : NetworkX graph
        A binomial tree of $2^n$ vertices and $2^n - 1$ edges.

    """
    G = nx.empty_graph(1, create_using=create_using)
    N = 1
    for i in range(n):
        edges = [(u + N, v + N) for (u, v) in G.edges]
        G.add_edges_from(edges)
        G.add_edge(0, N)
        N *= 2
    return G


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
        g = graph_helper(binomial_tree, options, True, "tree", 0, *args)
        if not g:
            return None
        g.G.n = n
        return g


def complete_graph_apply(self, n, expression, evaluation, options):
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


class CompleteGraph(_NetworkXBuiltin):
    """
    <dl>
      <dt>'CompleteGraph[$n$]'
      <dd>Returns the complete graph with $n$ vertices, $K_n$
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
        return complete_graph_apply(self, n, expression, evaluation, options)

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
        return f_r_t_apply(
            self, Integer(2), Integer(new_n_int), expression, evaluation, options
        )


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
        n_int = n.get_int_value()
        if n_int < 3:
            return complete_graph_apply(self, n, expression, evaluation, options)
        else:
            return hkn_harary_apply(
                self, Integer(2), n, expression, evaluation, options
            )


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
      <dd>Returns graph number $i$ from the Networkx's Graph
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


class KaryTree(_NetworkXBuiltin):
    """<dl>
      <dt>'KaryTree[$r$, $n$]'
      <dd>Creates binary tree of $n$ vertices.
    </dl>

    <dl>
      <dt>'KaryTree[$n$, $k]'
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

    def apply(self, n, expression, evaluation, options):
        "%(name)s[n_Integer, OptionsPattern[%(name)s]]"
        return f_r_t_apply(self, Integer(2), n, expression, evaluation, options)

    def apply_2(self, n, k, expression, evaluation, options):
        "%(name)s[n_Integer, k_Integer, OptionsPattern[%(name)s]]"
        return f_r_t_apply(self, k, n, expression, evaluation, options)


class LadderGraph(_NetworkXBuiltin):
    """
    <dl>
      <dt>'LadderGraph[$n$]'
      <dd>Returns the Ladder graph of length $n$.
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
        g = graph_helper(nx.ladder_graph, options, False, "spring", 0, *args)
        if not g:
            return None
        g.G.n = n
        return g

class PathGraph(_NetworkXBuiltin):
    """
    <dl>
      <dt>'PathGraph[{$v_1$, $v_2$, ...}]'
      <dd>Returns a Graph with a path with vertices $v_i$ and edges between $v-i$ and $v_i+1$ .
    </dl>
    >> PathGraph[{1, 2, 3}]
     = -Graph-
    """

    def apply(self, l, evaluation, options):
        "PathGraph[l_List, OptionsPattern[%(name)s]]"
        leaves = l.leaves

        def edges():
            for u, v in zip(leaves, leaves[1:]):
                yield Expression("UndirectedEdge", u, v)

        g = _graph_from_list(edges(), options)
        g.G.graph_layout = options["System`GraphLayout"].get_string_value() or "spiral_equidistant"
        return g


class RandomGraph(_NetworkXBuiltin):
    """
    <dl>
      <dt>'RandomGraph[{$n$, $m$}]'
      <dd>Returns a pseudorandom graph with $n$ vertices and $m$ edges.

      <dt>'RandomGraph[{$n$, $m$}, $k$]'
      <dd>Returns list of $k$ RandomGraph[{$n$, $m$}].
    </dl>
    """

    def _generate(self, n, m, k, evaluation, options):
        py_n = n.get_int_value()
        py_m = m.get_int_value()
        py_k = k.get_int_value()
        is_directed = has_directed_option(options)

        with RandomEnv(evaluation) as rand:
            for _ in range(py_k):
                # seed = rand.randint(0, 2 ** 63 - 1) # 2**63 is too large
                G = nx.gnm_random_graph(py_n, py_m, directed=is_directed)
                yield _convert_networkx_graph(G, options)

    def apply_nm(self, n, m, expression, evaluation, options):
        "%(name)s[{n_Integer, m_Integer}, OptionsPattern[%(name)s]]"
        g = list(self._generate(n, m, Integer(1), evaluation, options))[0]
        _process_graph_options(g, options)
        return g

    def apply_nmk(self, n, m, k, expression, evaluation, options):
        "%(name)s[{n_Integer, m_Integer}, k_Integer, OptionsPattern[%(name)s]]"
        return Expression("List", *self._generate(n, m, k, evaluation, options))


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
      <dd>Returns a star graph with $n$ vertices
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

WL_TO_NETWORKX_FN = {
    "DodecahedralGraph": (nx.dodecahedral_graph, None),
    "DiamondGraph": (nx.diamond_graph, "spring"),
    "PappusGraph": (nx.pappus_graph, "circular"),
    "IsohedralGraph": (nx.icosahedral_graph, "spring"),
    "PetersenGraph": (nx.petersen_graph, None),
}

class GraphData(_NetworkXBuiltin):
    """
    <dl>
      <dt>'GraphData[$name$]'
      <dd>Returns a graph with the specified name.
    </dl>

    >> GraphData["PappusGraph"]
    """
    def apply(self, name, expression, evaluation, options):
        "%(name)s[name_String, OptionsPattern[%(name)s]]"
        py_name = name.get_string_value()
        fn, layout = WL_TO_NETWORKX_FN.get(py_name, (None, None))
        if not fn:
            if not py_name.endswith("_graph"):
                py_name += "_graph"
            if py_name in ("LCF_graph", "make_small_graph"):
                # These graphs require parameters
                return
            import inspect
            fn = dict(inspect.getmembers(nx, inspect.isfunction)).get(py_name, None)
            # parameters = inspect.signature(nx.diamond_graph).parameters.values()
            # if len([p for p in list(parameters) if p.kind in [inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD]]) != 0:
            #     return
        if fn:
            g = graph_helper(fn, options, False, layout)
            g.G.name = py_name
            return g
