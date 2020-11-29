from pymathics.graph.__main__ import (
    DEFAULT_TREE_OPTIONS,
    Graph,
    _NetworkXBuiltin,
    nx,
)
from mathics.core.expression import String

# TODO: this code can be DRY'd a bit.


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
            return

        py_h = h.get_int_value()
        if py_h < 0:
            evaluation.message(self.get_name(), "ilsmp2", expression)
            return

        graph_create = nx.DiGraph if options["System`Directed"].to_python() else nx.Graph

        try:
            G = nx.balanced_tree(py_r, py_h, create_using=graph_create)
        except MemoryError:
            evaluation.message(self.get_name(), "mem", expression)
            return

        options["GraphLayout"] = options["System`GraphLayout"].get_string_value() or String(
            "tree"
        )
        options["VertexLabeling"] = options["System`VertexLabeling"]
        g = Graph(G, options=options)

        g.r = r
        g.h = h
        G.root = g.root = 0
        G.title = g.title = options["System`PlotLabel"]

        return g


class BarbellGraph(_NetworkXBuiltin):
    """
    <dl>
      <dt>'BarbellGraph[$m1$, $m2$]'
      <dd>Barbell Graph: two complete graphs connected by a path.
    </dl>

    >> BarBellGraph[2, 3]
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

        G = nx.barbell_graph(py_m1, py_m2)

        options["GraphLayout"] = options["System`GraphLayout"].get_string_value() or String(
            "spring"
        )
        options["VertexLabeling"] = options["System`VertexLabeling"]
        g = Graph(G, options=options)
        g.m1 = m1
        g.m2 = m2
        G.title = g.title = options["System`PlotLabel"]
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

        try:
            G = nx.binomial_tree(py_n)
        except MemoryError:
            evaluation.message(self.get_name(), "mem", expression)
            return

        options["GraphLayout"] = options["System`GraphLayout"].get_string_value() or String(
            "tree"
        )
        options["VertexLabeling"] = options["System`VertexLabeling"]
        g = Graph(G, options=options)
        g.n = n
        G.root = g.root = 0
        G.title = g.title = options["System`PlotLabel"]
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

        G = nx.complete_graph(py_n)

        options["GraphLayout"] = options["System`GraphLayout"].get_string_value() or String(
            "circular"
        )
        options["VertexLabeling"] = options["System`VertexLabeling"]
        g = Graph(G, options=options)
        g.n = n
        G.title = g.title = options["System`PlotLabel"]
        return g

    def apply_multipartite(self, n, evaluation, options):
        "%(name)s[n_List, OptionsPattern[%(name)s]]"
        if all(isinstance(i, Integer) for i in n.leaves):
            return Graph(
                nx.complete_multipartite_graph(*[i.get_int_value() for i in n.leaves])
            )


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
        py_r = r.get_int_value()

        if py_r < 0:
            evaluation.message(self.get_name(), "ilsmp", expression)
            return

        py_n = n.get_int_value()
        if py_n < 0:
            evaluation.message(self.get_name(), "ilsmp", expression)
            return

        graph_create = nx.DiGraph if options["System`Directed"].to_python() else nx.Graph

        try:
            G = nx.full_rary_tree(py_r, py_n, create_using=graph_create)
        except MemoryError:
            evaluation.message(self.get_name(), "mem", expression)
            return

        options["GraphLayout"] = options["System`GraphLayout"].get_string_value() or String(
            "tree"
        )
        options["VertexLabeling"] = options["System`VertexLabeling"]
        g = Graph(G, options=options)
        g.r = r
        g.n = n
        G.root = g.root = 0
        G.title = g.title = options["System`PlotLabel"]
        return g


class GraphAtlas(_NetworkXBuiltin):
    """
    <dl>
      <dt>'GraphAtlas[$n$]'
      <dd>gives graph number $i$ from the Networkx's Graph Atlas. There are about 1200 of them.
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

        G = nx.graph_atlas(py_n)
        g = Graph(G)
        g.n = n
        G.title = g.title = options["System`PlotLabel"]
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
        py_k = k.get_int_value()

        if py_k < 0:
            evaluation.message(self.get_name(), "ilsmp", expression)
            return

        py_n = n.get_int_value()
        if py_n < 0:
            evaluation.message(self.get_name(), "ilsmp2", expression)
            return

        from pymathics.graph.harary import hkn_harary_graph

        G = hkn_harary_graph(py_k, py_n)

        options["GraphLayout"] = options["System`GraphLayout"].get_string_value() or String(
            "spring"
        )
        options["VertexLabeling"] = options["System`VertexLabeling"]
        g = Graph(G, options=options)
        g.n = n
        G.root = g.root = 0
        G.title = g.title = options["System`PlotLabel"]
        return g


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

        G = hnm_harary_graph(py_n, py_m)

        options["GraphLayout"] = options["System`GraphLayout"].get_string_value() or String(
            "circular"
        )
        options["VertexLabeling"] = options["System`VertexLabeling"]
        g = Graph(G, options=options)
        g.n = n
        G.root = g.root = 0
        G.title = g.title = options["System`PlotLabel"]
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

        G = nx.random_tree(py_n)

        options["GraphLayout"] = options["System`GraphLayout"].get_string_value() or String(
            "circular"
        )
        options["VertexLabeling"] = options["System`VertexLabeling"]
        g = Graph(G, options=options)
        g.n = n
        G.root = g.root = 0
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

        try:
            G = nx.star_graph(py_n)
        except MemoryError:
            evaluation.message(self.get_name(), "mem", expression)
            return

        options["GraphLayout"] = options["System`GraphLayout"].get_string_value() or String(
            "spring"
        )
        options["VertexLabeling"] = options["System`VertexLabeling"]

        g = Graph(G, options=options)
        g.n = n
        G.title = g.title = options["System`PlotLabel"]
        return g
