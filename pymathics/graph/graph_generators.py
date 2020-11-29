from pymathics.graph.__main__ import _NetworkXBuiltin, nx, Graph
from mathics.core.expression import String

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

        try:
            G = nx.balanced_tree(py_r, py_h)
        except MemoryError:
            evaluation.message(self.get_name(), "mem", expression)
            return


        options["PlotTheme"] = options["System`PlotTheme"].get_string_value() or String("tree")
        g = Graph(G, options=options)
        g.r = r
        g.h = h
        G.root = g.root = 0
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

        options["PlotTheme"] = options["System`PlotTheme"].get_string_value() or String("spring")
        g = Graph(G, options=options)
        g.m1 = m1
        g.m2 = m2
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
    }

    def apply(self, n, expression, evaluation, options):
        "%(name)s[n_Integer, OptionsPattern[%(name)s]]"
        py_n = n.get_int_value()

        if py_n < 0:
            evaluation.message(self.get_name(), "ilsmp", expression)
            return

        G = nx.binomial_tree(py_n)

        options["PlotTheme"] = options["System`PlotTheme"].get_string_value() or String("tree")
        g = Graph(G, options=options)
        g.n = n
        G.root = g.root = 0
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

        options["PlotTheme"] = options["System`PlotTheme"].get_string_value() or String("circular")
        g = Graph(G, options=options)
        g.n  = n
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
    }

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

        G = nx.full_rary_tree(py_r, py_n)

        options["PlotTheme"] = options["System`PlotTheme"].get_string_value() or String("tree")
        g = Graph(G, options=options)
        g.r = r
        g.n = n
        G.root = g.root = 0
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

        options["PlotTheme"] = options["System`PlotTheme"].get_string_value() or String("tree")
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

        G = nx.star_graph(py_n)

        options["PlotTheme"] = options["System`PlotTheme"].get_string_value() or String("spring")

        g = Graph(G, options=options)
        g.n = n
        return g
