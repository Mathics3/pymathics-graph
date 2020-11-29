from pymathics.graph.__main__ import _NetworkXBuiltin, nx, Graph
from mathics.core.expression import String

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

        return Graph(G)

    def apply_multipartite(self, n, evaluation, options):
        "%(name)s[n_List, OptionsPattern[%(name)s]]"
        if all(isinstance(i, Integer) for i in n.leaves):
            return Graph(
                nx.complete_multipartite_graph(*[i.get_int_value() for i in n.leaves])
            )


class BalancedTree(_NetworkXBuiltin):
    """
    <dl>
      <dt>'BalancedTree[$r$, $h$]'
      <dd>Returns the perfectly balanced r-ary tree of height h.
    </dl>

    >> BalancedTree[2, 3]
     = -Graph-

    """

    messages = {
        "ilsmp": "Expected a non-negative integer at position 1 in ``.",
        "ilsmp2": "Expected a non-negative integer at position 2 in ``.",
    }

    def apply(self, r, h, expression, evaluation, options):
        "%(name)s[r_Integer, h_Integer, OptionsPattern[%(name)s]]"
        py_r = r.get_int_value()

        if py_r < 0:
            evaluation.message(self.get_name(), "ilsmp", expression)
            return

        py_h = h.get_int_value()
        if py_h < 0:
            evaluation.message(self.get_name(), "ilsmp", expression)
            return

        G = nx.balanced_tree(py_r, py_h)

        options["PlotTheme"] = options["System`PlotTheme"].get_string_value() or String("tree")
        return Graph(G, options=options)


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
        return Graph(G)
