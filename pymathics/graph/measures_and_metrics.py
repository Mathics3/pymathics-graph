"""
Graph Measures and Metrics

Measures include basic measures, such as the number of vertices and edges, \
connectivity, degree measures, centrality, and so on.
"""

from typing import Optional

import networkx as nx
from mathics.core.atoms import Integer, Integer1, Integer2, Integer3
from mathics.core.convert.expression import ListExpression, to_mathics_list
from mathics.core.evaluation import Evaluation
from mathics.core.expression import Expression, from_python
from mathics.core.systemsymbols import SymbolCases, SymbolDirectedInfinity, SymbolLength

from pymathics.graph.base import _NetworkXBuiltin

# FIXME put this in its own file/module basic
# when pymathics doc can handle this.
# """
# Basic Graph Measures
# """


class _PatternCount(_NetworkXBuiltin):
    """
    Counts of vertices or edges, allowing rules to specify the graph.
    """

    no_doc = True

    def eval(self, graph, expression, evaluation, options) -> Optional[Integer]:
        "%(name)s[graph_, OptionsPattern[%(name)s]]"
        graph = self._build_graph(graph, evaluation, options, expression)
        if graph:
            return Integer(len(self._items(graph)))

    def eval_patt(
        self, graph, patt, expression, evaluation, options
    ) -> Optional[Expression]:
        "%(name)s[graph_, patt_, OptionsPattern[%(name)s]]"
        graph = self._build_graph(graph, evaluation, options, expression)
        if graph:
            return Expression(
                SymbolLength,
                Expression(
                    SymbolCases,
                    ListExpression(*(from_python(item) for item in self._items(graph))),
                    patt,
                ),
            )


class EdgeCount(_PatternCount):
    """
    <url>
    :NetworkX:
    https://networkx.org/documentation/latest/reference/generated/networkx.classes.function.edges.html#edges</url>, <url>
    :WMA:
    https://reference.wolfram.com/language/ref/EdgeCount.html</url>

    <dl>
       <dt>'EdgeCount[$g$]'
       <dd>returns a count of the number of edges in graph $g$.

       <dt>'EdgeCount[$g$, $patt$]'
       <dd>returns the number of edges that match the pattern $patt$.

       <dt>'EdgeCount[{$v$->$w}, ...}, ...]'
       <dd>uses rules $v$->$w$ to specify the graph $g$.
    </dl>

    >> EdgeCount[{1 -> 2, 2 -> 3}]
     = 2
    """

    no_doc = False
    summary_text = "count edges in graph"

    def _items(self, graph):
        return graph.G.edges


class GraphDistance(_NetworkXBuiltin):
    """
    <url>
    :NetworkX:
    https://networkx.org/documentation/latest/reference/algorithms/generated/networkx.algorithms.shortest_paths.generic.shortest_path_length.html</url>, <url>
    :WMA:
    https://reference.wolfram.com/language/ref/GraphDistance.html</url>

    <dl>
      <dt>'GraphDistance'[$g$, $s$, $t$]
      <dd>returns the distance from source vertex $s$ to target vertex $t$ in the graph $g$.
    </dl>

    <dl>
      <dt>'GraphDistance[$g$, $s$]'
      <dd>returns the distance from source vertex $s$ to all vertices in the graph $g$.
    </dl>

    <dl>
      <dt>'GraphDistance[{$v$->$w$, ...}, ...]'
      <dd>use rules $v$->$w$ to specify the graph $g$.
    </dl>

    >> g = Graph[{1 -> 2, 2 <-> 3, 4 -> 3, 2 <-> 4, 4 -> 5}, VertexLabels->True]
     = -Graph-

    >> GraphDistance[g, 1, 5]
     = 3

    >> GraphDistance[g, 4, 2]
     = 1

    >> GraphDistance[g, 5, 4]
     = Infinity

    >> GraphDistance[g, 5]
     = {Infinity, Infinity, Infinity, Infinity, 0}

    >> GraphDistance[g, 3]
     = {Infinity, 1, 2, 0, 3}

    >> GraphDistance[g, 4]
     = {Infinity, 1, 0, 1, 1}
    """

    summary_text = "get path distance"

    def eval_s(
        self, graph, s, expression, evaluation: Evaluation, options: dict
    ) -> Optional[ListExpression]:
        "GraphDistance[graph_, s_, OptionsPattern[GraphDistance]]"
        graph = self._build_graph(graph, evaluation, options, expression)
        if graph:
            weight = graph.update_weights(evaluation)
            d = nx.shortest_path_length(graph.G, source=s, weight=weight)
            inf = Expression(SymbolDirectedInfinity, Integer1)
            return to_mathics_list(*[d.get(v, inf) for v in graph.vertices])

    def eval_s_t(self, graph, s, t, expression, evaluation: Evaluation, options: dict):
        "GraphDistance[graph_, s_, t_, OptionsPattern[GraphDistance]]"
        graph = self._build_graph(graph, evaluation, options, expression)
        if not graph:
            return
        G = graph.G
        if not G.has_node(s):
            self._not_a_vertex(expression, Integer2, evaluation)
        elif not G.has_node(t):
            self._not_a_vertex(expression, Integer3, evaluation)
        else:
            try:
                weight = graph.update_weights(evaluation)
                return from_python(
                    nx.shortest_path_length(graph.G, source=s, target=t, weight=weight)
                )
            except nx.exception.NetworkXNoPath:
                return Expression(SymbolDirectedInfinity, Integer1)


class VertexCount(_PatternCount):
    """
    <url>
    :NetworkX:
    https://networkx.org/documentation/latest/reference/generated/networkx.classes.function.nodes.html</url>, <url>
    :WMA:
    https://reference.wolfram.com/language/ref/VertexCount.html</url>

    <dl>
       <dt>'VertexCount[$g$]'
       <dd>returns a count of the number of vertices in graph $g$.

       <dt>'VertexCount[$g$, $patt$]'
       <dd>returns the number of vertices that match the pattern $patt$.

       <dt>'VertexCount[{$v$->$w}, ...}, ...]'
       <dd>uses rules $v$->$w$ to specify the graph $g$.
    </dl>

    >> VertexCount[{1 -> 2, 2 -> 3}]
     = 3

    >> VertexCount[{1 -> x, x -> 3}, _Integer]
     = 2
    """

    no_doc = False
    summary_text = "count vertices in graph"

    def _items(self, graph):
        return graph.G.nodes


# Put this in its own file/module "degree.py"
# when pymathics doc can handle.
# """
# Graph Degree Measures
# """


class VertexDegree(_NetworkXBuiltin):
    """
    <url>
    :NetworkX:
    https://networkx.org/documentation/latest/reference/classes/generated/networkx.Graph.degree.html</url>, <url>
    :WMA:
    https://reference.wolfram.com/language/ref/VertexDegree.html</url>

    <dl>
       <dt>'VertexDegree[$g$]'
       <dd>returns a list of the degrees of each of the vertices in graph $g$.

       <dt>'EdgeCount[$g$, $patt$]'
       <dd>returns the number of edges that match the pattern $patt$.

       <dt>'EdgeCount[{$v$->$w}, ...}, ...]'
       <dd>uses rules $v$->$w$ to specify the graph $g$.
    </dl>

    >> VertexDegree[{1 <-> 2, 2 <-> 3, 2 <-> 4}]
     = {1, 3, 1, 1}
    """

    no_doc = False
    summary_text = "list graph vertex degrees"

    def eval(self, graph, evaluation, options):
        "%(name)s[graph_, OptionsPattern[%(name)s]]"

        def degrees(graph):
            degrees = dict(list(graph.G.degree(graph.vertices)))
            return ListExpression(*[Integer(degrees.get(v, 0)) for v in graph.vertices])

        return self._evaluate_atom(graph, options, degrees)


# TODO: VertexInDegree, VertexOutDegree
