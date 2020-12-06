# -*- coding: utf-8 -*-
"""
Algorithms on Graphs.

networkx does all the heavy lifting.
"""

from mathics.core.expression import Expression

from pymathics.graph.__main__ import (
    _NetworkXBuiltin,
    nx,
)

from mathics.core.expression import (
    from_python,
)



class GraphDistance(_NetworkXBuiltin):
    """
    <dl>
      <dt>'GraphDistance[$g$, $s$, $t$]'
      <dd>returns the distance from source vertex $s$ to target vertex $t$ in the graph $g$.
    </dl>

    <dl>
      <dt>'GraphDistance[$g$, $s$]'
      <dd>returns the distance from source vertex $s$ to all vertices in the graph $g$.
    </dl>

    <dl>
      <dt>'GraphDistance[{$v$->$w$, ...}, ...]'
      <dd>use rules $v$->$w$ to specify the graph $g$
    </dl>

    >> GraphDistance[{1 <-> 2, 2 <-> 3, 3 <-> 4, 2 <-> 4, 4 -> 5}, 1, 5]
     = 3

    >> GraphDistance[{1 <-> 2, 2 <-> 3, 3 <-> 4, 4 -> 2, 4 -> 5}, 1, 5]
     = 4

    >> GraphDistance[{1 <-> 2, 2 <-> 3, 4 -> 3, 4 -> 2, 4 -> 5}, 1, 5]
     = Infinity

    >> GraphDistance[{1 <-> 2, 2 <-> 3, 3 <-> 4, 2 <-> 4, 4 -> 5}, 3]
     = {2, 1, 0, 1, 2}

    >> GraphDistance[{1 <-> 2, 3 <-> 4}, 3]
     = {Infinity, Infinity, 0, 1}

    #> GraphDistance[{}, 1, 1]
     : The vertex at position 2 in GraphDistance[{}, 1, 1] does not belong to the graph at position 1.
     = GraphDistance[{}, 1, 1]
    #> GraphDistance[{1 -> 2}, 3, 4]
     : The vertex at position 2 in GraphDistance[{1 -> 2}, 3, 4] does not belong to the graph at position 1.
     = GraphDistance[{1 -> 2}, 3, 4]
    """

    def apply_s(self, graph, s, expression, evaluation, options):
        "%(name)s[graph_, s_, OptionsPattern[%(name)s]]"
        graph = self._build_graph(graph, evaluation, options, expression)
        if graph:
            weight = graph.update_weights(evaluation)
            d = nx.shortest_path_length(graph.G, source=s, weight=weight)
            inf = Expression("DirectedInfinity", 1)
            return Expression(
                "List", *[d.get(v, inf) for v in graph.vertices]
            )

    def apply_s_t(self, graph, s, t, expression, evaluation, options):
        "%(name)s[graph_, s_, t_, OptionsPattern[%(name)s]]"
        graph = self._build_graph(graph, evaluation, options, expression)
        if not graph:
            return
        G = graph.G
        if not G.has_node(s):
            self._not_a_vertex(expression, 2, evaluation)
        elif not G.has_node(t):
            self._not_a_vertex(expression, 3, evaluation)
        else:
            try:
                weight = graph.update_weights(evaluation)
                return from_python(
                    nx.shortest_path_length(graph.G, source=s, target=t, weight=weight)
                )
            except nx.exception.NetworkXNoPath:
                return Expression("DirectedInfinity", 1)
