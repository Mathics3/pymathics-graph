"""
Graph Degree Measures
"""

from mathics.core.atoms import Integer
from mathics.core.convert.expression import ListExpression
from pymathics.graph.base import _NetworkXBuiltin


class VertexDegree(_NetworkXBuiltin):
    """
    >> VertexDegree[{1 <-> 2, 2 <-> 3, 2 <-> 4}]
     = {1, 3, 1, 1}
    """

    def eval(self, graph, evaluation, options):
        "%(name)s[graph_, OptionsPattern[%(name)s]]"

        def degrees(graph):
            degrees = dict(list(graph.G.degree(graph.vertices)))
            return ListExpression(*[Integer(degrees.get(v, 0)) for v in graph.vertices])

        return self._evaluate_atom(graph, options, degrees)


# TODO: VertexInDegree, VertexOutDegree
