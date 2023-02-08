"""
Graph Measures and Metrics

Measures include basic measures, such as the number of vertices and edges, \
connectivity, degree measures, centrality, and so on.
"""


from typing import Optional

from mathics.core.atoms import Integer
from mathics.core.convert.expression import ListExpression
from mathics.core.expression import Expression
from mathics.core.symbols import Symbol
from mathics.core.systemsymbols import SymbolLength

from pymathics.graph.base import _NetworkXBuiltin

# FIXME: add context
SymbolCases = Symbol("Cases")


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
                Expression(SymbolCases, ListExpression(*self._items(graph)), patt),
            )


class EdgeCount(_PatternCount):
    """
    <url>
    :WMA link:
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


class VertexCount(_PatternCount):
    """
    <url>
    :WMA link:
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
    :WMA link:
    https://reference.wolfram.com/language/ref/EdgeCount.html</url>

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
