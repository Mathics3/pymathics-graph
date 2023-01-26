"""
Basic Graph Measures
"""

from typing import Optional

from mathics.core.atoms import Integer
from mathics.core.convert.expression import ListExpression
from mathics.core.expression import Expression
from mathics.core.symbols import Symbol

from pymathics.graph.base import _NetworkXBuiltin

# FIXME: add context
SymbolLength = Symbol("Length")
SymbolCases = Symbol("Cases")


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

    summary_text = "count vertices in graph"

    def _items(self, graph):
        return graph.G.nodes
