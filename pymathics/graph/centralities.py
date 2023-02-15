# -*- coding: utf-8 -*-

"""
Centralities

<url>:Centralities:https://en.wikipedia.org/wiki/Centrality</url>


Routines to evaluate centralities of a graph.

In graph theory and network analysis, the centrality is a \
ranking between pairs of node according some metric.

"""

import networkx as nx
from mathics.core.atoms import Integer, Real
from mathics.core.convert.expression import ListExpression

from pymathics.graph.base import _NetworkXBuiltin


def _components(G):
    if G.is_directed():
        return nx.strongly_connected_components(G)
    else:
        return nx.connected_components(G)


class _Centrality(_NetworkXBuiltin):
    options = {
        "WorkingPrecision": "MachinePrecision",
    }
    pass


class _ComponentwiseCentrality(_Centrality):
    def _centrality(self, g, weight):
        raise NotImplementedError

    def _compute(self, graph, evaluation, reverse=False, normalized=True, **kwargs):
        vertices = graph.vertices
        G, weight = graph.coalesced_graph(evaluation)
        if reverse:
            G = G.reverse()

        components = list(_components(G))
        components = [c for c in components if len(c) > 1]

        result = [0] * len(vertices)
        for bunch in components:
            g = G.subgraph(bunch)
            centrality = self._centrality(g, weight, **kwargs)
            values = [centrality.get(v, 0) for v in vertices]
            if normalized:
                s = sum(values) * len(components)
            else:
                s = 1
            if s > 0:
                for i, x in enumerate(values):
                    result[i] += x / s

        return ListExpression(*[Real(x) for x in result])


class BetweennessCentrality(_Centrality):
    """
    <url>
    :Betweenness centrality:
    https://en.wikipedia.org/wiki/Betweenness_centrality</url> (<url>
    :NetworkX:
    https://networkx.org/documentation/networkx-2.8.8/reference/algorithms/generated/networkx.algorithms.centrality.betweenness_centrality.html</url>,\
    <url>
    :WMA:
    https://reference.wolfram.com/language/ref/BetweennessCentrality.html</url>)

    <dl>
      <dt>'BetweennessCentrality'[$g$]
      <dd>gives a list of betweenness centralities for the vertices \
          in a 'Graph' or a list of edges $g$.
    </dl>

    >> g = Graph[{a -> b, b -> c, d -> c, d -> a, e -> c, d -> b}]
     = -Graph-

    >> BetweennessCentrality[g]
     = {0., 1., 0., 0., 0.}

    >> g = Graph[{a -> b, b -> c, c -> d, d -> e, e -> c, e -> a}]
     = -Graph-

    >> BetweennessCentrality[g]
     = ...
    """

    summary_text = "get Betweenness centrality"

    def eval(self, graph, expression, evaluation, options):
        "%(name)s[graph_, OptionsPattern[%(name)s]]"
        graph = self._build_graph(graph, evaluation, options, expression)
        if graph:
            weight = graph.update_weights(evaluation)
            centrality = nx.betweenness_centrality(
                graph.G, normalized=False, weight=weight
            )
            return ListExpression(
                *[Real(centrality.get(v, 0.0)) for v in graph.vertices],
            )


class ClosenessCentrality(_Centrality):
    """
    <url>
    :Betweenness centrality:
    https://en.wikipedia.org/wiki/Closeness_centrality</url> (<url>
    :NetworkX:
    https://networkx.org/documentation/networkx-2.8.8/reference/algorithms/generated/\
    networkx.algorithms.centrality.closeness_centrality.html</url>,\
    <url>
    :WMA:
    https://reference.wolfram.com/language/ref/ClosenessCentrality.html</url>)


    <dl>
      <dt>'ClosenessCentrality'[$g$]
      <dd>gives a list of closeness centralities for the vertices \
          in a 'Graph' or a list of edges $g$.
    </dl>

     >> g = Graph[{a -> b, b -> c, d -> c, d -> a, e -> c, d -> b}]
      = -Graph-

     >> ClosenessCentrality[g]
      = {0.666667, 1., 0., 1., 1.}

     >> g = Graph[{a -> b, b -> c, c -> d, d -> e, e -> c, e -> a}]
      = -Graph-

    >> ClosenessCentrality[g]
      = {0.4, 0.4, 0.4, 0.5, 0.666667}
    """

    summary_text = "get the closeness centrality"

    def eval(self, graph, expression, evaluation, options):
        "%(name)s[graph_, OptionsPattern[%(name)s]]"
        graph = self._build_graph(graph, evaluation, options, expression)
        if graph:
            weight = graph.update_weights(evaluation)
            G = graph.G
            if G.is_directed():
                G = G.reverse()
            centrality = nx.closeness_centrality(G, distance=weight, wf_improved=False)
            return ListExpression(
                *[Real(centrality.get(v, 0.0)) for v in graph.vertices],
            )


class DegreeCentrality(_Centrality):
    """
    <url>
    :Degree centrality:
    https://en.wikipedia.org/wiki/Degree_centrality</url> (<url>
    :NetworkX:
    https://networkx.org/documentation/networkx-2.8.8/reference/algorithms/generated/\
    networkx.algorithms.centrality.degree_centrality.html</url>,\
    <url>
    :WMA:
    https://reference.wolfram.com/language/ref/DegreeCentrality.html</url>)

    <dl>
      <dt>'DegreeCentrality'[$g$]
      <dd>gives a list of degree centralities for the vertices \
          in a 'Graph' or a list of edges $g$.
    </dl>

    >> g = Graph[{a -> b, b <-> c, d -> c, d -> a, e <-> c, d -> b}]
     = -Graph-

    >> DegreeCentrality[g]
     = ...

    >> DegreeCentrality[g, "In"]
     = ...

    >> DegreeCentrality[g, "Out"]
     = ...
    """

    summary_text = "get the degree centrality"

    def _from_dict(self, graph, centrality):
        s = len(graph.G) - 1  # undo networkx's normalization
        return ListExpression(
            *[Integer(s * centrality.get(v, 0)) for v in graph.vertices],
        )

    def eval(self, graph, expression, evaluation, options):
        "Pymathics`DegreeCentrality[graph_, OptionsPattern[]]"
        graph = self._build_graph(graph, evaluation, options, expression)
        if graph:
            return self._from_dict(graph, nx.degree_centrality(graph.G))

    def eval_in(self, graph, expression, evaluation, options):
        '%(name)s[graph_, "In", OptionsPattern[]]'
        graph = self._build_graph(graph, evaluation, options, expression)
        if graph:
            return self._from_dict(graph, nx.in_degree_centrality(graph.G))

    def eval_out(self, graph, expression, evaluation, options):
        '%(name)s[graph_, "Out", OptionsPattern[]]'
        graph = self._build_graph(graph, evaluation, options, expression)
        if graph:
            return self._from_dict(graph, nx.out_degree_centrality(graph.G))


class EigenvectorCentrality(_ComponentwiseCentrality):
    """
    <url>
    :Eigenvector Centrality:
    https://en.wikipedia.org/wiki/Eigenvector_centrality</url> (<url>
    :NetworkX:
    https://networkx.org/documentation/networkx-2.8.8/reference/algorithms\
/generated/networkx.algorithms.centrality.eigenvector_centrality.html</url>,\
<url>
    :WMA:
    https://reference.wolfram.com/language/ref/EgenvectorCentrality.html</url>)

    <dl>
      <dt>'EigenvectorCentrality'[$g$]
      <dd>gives a list of eigenvector centralities for\
          the vertices in the graph $g$.
      <dt>'EigenvectorCentrality'[$g$, "In"]
      <dd>gives a list of eigenvector in-centralities for\
          the vertices in the graph $g$.
      <dt>'EigenvectorCentrality'[$g$, "Out"]
      <dd>gives a list of eigenvector out-centralities for\
          the vertices in the graph $g$.
    </dl>

    >> g = Graph[{a -> b, b -> c, c -> d, d -> e, e -> c, e -> a}]; EigenvectorCentrality[g, "In"]
     = {0.16238, 0.136013, 0.276307, 0.23144, 0.193859}

    >> EigenvectorCentrality[g, "Out"]
     = {0.136013, 0.16238, 0.193859, 0.23144, 0.276307}

    >> g = Graph[{a <-> b, b <-> c, c <-> d, d <-> e, e <-> c, e <-> a}]; EigenvectorCentrality[g]
     = {0.162435, 0.162435, 0.240597, 0.193937, 0.240597}

    >> g = Graph[{a <-> b, b <-> c, a <-> c, d <-> e, e <-> f, f <-> d, e <-> d}]; EigenvectorCentrality[g]
     = {0.166667, 0.166667, 0.166667, 0.183013, 0.183013, 0.133975}

    #> g = Graph[{a -> b, b -> c, c -> d, b -> e, a -> e}]; EigenvectorCentrality[g]
     = {0., 0., 0., 0., 0.}

    >> g = Graph[{a -> b, b -> c, c -> d, b -> e, a -> e, c -> a}]; EigenvectorCentrality[g]
     = {0.333333, 0.333333, 0.333333, 0., 0.}
    """

    summary_text = "compute Eigenvector centralities"

    def _centrality(self, g, weight):
        return nx.eigenvector_centrality(g, max_iter=10000, tol=1.0e-7, weight=weight)

    def eval(self, graph, expression, evaluation, options):
        "%(name)s[graph_, OptionsPattern[%(name)s]]"
        graph = self._build_graph(graph, evaluation, options, expression)
        if graph:
            return self._compute(graph, evaluation)

    def eval_in_out(self, graph, dir, expression, evaluation, options):
        "%(name)s[graph_, dir_String, OptionsPattern[%(name)s]]"
        py_dir = dir.get_string_value()
        if py_dir not in ("In", "Out"):
            return
        graph = self._build_graph(graph, evaluation, options, expression)
        if graph:
            return self._compute(graph, evaluation, py_dir == "Out")


class HITSCentrality(_Centrality):
    """
    <url>
    :NetworkX:
    https://networkx.org/documentation/networkx-2.8.8/reference/algorithms/\
generated/networkx.algorithms.link_analysis.hits_alg.hits.html</url>, \
    <url>
    :WMA:
    https://reference.wolfram.com/language/ref/HITSCentrality.html</url>

    <dl>
      <dt>'HITSCentrality'[$g$]
      <dd>gives a list of authority and hub centralities for\
          the vertices in the graph $g$.
    </dl>

    """

    summary_text = "get HITS centrality"

    def eval(self, graph, expression, evaluation, options):
        "%(name)s[graph_, OptionsPattern[%(name)s]]"
        graph = self._build_graph(graph, evaluation, options, expression)
        if graph:
            G, _ = graph.coalesced_graph(evaluation)  # FIXME warn if weight > 1

            tol = 1.0e-14
            _, a = nx.hits(G, normalized=True, tol=tol)
            h, _ = nx.hits(G, normalized=False, tol=tol)

            def _crop(x):
                return 0 if x < tol else x

            vertices = graph.vertices
            return ListExpression(
                ListExpression(*[Real(_crop(a.get(v, 0))) for v in vertices]),
                ListExpression(*[Real(_crop(h.get(v, 0))) for v in vertices]),
            )


class KatzCentrality(_ComponentwiseCentrality):
    """
    <url>
    :Katz Centrality:
    https://en.wikipedia.org/wiki/Katz_centrality</url> (<url>
    :NetworkX:
    https://networkx.org/documentation/networkx-2.8.8/reference/algorithms\
/generated/networkx.algorithms.centrality.katz_centrality.html\
#networkx.algorithms.centrality.katz_centrality</url>, \
    <url>
    :WMA:
    https://reference.wolfram.com/language/ref/KatzCentrality.html</url>)

    <dl>
      <dt>'KatzCentrality'[$g$, $alpha$]
      <dd>gives a list of Katz centralities for the \
          vertices in the graph $g$ and weight $alpha$.
      <dt>'KatzCentrality'[$g$, $alpha$, $beta$]
      <dd>gives a list of Katz centralities for the \
          vertices in the graph $g$ and weight $alpha$ and initial centralities $beta$.
    </dl>

    >> g = Graph[{a -> b, b -> c, c -> d, d -> e, e -> c, e -> a}]
     = -Graph-
    >> KatzCentrality[g, 0.2]
     = {1.25202, 1.2504, 1.5021, 1.30042, 1.26008}

    >> g = Graph[{a <-> b, b <-> c, a <-> c, d <-> e, e <-> f, f <-> d, e <-> d}]
     = -Graph-

    >> KatzCentrality[g, 0.1]
     = {1.25, 1.25, 1.25, 1.41026, 1.41026, 1.28205}
    """

    summary_text = "get the Katz centrality"

    rules = {
        "Pymathics`KatzCentrality[Pymathics`g_, Pymathics`alpha_]": "Pymathics`KatzCentrality[Pymathics`g, Pymathics`alpha, 1]",
    }

    def _centrality(self, g, weight, alpha, beta):
        return nx.katz_centrality(
            g, alpha=alpha, beta=beta, normalized=False, weight=weight
        )

    def eval(self, graph, alpha, beta, expression, evaluation, options):
        "Pymathics`KatzCentrality[Pymathics`graph_, alpha_, beta_, OptionsPattern[%(name)s]]"
        graph = self._build_graph(graph, evaluation, options, expression)
        if graph:
            try:
                py_alpha = alpha.to_mpmath()
                py_beta = beta.to_mpmath()
            except NotImplementedError:
                return
            if py_alpha is None or py_beta is None:
                return
            return self._compute(
                graph, evaluation, normalized=False, alpha=py_alpha, beta=py_beta
            )


class PageRankCentrality(_Centrality):
    """
    <url>
    :Pagerank Centrality:
    https://en.wikipedia.org/wiki/Pagerank</url> (<url>
    :NetworkX:
    https://networkx.org/documentation/networkx-2.8.8/reference/algorithms\
    /generated/networkx.algorithms.link_analysis.pagerank_alg.pagerank.html</url>,\
    <url>
    :WMA:
    https://reference.wolfram.com/language/ref/PageRankCentrality.html</url>)

    <dl>
      <dt>'PageRankCentrality'[$g$, $alpha$]
      <dd>gives a list of page rank centralities for the \
          vertices in the graph $g$ and weight $alpha$.
      <dt>'PageRankCentrality'[$g$, $alpha$, $beta$]
      <dd>gives a list of page rank centralities for the \
          vertices in the graph $g$ and weight $alpha$ and initial centralities $beta$.
    </dl>

    >> g = Graph[{a -> d, b -> c, d -> c, d -> a, e -> c, d -> c}]; PageRankCentrality[g, 0.2]
     = {0.184502, 0.207565, 0.170664, 0.266605, 0.170664}
    """

    summary_text = "get the page rank centralities"

    def eval_alpha_beta(self, graph, alpha, expression, evaluation, options):
        "%(name)s[graph_, alpha_, OptionsPattern[%(name)s]]"
        graph = self._build_graph(graph, evaluation, options, expression)
        if graph:
            py_alpha = float(alpha.to_mpmath())
            if py_alpha is None:
                return
            G, weight = graph.coalesced_graph(evaluation)
            centrality = nx.pagerank(G, alpha=py_alpha, weight=weight, tol=1.0e-7)
            return ListExpression(
                *[Real(centrality.get(v, 0)) for v in graph.vertices],
            )
