# -*- coding: utf-8 -*-

"""
Collection classes
"""


from pymathics.graph.graphsymbols import SymbolDirectedEdge

# There is no user-facing documentation here.
no_doc = True


def _count_edges(counts, edges, sign):
    n_directed, n_undirected = counts
    for edge in edges:
        if edge.head is SymbolDirectedEdge:
            n_directed += sign
        else:
            n_undirected += sign
    return n_directed, n_undirected


class _Collection:
    def __init__(self, expressions, properties=None, index=None):
        self.expressions = expressions
        self.properties = properties if properties else None
        self.index = index

    def clone(self):
        properties = self.properties
        return _Collection(
            self.expressions[:], properties[:] if properties else None, None
        )

    def filter(self, expressions):
        index = self.get_index()
        return [expr for expr in expressions if expr in index]

    def extend(self, expressions, properties):
        if properties:
            if self.properties is None:
                self.properties = [None] * len(self.expressions)
            self.properties.extend(properties)
        self.expressions.extend(expressions)
        self.index = None
        return expressions

    def delete(self, expressions):
        index = self.get_index()
        trash = set(index[x] for x in expressions)
        deleted = [self.expressions[i] for i in trash]
        self.expressions = [x for i, x in enumerate(self.expressions) if i not in trash]
        self.properties = [x for i, x in enumerate(self.properties) if i not in trash]
        self.index = None
        return deleted

    def data(self):
        return self.expressions, list(self.get_properties())

    def get_index(self):
        index = self.index
        if index is None:
            index = dict((v, i) for i, v in enumerate(self.expressions))
            self.index = index
        return index

    def get_properties(self):
        if self.properties:
            for p in self.properties:
                yield p
        else:
            for _ in range(len(self.expressions)):
                yield None

    def get_sorted(self):
        index = self.get_index()
        return lambda c: sorted(c, key=lambda v: index[v])

    def get_property(self, element, name):
        properties = self.properties
        if properties is None:
            return None
        index = self.get_index()
        i = index.get(element)
        if i is None:
            return None
        p = properties[i]
        if p is None:
            return None
        return p.get(name)


class _EdgeCollection(_Collection):
    def __init__(
        self, expressions, properties=None, index=None, n_directed=0, n_undirected=0
    ):
        super(_EdgeCollection, self).__init__(expressions, properties, index)
        self.counts = (n_directed, n_undirected)

    def is_mixed(self):
        n_directed, n_undirected = self.counts
        return n_directed > 0 and n_undirected > 0

    def clone(self):
        properties = self.properties
        n_directed, n_undirected = self.counts
        return _EdgeCollection(
            self.expressions[:],
            properties[:] if properties else None,
            None,  # index
            n_directed,
            n_undirected,
        )

    def extend(self, expressions, properties):
        added = super(_EdgeCollection, self).extend(expressions, properties)
        self.counts = _count_edges(self.counts, added, 1)
        return added

    def delete(self, expressions):
        deleted = super(_EdgeCollection, self).delete(expressions)
        self.counts = _count_edges(self.counts, deleted, -1)
        return deleted
