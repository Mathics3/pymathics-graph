9.0.0
-----

Aug 29, 2025

Add support for Python 3.13. Drop Support for Python 3.8 and Python 3.9.

Track API changes in Mathics3 Kernel.

8.0.1
-----

Feb 8, 2025

Documentation has been revised to make better use of TeX math mode.
Adjust and correct packaging (thanks to Atri Bhattacharya for testing and packaging in OpenSUSE) .

8.0.0
-----

Jan 26, 2025

This release tracks the API changes in the Mathics Kernel.

* Added builtin function `GraphQ[]`. (Combinatorica uses this)
* Redo `DirectedEdge` and `UndirectedEdge` as operators

7.0.0
-----

Aug 10, 2025


* Revise for 7.0.0 Mathics3 API; we need to explicilty load builtins
* Newer matplotlib requires a plot close.
* Networkx 3.3 supported
* sort connected components in output


6.0.0
-----

Revise for 6.0.0 Mathics3 API and current Mathics3 builtin standards
described in [Guidelines for Writing
Documentation](https://mathics-development-guide.readthedocs.io/en/latest/extending/developing-code/extending/documentation-markup.html#guidelines-for-writing-documentation).

This package has undergone a major overhaul. Modules have been split out along into logical groups following the documentation structure.

We have gradually been rolling in more Python type annotations and
have been using current Python practices. Tools such as using
``isort``, ``black`` and ``flake8`` are used as well.

Evaluation methods of built-in functions start ``eval_`` not
``apply_``.

There is more refactoring more to do here, Upgrade to NetworkX is also
desirable.

5.0.0.alpha0
------------

Track API changes in Mathics 5.0.0.

Changed to use networkx 2.8 or greater.

Some functionality has been removed for now, because networkx 2.8's API is a bit different with its new NodeView and EdgeView API.

2.3.0
-----

Small API changes to track Mathics 3.0.0.

Blacken everything



2.2.0
-----

Small changes to track Mathics 2.2.0

1.0.0
-----

First public release.

The names below follow the WL names. Look up the corresponding documentation for information about these.

Functions provided
------------------

- ``AcyclicGraphQ``
- ``AdjacencyList``
- ``BalancedTree``
- ``BarbellGraph``
- ``BetweennessCentrality``
- ``BinomialTree``
- ``ClosenessCentrality``
- ``CompleteGraph``
- ``CompleteKaryTree``
- ``ConnectedComponents``
- ``ConnectedGraphQ``
- ``CycleGraph``
- ``DegreeCentrality``
- ``DirectedEdge``
- ``DirectedGraphQ``
- ``EdgeAdd``
- ``EdgeConnectivity``
- ``EdgeCount``
- ``EdgeDelete``
- ``EdgeIndex``
- ``EdgeList``
- ``EdgeRules``
- ``EigenvectorCentrality``
- ``FindShortestPath``
- ``FindSpanningTree``
- ``FindVertexCut``
- ``FullRAryTree``
- ``Graph``
- ``GraphAtlas``
- ``GraphBox``
- ``GraphData``
- ``GraphDistance``
- ``HITSCentrality``
- ``HighlightGraph``
- ``HknHararyGraph``
- ``HmnHararyGraph``
- ``KaryTree``
- ``KatzCentrality``
- ``LadderGraph``
- ``LoopFreeGraphQ``
- ``MixedGraphQ``
- ``MultigraphQ``
- ``PageRankCentrality``
- ``PathGraph``
- ``PathGraphQ``
- ``PlanarGraphQ``
- ``Property``
- ``PropertyValue``
- ``RandomGraph``
- ``RandomTree``
- ``SimpleGraphQ``
- ``StarGraph``
- ``TreeGraph``
- ``TreeGraphQ``
- ``UndirectedEdge``
- ``VertexAdd``
- ``VertexConnectivity``
- ``VertexCount``
- ``VertexDegree``
- ``VertexDelete``
- ``VertexIndex``
- ``VertexList``
- ``WeaklyConnectedComponents``


GraphData names
----------------

- ``DodecahedralGraph``
- ``DiamondGraph``
- ``PappusGraph``
- ``IsohedralGraph``
- ``PetersenGraph``

The names below follow the WL names. Look up the corresponding documentation for information about these.
However you can also use the corresponding networkx name, e.g. "c" for "Circle", "D" for "Diamond", etc.

Node Marker Names
----------------

- ``Circle``
- ``Diamond``
- ``Square``
- ``Star``
- ``Pentagon``
- ``Octagon``
- ``Hexagon``
- ``Triangle``
