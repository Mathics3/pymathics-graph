|Pypi Installs| |Latest Version| |Supported Python Versions|

`Mathics3 <https://mathics.org>`_ Graph Module using `NetworkX <https://networkx.org/>`_ and `Matplotlib <https://matplotlib.org>`_

Example Session
---------------

::

   $ mathicsscript
   Mathicscript: 5.0.0, Mathics 6.0.0
   on CPython 3.10.4 (main, Jun 29 2022, 12:14:53) [GCC 11.2.0]
   using SymPy 1.9, mpmath 1.2.1, numpy 1.21.5
   matplotlib 3.5.2,
   Asymptote version 2.81

   Copyright (C) 2011-2023 The Mathics3 Team.
   This program comes with ABSOLUTELY NO WARRANTY.
   This is free software, and you are welcome to redistribute it
   under certain conditions.
   See the documentation for the full license.

   Quit by pressing CONTROL-D

   In[1]:= LoadModule["pymathics.graph"]
   Out[1]= pymathics.graph
   In[2]:= BinomialTree[3]
   In[3]:= BinomialTree[6]
   In[4]:= CompleteKaryTree[3, VertexLabels->True]

Screenshots
-----------

|screenshot|

The above is the is the matplotlib graph for ``BinomialTree[3]`` in the session above.

See the `screenshot directory <https://github.com/Mathics3/pymathics-graph/tree/master/screenshots>`_ the other graphs.

Installation
-------------

From pip:

::

   $ pip install pymathics-graph

From git:

::

   $ make develop  # or make install

Note:
-----

Currently this works well in `mathicsscript` but not in the Django interface, although graphs are created in a temporary directory, e.g. ``/tmp/``.


.. |screenshot| image:: https://github.com/Mathics3/pymathics-graph/blob/master/screenshots/BinomialTree-3.png
.. |Latest Version| image:: https://badge.fury.io/py/pymathics-graph.svg
		 :target: https://badge.fury.io/py/pymathics-graph
.. |Pypi Installs| image:: https://pepy.tech/badge/pymathics-graph
.. |Supported Python Versions| image:: https://img.shields.io/pypi/pyversions/pymathics-graph.svg
.. |Packaging status| image:: https://repology.org/badge/vertical-allrepos/pymathics-graph.svg
			    :target: https://repology.org/project/pymathics-graph/versions
