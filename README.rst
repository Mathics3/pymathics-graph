`Mathics <https://mathics.org>`_ Graph Module using `NetworkX <https://networkx.org/>`_ and `Matplotlib <https://matplotlib.org>`_

Example Session
---------------

::

   $ mathicsscript
   Mathics 1.1.1
   on CPython 3.6.12 (default, Oct 24 2020, 10:34:18)
   using SymPy 1.8.dev, mpmath 1.1.0, cython 0.29.21

   Copyright (C) 2011-2020 The Mathics Team.
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
