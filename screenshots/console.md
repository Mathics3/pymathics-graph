The graphs in this directory are screenshots of [matplotlib-generated](https://matplotlib.org/) graphs from the session below.


```
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
In[5]:= CompleteKaryTree[6]
```

The files/graphs that have ``uncorrected`` the name use formatting that doesn't adjust node, font and edge sizes.
