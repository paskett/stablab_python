Copyright (c) 2018 Blake Barker, Jeffrey Humpherys, Joshua Lytle, Jalen Morgan,
                  Taylor Paskett

stablab is a Python package implementing the functionality of
STABLAB, a non-native MATLAB package. stablab provides numerical tools for
studying the stability of traveling waves. Specifically, unstable spectra of
traveling waves can be ruled out or located by numerically computing
the Evans function.

Several scripts are provided that compute the Evans function for several
example systems, including Burgers equation, the gKdV equation, and the
Boussinesq equation.

Basic dependencies include Python and the standard Python library,
numpy, scipy, and matplotlib.  Numpy and scipy are popular
open source software packages used in science, engineering, and mathematics.  
These libraries consist of a Python object-oriented interface to precompiled
C code, and support fast, vectorized arithmetic operations and a wide
assortment of functions.  Traveling wave profiles and Evans function output
are visualized using matplotlib.

This software was developed on a MacOS Sierra 10.12.6, using a Python
environment distributed by Anaconda (using Python 3.6).
