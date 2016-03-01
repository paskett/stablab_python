Copyright (C) 2015 Blake Barker, Jeffrey Humpherys, Joshua Lytle, Kevin Zumbrun


pystablab is a Python package implementing the functionality of 
STABLAB, a non-native MATLAB package. pystablab provides numerical tools for 
studying the stability of traveling waves. Specifically, unstable spectra of 
traveling waves can be ruled out or located by numerically computing 
the Evans function.

Several scripts are provided that compute the Evans function for several 
example systems, including Burgers equation, the gKdV equation, and the 
Boussinesq equation. 

Basic dependencies include Python and the standard Python library, 
numpy, scipy, matplotlib, and mpi4py.  Numpy and scipy are popular 
open source software packages used in science, engineering, and mathematics.  
These libraries consist of a Python object-oriented interface to precompiled 
C code, and support fast, vectorized arithmetic operations and a wide 
assortment of functions.  Traveling wave profiles and Evans function output
are visualized using matplotlib.  Finally, mpi4py allows the Evans function 
to be evaluated in parallel. 

This software was developed on a Mac OS X 10.10.5, using a Python 
environment distributed by Anaconda (using Python 2.7).

sys.path in Python  can be updated either by the relevant commands in a 
Python script, or by modifying the $PYTHONPATH variable in .bash_profile. 
In this last case .bash_profile must then be saved and the user must log out 
and then log in again. 

The testing suite may be run by cd-ing to the pystablab directory and 
typing the following command in the terminal: 
python -m unittest discover -p '*_test.py'

Run tests at the command line with 
python -m unittest discover -v
ipython nbconvert --to=html psuedospectral_example.ipynb
ipython nbconvert --to=html 