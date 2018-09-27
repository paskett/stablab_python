"""
Example of 1-dimensional stationary burgers equation

Created on Friday, June 30, 2017
@authors:  Blake Barker, Jalen Morgan, Taylor Paskett

see README.txt for help on using PyPDE.

************************************************************
*********************************************************"""
from stablab import finite_difference as fd
import numpy as np

def lBoundFunction(UIn, n):
    U = UIn[0]
    return [U[n+1,0]-2]

def rBoundFunction(U, n):
    return [U[0][n+1,-1]]

def lBoundDerivative(U, n):
    UValue = [1]
    return UValue

def rBoundDerivative(U, n):
    UValue = [1]
    return UValue

if __name__ == "__main__":
    #burgerFunctions = fd.generateFiniteDifference("(U)_t + (U**2/2)_x + 0 - U_xx","U")
    #burgerFunctions = fd.generateFiniteDifference(['(U)_t + (U**2/2)_x + 0 - (U_x_x)'],['U'],)
    burgerFunctions = fd.generateFiniteDifferenceConservation("U","U**2/2",0,"1.0","U")
    #testFunctions = fd.generateFiniteDifferenceConservation(["U","V"],["U**2/2","V**2/2"],[0,0],["1.0","1.0"],["U","V"])
    xPoints = np.linspace(-4,8,40)
    tPoints = np.linspace(0,8,40)
    t0 = fd.getInitialCondition(xPoints, lambda x: 1-np.tanh(x)-1/(1+x**2))

    #Create the lBound and rBound functions
    lBound = (lBoundFunction, lBoundDerivative)
    rBound = (rBoundFunction, rBoundDerivative)

    #Get the solution matrix, passing all the parameters, and graph it.
    (U,) = fd.evolve(xPoints, tPoints, lBound, rBound, t0, burgerFunctions)
    fd.graph("U", U)
