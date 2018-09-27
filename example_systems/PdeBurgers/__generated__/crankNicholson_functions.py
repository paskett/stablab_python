

# -*- coding: utf-8 -*-
"""
Contains the generated files with finite difference methods interlaced.  f is 
the output of the system of equations given some input.  createJacobian creates
the jacobian for use in Newton's method.

@author: pypde generated file
"""

import numpy as np


#Write the boundary functions, most basically defined as 0.
def lBoundFunction(U, n):
    return [U[n+1,0]]
def rBoundFunction(U, n):
    return [U[n+1,-1]-1.7]
def lBoundDerivative():
    return [0]
def rBoundDerivative():
    return [0]

#loop through equations, creating a vector for all j's
def f(matrices, parameters, K, H, n):
    U = matrices[0]
    outVector = []
    
    #Add values for equation 1
    outVector.append(0)
    for j in range(1,len(matrices[0][0])-1):
        outVector.append(U[n+1,j]*((U[n+1,j+1] - U[n+1,j-1])/(4*H) + (U[n,j+1] - U[n,j-1])/(4*H)) + (U[n+1,j] - U[n,j])/K - 0.5*(U[n+1,j+1] + U[n+1,j-1] - 2*U[n+1,j])/H**2 - 0.5*(U[n,j+1] + U[n,j-1] - 2*U[n,j])/H**2)
    outVector.append(0)
    return outVector

    
def createJacobian(matrices, time, K, H, parameters):
    #Get the boundary values
    U = matrices[0]
    
    #Create the empty matrix.
    jacobianDimensions = len(matrices)*(len(matrices[0][0]))
    jacobian = np.zeros((jacobianDimensions, jacobianDimensions))
    quadrantDimensions = len(matrices[0][0])
    n = time -1
            
    #Loop through quadrant (1,1).
    jacobian[quadrantDimensions*0,quadrantDimensions*0] = 1
    jacobian[quadrantDimensions*(1+0)-1,quadrantDimensions*(1+0)-1] = 1
    for row in range(quadrantDimensions-2):
        for col in range(quadrantDimensions-2):
            row += 1
            col += 1
            j = col
            if (row - col == 1): jacobian[row+quadrantDimensions*0,col+quadrantDimensions*0] = -U[n+1,j]/(4*H) - 0.5/H**2
            if (row == col): jacobian[row+quadrantDimensions*0,col+quadrantDimensions*0] = 1/K + (U[n+1,j+1] - U[n+1,j-1])/(4*H) + (U[n,j+1] - U[n,j-1])/(4*H) + 1.0/H**2
            if (row - col == -1): jacobian[row+quadrantDimensions*0,col+quadrantDimensions*0] = U[n+1,j]/(4*H) - 0.5/H**2
    return(jacobian)
    