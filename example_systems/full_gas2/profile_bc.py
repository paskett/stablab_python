import numpy as np
import scipy.linalg as scipylin

def projection(matrix,posneg,eps):
    D,R = np.linalg.eig(matrix)
    L = np.linalg.inv(R)
    P = np.zeros(R.shape,dtype=np.complex)

    if posneg == 1:
        index = np.where(np.real(D) > eps)
    elif posneg == -1:
        index = np.where(np.real(D) < eps)
    elif posneg == 0:
        index = np.where(np.abs(np.real(D)) < eps)
    for j in index:
        P = P + np.dot(R[:,j],L[j,:])

    Q = np.concatenate([np.dot(P,R[:,j]) for j in index])
    #out = np.concatenate([P,Q],axis=1)
    return P

# Return a 1d array of conditions that should go to zero.
def profile_bc(fun, s, p):


    # Unpack the values at [[u0, uPrime0]] = f(0)
    (uVals,temp) = fun(0)
    (uValsR,temp) = fun(s['R'])
    u0 = uVals[0][0]
    actualRValues = uValsR[:,0]
    expectedRValues = s['UR']

    # Create the boundary conditions
    a = 0.5 * (s['UL'] + s['UR'])
    phaseCondition = u0 - a[0]

    #Get the projected boundary condition.
    #A_min = s['Flinear'](s['UL'],p)
    #s['LM'] = scipylin.orth(projection(A_min,-1,0)) # Removed .T inside orth
    A_plus = s['Flinear'](s['UR'],p)
    dotWith = scipylin.orth(projection(A_plus,1,0))[:,0] # Removed .T inside orth
    boundCondition = np.dot(dotWith, actualRValues - expectedRValues)

    #(What it is on the right - what it should be on the right) dot product with the boundCondition.

    #print(phaseCondition, boundCondition)
    out = [phaseCondition, boundCondition]
    return np.array(out)

