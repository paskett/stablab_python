from __future__ import division
import numpy as np
from core.bin import soln

################################################################################	
def A(x,lmbda,s,p):
# 	if np.abs(x) > s['R']:
# 		print ; print "Alert. Max x value exceeded."; print ; 
	temp = soln(x,s)
	# print "temp = ", temp
	v = np.zeros(4)
	v[0]=temp[0]
	v[1]=temp[1]
	v[3]=temp[2]
	p['speed'] = temp[3]
	v[2]= (1.0/p['speed'])*(p['speed']-p['beta']*p['speed']*temp[0]-p['beta']*v[1]-p['tau']*temp[2])

# 	if (v[0]>=1e-4): 
# 		e = np.exp(-1.0/v[0]) 
# 	else: e=0
	
	if (v[0]!=0.0): 
		e = np.exp(-1.0/v[0]) 
	else: e=0

	vu2 = (1.0*v[2])/(v[0]**2)
	
	a21 = (lmbda - vu2*e) 
	a23 = -e
	a41 = (p['beta']*vu2*e/p['tau'])

	a43 = (lmbda + p['beta']*e)/(1.0*p['tau'])
	a44 = (-p['speed']/p['tau'])
	
	out = np.array([[0.0,        1.0,        0.0,        0.0],
		[a21,      -p['speed'],       a23,      0.0],
		[0.0,        0.0,        0.0,        1.0],
		[a41,      0.0,        a43,      a44]],dtype=complex)

	return out

	
################################################################################	
def Aadj(x,lmbda,s,p):
	'''
	Aadj(x,lambda,s,p)
	
	Returns the conjugate transpose of the matrix, s.A(x,lambda,s,p)
	'''
	out = -np.conj(A(x,lmbda,s,p).T)
	return out