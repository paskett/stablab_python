import numpy as np

################################################################################
def A(x,lmbda,s,p):
	# Evans matrix for Burgers system in unintegrated coordinates.
	a=.5*(p['ul']-p['ur'])
	cc=.5*(p['ul']+p['ur']) # wave speed
	u=cc-a*np.tanh(a*x/2.0) # profile
	uder=(-a**2/2.0)*( (np.cosh(a*x/2.0))**(-1) )**2 # profile derivative
	out=np.array( \
	[[0,1],\
	[lmbda+uder,u-cc]]\
	)
	return out


################################################################################
def Aadj(x,lmbda,s,p):
	'''
	Aadj(x,lambda,s,p)

	Returns the conjugate transpose of the matrix, s.A(x,lambda,s,p)
	'''
	out = -np.conj(A(x,lmbda,s,p).T)
	return out




 