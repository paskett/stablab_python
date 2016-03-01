import numpy as np

################################################################################	
def A(x,lmbda,s,p):

	gamma = .5*np.sqrt(1-p['S']**2);
	u = 1.5*(1-p['S']**2)*(np.cosh(gamma*x))**(-2);
	ux = -2*gamma*u*np.tanh(gamma*x);
	uxx = 2*(gamma**2)*u*( 2-3*(np.cosh(gamma*x))**(-2) );
 
	a41 = -lmbda**2-2*uxx;
	a42 = 2*lmbda*p['S']-4*ux;
	a43 = (1-p['S']**2)-2*u;
 
	out = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [a41, a42, a43, 0]])

	return out
	
	
################################################################################	
def Aadj(x,lmbda,s,p):
	'''
	Aadj(x,lambda,s,p)
	
	Returns the conjugate transpose of the matrix, s.A(x,lambda,s,p)
	'''
	out = -np.conj(A(x,lmbda,s,p).T)
	return out