import numpy as np

################################################################################	
def A(x,lmbda,s,p):
# 
# u=(.5*p.p*(p.p+1))^(1/(p.p-1))*(sech(.5*(1-p.p)*x))^(2/(p.p-1));
#  
# out=[0 1 0; 0 0 1; -lambda 1-u^(p.p-1) 0];
#  
#  
	u=(.5*p['p']*(p['p']+1))**(1.0/(p['p']-1))*(np.cosh(.5*(1-p['p'])*x))**(-2.0/(p['p']-1))
	
	out=np.array([  [0, 1.0, 0], 
					[0, 0, 1.0], 	
					[-lmbda, 1-u**(p['p']-1), 0]
				 ])
	
	return out


	
	
################################################################################	
def Aadj(x,lmbda,s,p):
	'''
	Aadj(x,lambda,s,p)
	
	Returns the conjugate transpose of the matrix, s.A(x,lambda,s,p)
	'''
	out = -np.conj(A(x,lmbda,s,p).T)
	return out