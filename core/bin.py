import numpy as np
from scipy import linalg
from pybvp6c.bvp6c import bvp6c, bvpinit, deval


def projection2(matrix,posneg,eps): 
	'''
	def projection2(matrix,posneg,eps):
		Algorithm
		return P,Q1
	Returns a projector P and an orthonormal spanning set Q1
	of the invariant subspace associated with the given matrix
	and the specified subspace.
	
	Input "matrix" is the matrix from which the eigenprojection comes,
	"posneg" is 1 or -1 depending on whether the unstable or stable space is
	sought. The input eps gives a bound on how small the eigenvalues sought
	can be, which is desirable when a zero mode should be avoided.'''
	
	T1,U1,sdim1 = linalg.schur(matrix,output='complex',sort=lambda x: posneg*x.real>eps)
	Q1 = U1[:,:sdim1]
	try:
		
		T2,U2,sdim2 = linalg.schur(-matrix,output='complex',sort=lambda x: posneg*x.real>-eps)
		Q2 = U2[:,:sdim2]
	except:
		print "Error"
		# print "matrix = "
		# print matrix
		# print matrix.shape
		# print matrix[0]
		# print type(matrix[0])
	
	R = np.concatenate((Q1, Q2), axis = 1)
	
	L = linalg.inv(R)
	P = np.zeros(matrix.shape)
	
	for i in range(sdim1): 
		P = P + np.outer(R[:,i],L[i,:] )
	return P,Q1


def projection1(matrix,posneg,eps):
	'''
	def projection1(matrix,posneg,eps):
		Algorithm
		return P,Q1
	Returns a projector P and an orthonormal spanning set Q1
	of the invariant subspace associated with the given matrix
	and the specified subspace.
	
	Input "matrix" is the matrix from which the eigenprojection comes,
	"posneg" is 1,-1, or 0 if the unstable, stable, or center space is
	sought respectively. The input eps gives a bound on how small the eigenvalues sought
	can be, which is desirable when a zero mode should be avoided.'''
	
	
	if posneg ==1:
		T1,U1,sdim1 = linalg.schur(matrix,output='complex',sort=lambda x: x.real>eps)
		Q1 = U1[:,:sdim1]
		
		T2,U2,sdim2 = linalg.schur(matrix,output='complex',sort=lambda x: x.real<=eps)
		Q2 = U2[:,:sdim2]
	elif posneg == -1: 
		T1,U1,sdim1 = linalg.schur(matrix,output='complex',sort=lambda x: x.real<-eps)
		Q1 = U1[:,:sdim1]
		
		T2,U2,sdim2 = linalg.schur(matrix,output='complex',sort=lambda x: x.real>=-eps)
		Q2 = U2[:,:sdim2]
	elif posneg == 0: 
		T1,U1,sdim1 = linalg.schur(matrix,output='complex',sort=lambda x: abs(x.real)<eps)
		Q1 = U1[:,:sdim1]
		
		T2,U2,sdim2 = linalg.schur(matrix,output='complex',sort=lambda x: abs(x.real)>=eps)
		Q2 = U2[:,:sdim2]
		
	R = np.concatenate((Q1, Q2), axis = 1); L = linalg.inv(R)
	P = np.zeros(matrix.shape)
	
	for i in range(sdim1): 
		P = P + np.outer(R[:,i],L[i,:] )
	
	return P,Q1


def soln(x,s): 
	"""
	Returns the solution of bvp problem where the domain was split in half
	
	Input "x" is the value where the solution is evaluated and "s" is a
	structure described in the STABLAB documenation
	
	s is a python dictionary that must contain the keywords
	side, I, L, R, solution, larray, rarray."""
	
	out = None
	if x < 0:
		try: 
			x = s['side']*s['I']/s['L']*x
			temp = deval(s['solution'],np.array([x]))[0]
			out = np.ones(len(s['larray']) )
			for j in range(0,len(s['larray']) ): 
				out[j] = temp[ s['larray'][j] ]
		except:
			raise TypeError, "Failure of soln"
	else:
		try: 
			x = s['side']*s['I']/s['R']*x
			temp = deval(s['solution'],np.array([x]))[0]
			out = np.ones(len(s['rarray']) )
			for j in range(0,len(s['rarray']) ): 
				out[j] = temp[ s['rarray'][j] ]
		except:
			raise TypeError, "Failure of soln"
	return out


