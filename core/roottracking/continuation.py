import numpy as np
from scipy import linalg
from scipy.integrate import complex_ode


def eigf_initshooting(s,e,m,c,p,x1,x2):
	# This calculates an initial guess for the eigenfunction W associated with the 
	# eigenvalue lmbda = p['rooot'], using a modification of the polar coordinate method
	# Specifically, alpha is explicitly computed, rather than using the simplifying radial
	# equation
	
	pl, yl = projection2(e['LA'](s['L'],p['rooot'],s,p),1,1e-8)
	pr, yr = projection2(e['RA'](s['R'],p['rooot'],s,p),-1,1e-8)
	lmbda = p['rooot'] 
	
	# Input "yl" and "yr" are respectively the initializing values on the left
	# and right for the desired manifolds, "lmbda" is the value in the complex
	# plane where the Evans function is evaluated, and s,p,m,e are structures
	# explained in the STABLAB documentation.
	
	omegal, alphal = mod_manifold_polar(x1,linalg.orth(yl),lmbda,e['LA'],s,p,m,e['kl'])
	omegar, alphar = mod_manifold_polar(x2,linalg.orth(yr),lmbda,e['RA'],s,p,m,e['kr'])
	
	return omegal, alphal, omegar, alphar


def mod_manifold_polar(x,y,lmbda,A,s,p,m,k):
	'''
	def manifold_polar(x,y,lambda,A,s,p,m,k,mu):
	return omega, alpha
	Returns "omega", the orthogonal basis for the manifold evaluated at x(2)
	and "gamma" the radial equation evaluated at x(2).

	Input "x" is the interval on which the manifold is solved, "y" is the
	initializing vector, "lambda" is the point in the complex plane where the
	Evans function is evaluated, "A" is a function handle to the Evans
	matrix, s,p,m are structures explained in the STABLAB documentation, and k
	is the dimension of the manifold sought.'''
	
	def ode_f(x,y): 
		return m['method'](x,y,lmbda,A,s,p,m['n'],k)
	
	omega = np.zeros( (m['n'],k,len(x)), dtype = complex)
	alpha = np.zeros( (k,k,len(x)), dtype = complex)
	omega[:,:,0], alpha[:,:,0] = y, np.eye(k)
	
	t0, y0 = x[0], y.reshape(m['n']*k,order = 'F')
	y0 = np.concatenate( ( y0, (np.eye(k)).reshape(k*k,order = 'F') ) )
	
	test = complex_ode(ode_f).set_integrator( 'dopri5', 
											  atol=m['options']['AbsTol'], 
											  rtol=m['options']['RelTol'], 
											  nsteps=5000  ) 
	test.set_initial_value(y0,t0) 
	
	for j in range(1,len(x)): 
		test.integrate(x[j])
		omega[:,:,j] = test.y[0:k*m['n']].reshape(m['n'],k,order = 'F')
		alpha[:,:,j] = test.y[k*m['n']:].reshape(k,k,order = 'F')
	return omega, alpha


def mod_drury(t,y,lambda0,A,s,p,n,k):
	'''
	drury(t,y,lmbda,A,s,p,n,k,mu,damping):
	return ydot
	Returns the ODE output for the polar method using the method of Drury

	Input "t" and "y" are provided by dopri5, "A" is a function handle to the
	desired Evans matrix, s,p are structures explained in the STABLAB
	documentation, "n" is the dimension of the system and "k" is the
	dimension of the manifold, "mu" is the rescaling value for increased stability,
	and "damping" is the damping coefficeint.'''
	
	Omega = y[0:k*n].reshape(n,k,order='F')
	alpha = y[k*n:].reshape(k,k,order='F')
	
	ydot, A_temp = np.empty((n+k)*k), A(t,lambda0,s,p)
	ydot[0:k*n] =   ( (np.eye(n)-Omega.dot( np.conj(Omega.T) ) 
				 	).dot(A_temp.dot(Omega) )).real.reshape(n*k,order = 'F')
	ydot[k*n:] = ( (np.conj(Omega.T)).dot(A_temp.dot( Omega.dot(alpha) ) ) 
	           	 ).real.reshape(k*k,order = 'F')
	return ydot


def eigf_init_guess(s,e,m,c,p):
	"""Creates an initial guess for the eigenfunction and eigenvalue pair (W,lmbda)
	"""	
	x1 = np.linspace(s['L'],0,p['temp_n']); x2 = np.linspace(s['R'],0,p['temp_n'])
	[omegal, alphal, omegar, alphar] = EvansBin.eigf_initshooting(s,e,m,c,p,x1,x2)
	
	X = np.concatenate( (omegal[:,:,-1].dot(alphal[:,:,-1]), omegar[:,:,-1].dot(alphar[:,:,-1])), axis = 1)
	w,v = linalg.eig(X)
	Y = np.min(np.abs(w)); I = np.argmin(np.abs(w))
	C = v[:,I]
	
	s['guess'] = np.zeros( (2*m['n']+1,p['temp_n']),dtype = complex)
	
	for j in range(0,p['temp_n']):
		Wr = omegar[:,:,-1-j].dot( alphar[:,:,-1-j] )
		Wl = omegal[:,:,-1-j].dot( alphal[:,:,-1-j] )
		
		s['guess'][:m['n'],j] =	 C[2] *Wr[:,0]
		s['guess'][m['n']:2*m['n'],j] =	 -C[0] *Wl[:,0] - C[1]*Wl[:,1]
		
	s['guess'][:m['n'],:] = -s['guess'][:m['n'],:]/( linalg.norm(s['guess'][:m['n'],0]) )
	s['guess'][m['n']:2*m['n'],:] = s['guess'][m['n']:2*m['n'],:]/( linalg.norm(s['guess'][m['n']:2*m['n'],0]) )
	s['guess'][2*m['n'],:] = p['rooot']*np.ones((s['guess'][2*m['n'],:]).shape,dtype = complex)
	return s,x1,x2


# def the_omegas(yl,yr,lmbda,s,p,m,e):
# 	muL = np.trace( np.dot( np.dot(np.conj( (linalg.orth(yl) ).T),e['LA'](e['Li'][0],lmbda,s,p) ),linalg.orth(yl) ) );
# 	muR = np.trace( np.dot( np.dot(np.conj( (linalg.orth(yr) ).T),e['RA'](e['Ri'][0],lmbda,s,p) ),linalg.orth(yr) ) );
# 	[omegal,gammal]=manifold_polar(e['Li'],linalg.orth(yl),lmbda,e['LA'],s,p,m,e['kl'],muL)
# 	[omegar,gammar]=manifold_polar(e['Ri'],linalg.orth(yr),lmbda,e['RA'],s,p,m,e['kr'],muR)
#
# 	if e['evans'] =='reg_reg_polar':
# 		out = ( linalg.det( np.dot(np.conj( linalg.orth(yl).T ),yl) )*
# 		linalg.det( np.dot(np.conj( linalg.orth(yr).T ),yr) )*
# 		gammal*gammar*linalg.det( np.concatenate( (omegal, omegar),axis=1)	   )  )
#
# 	if e['evans'] =='adj_reg_polar':
# 		out = ( np.conj( linalg.det( np.dot(np.conj( linalg.orth(yl).T ),yl) ) )*
# 		linalg.det( np.dot(np.conj( linalg.orth(yr).T ),yr) )*
# 		np.conj(gammal)*gammar*linalg.det(	(np.conj(omegal.T).dot( omegar) )	 )	)
#
# 	return omegal, omegar
# 
# def basis_compute_at0(preimage,c,s,p,m,e):
# 	lbasis,lproj = EvansBin.analytic_basis(EvansBin.projection2,c['L'],preimage,s,p,c['LA'],1,c['epsl'])
# 	
# 	rbasis, rproj = EvansBin.analytic_basis(EvansBin.projection2,c['R'],preimage,s,p,c['RA'],-1,c['epsr'])
# 	
# 	''' Finds the subset on which Evans function is initially evaluated'''
# 	index =	 np.arange(1,len(preimage),c['ksteps'])
# 	preimage2 = preimage[index]; lbasis2 = lbasis[:,:,index]; rbasis2 = rbasis[:,:,index]
# 	
# 	# out =np.zeros(len(preimage2),dtype ='complex')
# 	# ''' Computes the Evans function'''
# 	# for j in np.arange(0,len(preimage2)):
# 	# 	out[j] = EvansBin.the_omegas(lbasis2[:,:,j],rbasis2[:,:,j],preimage2[j],s,p,m,e)
# 	out = np.ones((len(preimage2),4,2),dtype ='complex')
# 	# out = [0]*len(preimage2)
# 	for j in np.arange(0,len(preimage2)):
# 		out[j,:,:] = EvansBin.the_omegas(lbasis2[:,:,j],rbasis2[:,:,j],preimage2[j],s,p,m,e)[0]
# 		# out[j] = EvansBin.the_omegas(lbasis2[:,:,j],rbasis2[:,:,j],preimage2[j],s,p,m,e)
# 	return out, preimage2
# 
# 