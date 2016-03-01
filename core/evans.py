import numpy as np
from scipy import linalg
from scipy.integrate import complex_ode

from contour import analytic_basis
from bin import projection2


def evans(yl,yr,lmbda,s,p,m,e): 
	muL = np.trace( np.dot( np.dot(np.conj( (linalg.orth(yl) ).T),e['LA'](e['Li'][0],lmbda,s,p) ),linalg.orth(yl) ) );
	muR = np.trace( np.dot( np.dot(np.conj( (linalg.orth(yr) ).T),e['RA'](e['Ri'][0],lmbda,s,p) ),linalg.orth(yr) ) );
	[omegal,gammal]=manifold_polar(e['Li'],linalg.orth(yl),lmbda,e['LA'],s,p,m,e['kl'],muL)
	[omegar,gammar]=manifold_polar(e['Ri'],linalg.orth(yr),lmbda,e['RA'],s,p,m,e['kr'],muR)
	
	if e['evans'] =='reg_reg_polar':
		out = ( linalg.det( np.dot(np.conj( linalg.orth(yl).T ),yl) )*
				linalg.det( np.dot(np.conj( linalg.orth(yr).T ),yr) )*
		gammal*gammar*linalg.det( np.concatenate( (omegal, omegar),axis=1)	   )  )
	
	if e['evans'] =='adj_reg_polar':
		out = ( np.conj( linalg.det( np.dot(np.conj( linalg.orth(yl).T ),yl) ) )*
						 linalg.det( np.dot(np.conj( linalg.orth(yr).T ),yr) )*
		np.conj(gammal)*gammar*linalg.det(	(np.conj(omegal.T).dot( omegar) )	 )	)
	
	return out


def manifold_polar(x,y,lmbda,A,s,p,m,k,mu): 
	
	def ode_f(x,y): 
		return m['method'](x,y,lmbda,A,s,p,m['n'],k,mu,m['damping'])
	
	t0=x[0]; y0 = y.reshape(m['n']*k,1,order = 'F')
	y0 = np.concatenate( ( y0, np.array([[1.0]]) ),axis=0); y0 = y0.T[0]
	
	#initiate integrator object
	test = complex_ode(ode_f).set_integrator('dopri5',atol=1e-5) 
	
	test.set_initial_value(y0,t0) # set initial time and initial value
	test.integrate(0)
	Y = test.y
	Y = np.array([Y.T]).T
	
	omega = Y[0:k*m['n'],0].reshape(m['n'],k,order = 'F')
	gamma = Y[-1,0]; gamma=np.exp(gamma)
	
	return omega, gamma


def drury(t,y,lambda0,A,s,p,n,k,mu,damping):
	y = np.array([y.T]).T; W = y[0:k*n,0].reshape(n,k,order = 'F')
	A_temp = A(t,lambda0,s,p)
	
	y_temp = (np.eye(n)-W.dot(np.conj(W.T) ) ).dot( A_temp.dot(W) )+damping*W.dot( (np.eye(k)-np.conj(W.T).dot(W)) )
	ydot = np.concatenate( ( y_temp.reshape(n*k,1,order = 'F'), np.array([[0]]) ),axis=0)
	ydot[-1] = (np.trace(np.conj(W.T).dot( A_temp.dot(W) ) )-mu)
	
	return ydot.T[0]


def Evans_compute(preimage,c,s,p,m,e):
	lbasis,lproj = analytic_basis(projection2,c['L'],preimage,s,p,c['LA'],1,c['epsl'])
	# raise SystemError
	rbasis, rproj = analytic_basis(projection2,c['R'],preimage,s,p,c['RA'],-1,c['epsr'])
	
	
	''' Finds the subset on which Evans function is initially evaluated'''
	index =	 np.arange(1,len(preimage),c['ksteps'])
	preimage2 = preimage[index]
	lbasis2 = lbasis[:,:,index]
	rbasis2 = rbasis[:,:,index]
	
	out =np.zeros(len(preimage2),dtype ='complex')
	''' Computes the Evans function'''
	for j in np.arange(0,len(preimage2)):
		out[j] = c['evans'](lbasis2[:,:,j],rbasis2[:,:,j],preimage2[j],s,p,m,e)
		# print "lambda = ", preimage2[j]
		# print "E(lambda) = ", out[j]
	return out


def emcset(s,shock_type,eL,eR,Evan_type):
	'''
	def emcset(s,shock_type,eL,eR,Evan_type):
		
	Sets the values of the STABLAB structures e, m, and c to
	default values. Takes as input a string, shock_type, which is either
	"front" or "periodic". The input eL and eR are respectively the
	dimension of the left and right eigenspaces of the Evans matrix.
	The input Evan_type is an optional string. If not specified, Evan_type
	will be assigned the most advantageous polar coordinate method.
	Evan_type has the following options when shock_type = 'front':
	
	reg_reg_polar
	reg_adj_polar
	adj_reg_polar
	reg_adj_compound
	adj_reg_compound
	
	when shock_type = 'periodic', the choices are:
	
	regular_periodic
	balanced_periodic
	balanced_polar_scaled_periodic
	balanced_polar_periodic
	balanced_scaled_periodic'''
	
	e,m,c = initialize_front(s,eL,eR,Evan_type)
	return s,e,m,c


def initialize_front(s,kL,kR,Evan_type):
	e = {}
	c = {}
	m = {'n':kL+kR}
		
	# PFunctions = importlib.import_module(s['PFunctions'])
	EvansSystems = s['EvansSystems']
	# EvansSystems = s.EvansSystems
	
	if Evan_type=='default':
		if kL > m['n']/2:
			e['evans'] = 'adj_reg_polar'
		elif kL < m['n']/2:
			e['evans']='reg_adj_polar'
		else:
			e['evans'] ='reg_reg_polar'
	else:
		e['evans'] = Evan_type
	
	if e['evans'] =='reg_reg_polar':
		c.update({'LA':EvansSystems.A,'RA':EvansSystems.A})
		e.update({'LA':EvansSystems.A,'RA':EvansSystems.A,
					'kl':kL,'kr':kR})

	if e['evans'] =='adj_reg_polar':
		c.update({'LA':EvansSystems.Aadj,'RA':EvansSystems.A})
		e.update({'LA':EvansSystems.Aadj,'RA':EvansSystems.A,
					'kl':m['n']-kL,'kr':kR})
	# if e['evans'] =='reg_reg_polar':
	# 	c.update({'LA':EvansSystems['A'],'RA':EvansSystems['A']})
	# 	e.update({'LA':EvansSystems['A'],'RA':EvansSystems['A'],
	# 				'kl':kL,'kr':kR})
	#
	# if e['evans'] =='adj_reg_polar':
	# 	c.update({'LA':EvansSystems['Aadj'],'RA':EvansSystems['A']})
	# 	e.update({'LA':EvansSystems['Aadj'],'RA':EvansSystems['A'],
	# 				'kl':m['n']-kL,'kr':kR})
	
	c.update({'stats':'off',
			  'refine':'off',
		  	  'tol':0.2,
	  		  'ksteps':2**5,
  			  'lambda_steps':0,
			  'basisL':analytic_basis,
			  'basisR':analytic_basis,
		  	  'evans':evans})
	
	c.update({'epsl':0,
			  'epsr':0,
			  'Lproj':projection2,
			  'Rproj':projection2 })
	
	m.update({'damping':0,
			  'method':drury })
	# # m['options'] = odeset('RelTol',1e-6,'AbsTol',1e-8,'Refine',1,'Stats','off')
	# #dependent structure variables
	e.update({'Li':[s['L'], 0],
			  'Ri':[s['R'], 0]  })
	c.update({'L':s['L'],
			  'R':s['R']  })
	# e.update({'Li':[s.L, 0],
	# 		  'Ri':[s.R, 0]  })
	# c.update({'L':s.L,
	# 		  'R':s.R  })
	
	return e,m,c





