import numpy as np
from scipy import linalg
import scikits.bvp_solver

from combustionEvans import A
import bin.EvansBin as EvansBin
from bin.EvansBin import projection1, projection2
# Contains three functions: init_prof_guess, prof_solver, and 
# eigf_solver

def init_prof_guess(x,s,p):
	alpha = .45
	speed = np.sqrt(2.0/(p['beta']*p['tau']))*np.exp(-p['beta']/2.0)
	
	UL, UR = [1.0/p['beta'], 0, 0],  [0, 0, 0]
	# NInfty=s['L']; PInfty=s['R']; Infty=s['I']
	a, c = (UL[0]+UR[0])/2.0, (UL[0]-UR[0])/2.0
	
	out = np.zeros(8)
	out[0] = a-c*np.tanh(alpha*(s['R']/s['I'])*x)
	out[1:3] = -c*alpha*np.cosh(alpha*(s['R']/s['I'])*x)**(-2.)
	out[3] = speed
	out[4] = a-c*np.tanh(alpha*(s['L']/s['I'])*x)
	out[5:6] = -c*alpha*np.cosh(alpha*(s['L']/s['I'])*x)**(-2.)	
	out[7] = speed
	return out



def prof_solver(prof_guess,s,p):
	
	def profile_F(x,y,p): 
		if y[0]>=0.0001: 
			e = np.exp(-1.0/y[0])
		else: 
			e = 0.
		z = y[3]
		ve = (z - p['beta']*z*y[0]- p['beta']*y[1] - p['tau']*y[2] )*e
		out = np.array([ 
						y[1], 
				  		-z*y[1] - ve/z, 
						-z*y[2]/p['tau'] + p['beta']*ve/(p['tau']*z), 
					 	0.0])
		
		return out
	
	
	def double_profile_F(x,y): # ,s,p):
		n=s['n']
		yr, yl = list(y)[0:n], list(y)[n:2*n]
		
		out = np.concatenate(( 
								(s['R']/s['I'])*profile_F( x,np.array(yr),p ),
								(s['L']/s['I'])*profile_F( x,np.array(yl),p ) ))
		
		return out
	
	
	def Flinear(y,speed,p):
		beta, tau = p['beta'], p['tau']
		if y[0] == 0: 
			e = 0.
		elif y[0] > 0: 
			e = np.exp(-beta)
	#	if y[0]>=0.0001: e = np.exp(-1/y[0])
	#	else: e = 0
	#	print 'e = ', e
		out = np.array([
		  [ 0,				1,					  0 ],
		  [beta*e,			  -speed+beta*e/speed,		  tau*e/speed], 
		  [-beta**2*e/tau,	 -beta**2*e/(tau*speed),  -speed/tau - beta*e/speed]])
		
		return out
	
	
	def profile_bcs(Ya,Yb): # ,s,p):
		n, speed = s['n'], Ya[3]
		UL, UR = [1.0/p['beta'], 0, 0], [0, 0, 0]
		#
		AM, AP = Flinear(np.array(UL), speed, p), Flinear(np.array(UR), speed, p)
		Pm, _ = projection1(AM,-1,0.0) # Decay manifold at - infty
		Pp, _ = projection1(AP,0,1e-8) # Growth manifold at + infty
		LM, LP = linalg.orth(Pm.T), linalg.orth(Pp.T)
		# Conditions at left endpoint
		BCa = list(Ya[0:n]-Ya[n:2*n]) 				#matching conditions = 4
		BCa.append( Ya[0] - 0.5/p['beta'])			# phase condition = 1 
		# Conditions at right endpoint
		BCb = list(LM.T.dot(Yb[n:2*n-1] - UL)) 		# at -infty; = 2
		BCb.append( (LP.T.dot(Yb[0:n-1]-UR))[0]) 	# at +infty; = 1
		return np.array(BCa,dtype = float), np.array(BCb,dtype = float)
	
	
	
	# print "Setting up problem"
	pro_problem = scikits.bvp_solver.ProblemDefinition(
							num_ODE = 8,
							num_parameters = 0,
							num_left_boundary_conditions = 5,
							boundary_points = (0, s['I']),
							function = double_profile_F,
							boundary_conditions = profile_bcs)
	
	
	# print "Solving the problem"
	pro_solution = scikits.bvp_solver.solve(
						pro_problem,
						solution_guess = prof_guess,
						trace = 0,
						max_subintervals=1000,
						# parameter_guess = np.array([15.0]),
						tolerance=1e-9)
	
	
	# print "Returning from profile world"
	return pro_problem, pro_solution



def eigf_solver(s,p):
	
	def eigf_F(x,y): # ,s,p):
		n=s['n'] # = 4 = Dimension of first order profile equation
		eig_n = 2*n # = 8 = Dimension of first order eigenfunction equation
		out = np.concatenate( 
			(np.real( (s['R']/s['I'])*A((s['R']/s['I'])*x,(y[-2] + 1.0j*y[-1]),s,p).dot( np.array([y[0:n] + 1.0j*y[n:eig_n] ]).T ) ),
			np.imag( (s['R']/s['I'])*A((s['R']/s['I'])*x,(y[-2] + 1.0j*y[-1]),s,p).dot( np.array([y[0:n] + 1.0j*y[n:eig_n] ]).T) ),
			np.real( (s['L']/s['I'])*A((s['L']/s['I'])*x,(y[-2] + 1.0j*y[-1]),s,p).dot( np.array([y[eig_n:eig_n+n] + 1.0j*y[eig_n+n:2*eig_n] ]).T) ),
			np.imag( (s['L']/s['I'])*A((s['L']/s['I'])*x,(y[-2] + 1.0j*y[-1]),s,p).dot( np.array([y[eig_n:eig_n+n] + 1.0j*y[eig_n+n:2*eig_n] ]).T) ) 
			),
			axis=0)
		
		#
		out = list(out[:,0]); out.append(0.0); out.append(0.0)
		return np.array(out)
	
	
	def eigf_boundary_conditions(Ya,Yb): # ,s,p):
		n=s['n'] # = 4 = Dimension of first order profile equation
		eig_n = 2*n # = 8 = Dimension of first order eigenfunction equation
		ph = s['ph']
		AM = A(s['L'], ((Ya[-2] + 1.0j*Ya[-1]) + (Yb[-2] + 1.0j*Yb[-1]))/2.0,s,p)
		AP = A(s['R'], ((Ya[-2] + 1.0j*Ya[-1]) + (Yb[-2] + 1.0j*Yb[-1]))/2.0,s,p)
		#
		P1,Q1 = projection2(AM,-1,1e-6); LM = linalg.orth(P1.T)
		P2,Q2 = projection2(AP,+1,1e-6); LP = linalg.orth(P2.T)
		#
		BCa = list(Ya[0:eig_n]-Ya[eig_n:2*eig_n]) # 8 matching conditions
		BCa.append( Ya[ph[0,0]]- ph[0,1] )		# 2 phase conditions
		BCa.append( Ya[ph[1,0]]- ph[1,1] )
		#
		# 8 projective conditions
		BCb1 = np.real(LP.T.dot( Yb[0:n] + 1.0j*Yb[n:2*n] ))
		BCb2 = np.imag(LP.T.dot( Yb[0:n] + 1.0j*Yb[n:2*n] ))
		BCb3 = np.real(LM.T.dot( Yb[eig_n:eig_n+n] + 1.0j*Yb[eig_n+n:eig_n+2*n]	))
		BCb4 = np.imag(LM.T.dot( Yb[eig_n:eig_n+n] + 1.0j*Yb[eig_n+n:eig_n+2*n]	))
		BCb = np.concatenate((BCb1,BCb2,BCb3,BCb4),axis = 0)
		#
		return np.array(BCa,dtype = float), np.array(BCb,dtype = float)
	
	
	# print 'Hello eigenfunction world'
	
	# print "Setting up problem"
	eigf_problem = scikits.bvp_solver.ProblemDefinition(
						num_ODE = 18,
					  	num_parameters = 0,
					  	num_left_boundary_conditions = 10,
					  	boundary_points = (0, s['I']),
					  	function = eigf_F,
					  	boundary_conditions = eigf_boundary_conditions)
	
	
	# print "Solving the problem"
	eigf_solution = scikits.bvp_solver.solve(
						eigf_problem,
					  	solution_guess = s['guess'],
					  	initial_mesh = np.linspace(0,s['I'],p['temp_n']),
					  	trace = 1,
					  	tolerance=1e-9,
					  	max_subintervals=1200,
#						parameter_guess = np.array([15.0]),
					  	error_on_fail=True)
	
	
	# print "Returning from eigenfunction world"
	return eigf_problem, eigf_solution



