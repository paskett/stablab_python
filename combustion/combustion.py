from __future__ import division
import numpy as np
from scipy import linalg
# import scikits.bvp_solver
import numdifftools as nd
import matplotlib.pyplot as plt

from combustionEvansSystems import A

from core.bin import projection1, projection2
from core.roottracking.continuation import mod_drury
from core.pybvp6c.bvp6c import bvp6c, bvpinit, deval, struct

def init_prof_guess(x,s,p):
	alpha = .45
	speed = np.sqrt(2.0/(p['beta']*p['tau']))*np.exp(-p['beta']/2.0)
	
	UL, UR = [1.0/p['beta'], 0, 0],	 [0, 0, 0]
	a, c = (UL[0]+UR[0])/2.0, (UL[0]-UR[0])/2.0
	
	out = np.zeros(8)
	out[0] = a-c*np.tanh(alpha*(s['R']/s['I'])*x)
	out[1:3] = -c*alpha*np.cosh(alpha*(s['R']/s['I'])*x)**(-2.)
	out[3] = speed
	out[4] = a-c*np.tanh(alpha*(s['L']/s['I'])*x)
	out[5:6] = -c*alpha*np.cosh(alpha*(s['L']/s['I'])*x)**(-2.) 
	out[7] = speed
	return out


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


def double_profile_F(x,y,s,p):
	n=s['n']
	out = np.concatenate((
							(s['R']/s['I'])*profile_F( x,y[:n],p ),
							(s['L']/s['I'])*profile_F( x,y[n:],p ) ))
	return out


def double_F_jacobian(x,y,s,p):					# Testing numdifftools
	try:
		g = nd.Jacobian(lambda z:double_profile_F(x,z,s,p),
											delta=np.array([1e-7,1e-7]))
		return g(y)
	except:
		g = nd.Jacobian(lambda z:double_profile_F(x,z,s,p),				 
											delta=np.array([1e-8,1e-8])) 
		return g(y)
	

def Flinear(y,speed,p):
	beta, tau = p['beta'], p['tau']
	if y[0] == 0: 
		e = 0.
	elif y[0] > 0: 
		e = np.exp(-beta)
	out = np.array([
	  [ 0,				1,					  0 ],
	  [beta*e,			  -speed+beta*e/speed,		  tau*e/speed], 
	  [-beta**2*e/tau,	 -beta**2*e/(tau*speed),  -speed/tau - beta*e/speed]])
	return out


def profile_bcs(Ya,Yb, s,p):
	n, speed = s['n'], Ya[3]
	UL, UR = [1.0/p['beta'], 0, 0], [0, 0, 0]
	#
	AM, AP = Flinear(np.array(UL), speed, p), Flinear(np.array(UR), speed, p)
	Pm, _ = projection1(AM,-1,0.0) # Decay manifold at - infty
	Pp, _ = projection1(AP,0,1e-8) # Growth manifold at + infty
	LM = linalg.orth(Pm.T)
	LP = linalg.orth(Pp.T)
	# Conditions at left endpoint
	BC = np.zeros((8,))
	# Conditions at left endpoint
	BC[:4] = Ya[0:n]-Ya[n:2*n]				#matching conditions = 4
	BC[4] = Ya[0] - 0.5/p['beta']			# phase condition = 1 
	# Conditions at right endpoint
	BC[5:7] = LM.T.dot(Yb[n:2*n-1] - UL)		# at -infty; = 2
	BC[7] = LP.T.dot(Yb[0:n-1]-UR)[0]				# at +infty; = 1
	return np.array(BC,dtype = float)
	

def bc2_jacobian(ya,yb,s,p):				  # Testing numdifftools
	try:
		ga = nd.Jacobian(lambda z:profile_bcs(z,yb,s,p),delta=np.array([1e-6,1e-6]))
		gb = nd.Jacobian(lambda z:profile_bcs(ya,z,s,p),delta=np.array([1e-6,1e-6]))
		return ga(ya), gb(yb)
	except:
		ga = nd.Jacobian(lambda z:profile_bcs(z,yb,s,p),delta=np.array([1e-8,1e-8]))
		gb = nd.Jacobian(lambda z:profile_bcs(ya,z,s,p),delta=np.array([1e-8,1e-8]))
		return ga(ya), gb(yb)


def eigf_F(x,y,s,p):
	n=s['n'] # = 4 = Dimension of first order profile equation
	eig_n = 2*n # = 8 = Dimension of first order eigenfunction equation
	
	out = np.zeros(18)
	out[0:4] = np.real( (s['R']/s['I'])*A((s['R']/s['I'])*x,(y[-2] + 1.0j*y[-1]),s,p).dot( np.array([y[0:n] + 1.0j*y[n:eig_n] ]).T ) )[:,0]
	out[4:8] = np.imag( (s['R']/s['I'])*A((s['R']/s['I'])*x,(y[-2] + 1.0j*y[-1]),s,p).dot( np.array([y[0:n] + 1.0j*y[n:eig_n] ]).T) )[:,0]
	out[8:12] = np.real( (s['L']/s['I'])*A((s['L']/s['I'])*x,(y[-2] + 1.0j*y[-1]),s,p).dot( np.array([y[eig_n:eig_n+n] + 1.0j*y[eig_n+n:2*eig_n] ]).T) )[:,0]
	out[12:16] = np.imag( (s['L']/s['I'])*A((s['L']/s['I'])*x,(y[-2] + 1.0j*y[-1]),s,p).dot( np.array([y[eig_n:eig_n+n] + 1.0j*y[eig_n+n:2*eig_n] ]).T) )[:,0]
	return out


def double_eigfF_jacobian(x,y,s,p):					# Testing numdifftools
	try:
		g = nd.Jacobian(lambda z:eigf_F(x,z,s,p),
											delta=np.array([1e-7,1e-7]))
		return g(y)
	except:
		g = nd.Jacobian(lambda z:eigf_F(x,z,s,p),				 
											delta=np.array([1e-8,1e-8])) 
		return g(y)

	
def eigf_bcs(Ya,Yb,s,p):
	n=s['n'] # = 4 = Dimension of first order profile equation
	eig_n = 2*n # = 8 = Dimension of first order eigenfunction equation
	ph = s['ph']
	AM = A(s['L'], ((Ya[-2] + 1.0j*Ya[-1]) + (Yb[-2] + 1.0j*Yb[-1]))/2.0,s,p)
	AP = A(s['R'], ((Ya[-2] + 1.0j*Ya[-1]) + (Yb[-2] + 1.0j*Yb[-1]))/2.0,s,p)
	#
	P1,Q1 = projection2(AM,-1,1e-6); LM = linalg.orth(P1.T)
	P2,Q2 = projection2(AP,+1,1e-6); LP = linalg.orth(P2.T)
	#
	BCS = np.zeros(18)
	BCS[0:8] = (Ya[0:eig_n]-Ya[eig_n:2*eig_n]) # 8 matching conditions
	BCS[8] = Ya[ph[0,0]]- ph[0,1] 		# 2 phase conditions
	BCS[9] = Ya[ph[1,0]]- ph[1,1] 
	#
	# 8 projective conditions
	BCS[10:12] = np.real(LP.T.dot( Yb[0:n] + 1.0j*Yb[n:2*n] ))
	BCS[12:14] = np.imag(LP.T.dot( Yb[0:n] + 1.0j*Yb[n:2*n] ))
	BCS[14:16] = np.real(LM.T.dot( Yb[eig_n:eig_n+n] + 1.0j*Yb[eig_n+n:eig_n+2*n]	))
	BCS[16:18] = np.imag(LM.T.dot( Yb[eig_n:eig_n+n] + 1.0j*Yb[eig_n+n:eig_n+2*n]	))
	return BCS

def eigfF_bcs_jacobian(ya,yb,s,p):				  # Testing numdifftools
	try:
		ga = nd.Jacobian(lambda z:eigf_bcs(z,yb,s,p),delta=np.array([1e-6,1e-6]))
		out1 = ga(ya)
		gb = nd.Jacobian(lambda z:eigf_bcs(ya,z,s,p),delta=np.array([1e-6,1e-6]))
		out2 = gb(yb)
	except:
		ga = nd.Jacobian(lambda z:eigf_bcs(z,yb,s,p),delta=np.array([1e-8,1e-8]))
		out1 = ga(ya)
		gb = nd.Jacobian(lambda z:eigf_bcs(ya,z,s,p),delta=np.array([1e-8,1e-8]))
		out2 = gb(yb)
	return out1, out2 

def plot_soln(xint,Sxint,s,p,save=False):
	color = ['-k','-b','-g','-r']
	for j in range(3):
		plt.plot((s['R']/s['I'])*xint, Sxint[j],color[j],	linewidth=1.7)
		plt.plot((s['L']/s['I'])*xint, Sxint[j+4],color[j], linewidth=1.7)
	if save==True:
		plt.savefig(package_directory+'/combustion/data_combustion/'+'beta_'+
										str(p['beta'])+'.pdf')
	else:
		plt.show()
	plt.clf()


def plot_soln_eigf(xint,Sxint,s,p,save=False):
	color = ['-k','-b','-g','-r','-c','-m','-y','-*k']
	for j in range(8):
		plt.plot((s['R']/s['I'])*xint, Sxint[j],color[j],	linewidth=1.7)
		plt.plot((s['L']/s['I'])*xint, Sxint[j+8],color[j], linewidth=1.7)
	if save==True:
		plt.savefig(package_directory+'/combustion/data_combustion/'+"eigf_"+
										'beta_'+str(p['beta'])+'.pdf')
	else:
		plt.show()
	plt.clf()


def plot_this_profile(solution, interval,n_points,filestring,s):
	x = np.linspace(interval[0],interval[1], n_points)
	y1 = solution(x)
	y2 = solution(-x)

	for j in range(0,s['n']):
		plt.plot(x, y1,'k')
		plt.plot(np.flipud(-x),np.flipud(y2) ,'k')
		
	format = '.eps'
	plt.savefig('Testing/'+filestring+format)
	plt.show(); plt.clf()
	return 

def combustion():
	s = {'I':16,'R':16,'L':-32,'side':1,'n':4}
	p = {'beta':3.5, 'tau': .1}
	
	options = struct()
	# options include abstol, reltol, singularterm, stats, vectorized, maxnewpts,slopeout,xint
	options.fjacobian = lambda x,y: double_F_jacobian(x,y,s,p)
	options.bcjacobian = lambda x,y: bc2_jacobian(x,y,s,p)
	options.abstol = 1e-7
	options.reltol = 1e-6
	options.nmax = 5000
	options.stats='on'

	solinit = bvpinit(np.linspace(0,s['I'],100),lambda x: init_prof_guess(x,s,p))
	s['sol'] = bvp6c(lambda x,y: double_profile_F(x,y,s,p),
					 lambda ya,yb: profile_bcs(ya,yb,s,p),
					 solinit,options)
	xint = np.linspace(0,s['I'],400); Sxint,_ = deval(s['sol'],xint)
	plot_soln(xint,Sxint,s,p)
	

def combustion_profile_continuation():
	s = {'I':16,'R':16,'L':-32,'side':1,'n':4}
	p = {'beta': 3.5, 'tau': .1}
	s['L'], s['R'] = -8.0*np.exp(0.54*p['beta']), 8.0*np.exp(0.54*p['beta'])
	
	beta_vals = [3.5, 4, 5, 6, 7, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 12.5, 13, 13.5] +list(np.linspace(14,18,40))
	
	options = struct()
	# options include abstol, reltol, singularterm, stats, vectorized, maxnewpts,slopeout,xint
	options.fjacobian = lambda x,y: double_F_jacobian(x,y,s,p)
	options.bcjacobian = lambda x,y: bc2_jacobian(x,y,s,p)
	options.abstol = 1e-7
	options.reltol = 1e-6
	options.nmax = 5000
	options.stats='on'

	solinit = bvpinit(np.linspace(0,s['I'],100),lambda x: init_prof_guess(x,s,p))
	s['sol'] = bvp6c(lambda x,y: double_profile_F(x,y,s,p),
					 lambda ya,yb: profile_bcs(ya,yb,s,p),
					 solinit,options)
	xint = np.linspace(0,s['I'],100); Sxint,_ = deval(s['sol'],xint)
	
	for j in range(len(beta_vals)):
		p['beta']=beta_vals[j]
		s['L']=-8.0*np.exp(0.54*p['beta'])
		s['R']=8.0*np.exp(0.54*p['beta'])
		
		options.fjacobian = lambda x,y: double_F_jacobian(x,y,s,p)
		options.bcjacobian = lambda x,y: bc2_jacobian(x,y,s,p)
		solinit  = bvpinit(xint,Sxint)
		s['sol'] = bvp6c(lambda x,y: double_profile_F(x,y,s,p),
						 lambda ya,yb: profile_bcs(ya,yb,s,p),
						 solinit,options)
		
		print "p['beta'] = ", p['beta']
		Sxint,_ = deval(s['sol'],xint)
		plot_soln(xint,Sxint,s,p,save=True)
	




	
def combustion_roottracking():
	#
	# Parameters 
	# 
	p={'beta':3.5,'tau': 0.1,'temp_n':200,'rooot':1e-4*(.12375+ 6.3313j)} #'rooot' for beta = 7.1
	
	#
	# Structure variables
	# 
	s={'I':16.0,'R':16.0,'L':-32.0, 'side':1,'n':4, 'rarray':range(0,4), 
		'larray':range(4,8)}
	s['Grarray'] = range(0,8)
	s['Glarray'] = range(8,16)
	s['L'], s['R']=-8.0*np.exp(0.54*p['beta']), 8.0*np.exp(0.54*p['beta'])
	
	s['PFunctions'] = 'combustionEvans' # Must be called before calling emcset
	[s,e,m,c] = emcset(s,'front',2,2,'reg_reg_polar')
	m['method'] = mod_drury; m['options'] = {'RelTol':1e-8,'AbsTol':1e-8}
	
	''' Profile computation; Continuation of traveling wave. '''
	options_F = struct()
	# options include abstol, reltol, singularterm, stats, vectorized, maxnewpts,slopeout,xint
	options_F.abstol, options_F.reltol = 1e-10, 1e-9
	options_F.fjacobian = lambda x,y: double_F_jacobian(x,y,s,p)
	options_F.bcjacobian = lambda x,y: bc2_jacobian(x,y,s,p)
	options_F.nmax = 5000
	options_F.stats='on'
	
	xint = np.linspace(0,s['I'],100)
	solinit = bvpinit(xint,lambda x: init_prof_guess(x,s,p))
	s['solution'] = bvp6c(lambda x,y: double_profile_F(x,y,s,p),
						  lambda ya,yb: profile_bcs(ya,yb,s,p),
						  solinit,options_F)
	Sxint,_ = deval(s['solution'],xint)
	plot_soln(xint,Sxint,s,p)
	
	beta_vals = [3.5, 4, 5, 6, 7, 7.1] 
	for j in range(len(beta_vals)):
		p['beta']=beta_vals[j]
		s['L']=-8.0*np.exp(0.54*p['beta'])
		s['R']=8.0*np.exp(0.54*p['beta'])

		options_F.fjacobian = lambda x,y: double_F_jacobian(x,y,s,p)
		options_F.bcjacobian = lambda x,y: bc2_jacobian(x,y,s,p)
		solinit  = bvpinit(xint,Sxint)
		s['solution'] = bvp6c(lambda x,y: double_profile_F(x,y,s,p),
						 lambda ya,yb: profile_bcs(ya,yb,s,p),
						 solinit,options_F)

		print "beta = ", p['beta']
		Sxint,_ = deval(s['solution'],xint)
		plot_soln(xint,Sxint,s,p,save=True)

	s['eigf']=True  
	s, x1, x2 = EvansDF.eigf_init_guess(s,e,m,c,p)
	
	##############################################################################
	
	if s['eigf']:
		for j in range(0,m['n']):
			plt.plot(x1,np.flipud(np.real(s['guess'][j+m['n'],:])) ,'k')
			plt.plot(np.flipud(x2), np.real(s['guess'][j,:]),'k')

		plt.title('Eigenfunction for the combustion equation')
		plt.xlabel('x'); plt.ylabel('W (Initial Guess: Hello)');
		format = '.eps'
		plt.savefig(package_directory+'/combustion/data_combustion/'+
									'combustion_eigenfunction_guess'+format)
		# plt.show()
		
		n=s['n']; eig_n = 2*n
		s['guess'] = np.concatenate( ( np.real(s['guess'][0:n,:]),
			np.imag(s['guess'][0:n,:]),
			np.real(s['guess'][n:eig_n,:]),
			np.imag(s['guess'][n:eig_n,:]),
			np.array([np.real(s['guess'][eig_n,:]) ]),
			np.array([np.imag(s['guess'][eig_n,:]) ]) ),
			axis=0 )
		
		s['ph'] = np.array([[0,s['guess'][0,0] ],
					   [4,s['guess'][4,0] ] ])
		del n,eig_n
	
	
	options_eigfF = struct()
	# options include abstol, reltol, singularterm, stats, vectorized, maxnewpts,slopeout,xint
	options_eigfF.abstol, options_eigfF.reltol = 1e-10, 1e-9
	options_eigfF.fjacobian = lambda x,y: double_F_jacobian(x,y,s,p)
	options_eigfF.bcjacobian = lambda x,y: bc2_jacobian(x,y,s,p)
	options_eigfF.nmax = 5000
	options_eigfF.stats='on'
	
	data = []
	beta_vals = list(np.linspace(7.1,18,219))
	total_time = 0
	import time
	start = time.time()
	for j in range(len(beta_vals)):
		p['beta']=beta_vals[j]
		s['L']=-8.0*np.exp(0.54*p['beta'])
		s['R']=8.0*np.exp(0.54*p['beta'])

		options_F.fjacobian = lambda x,y: double_F_jacobian(x,y,s,p)
		options_F.bcjacobian = lambda x,y: bc2_jacobian(x,y,s,p)
		solinit  = bvpinit(xint,Sxint)
		s['solution'] = bvp6c(lambda x,y: double_profile_F(x,y,s,p),
						 lambda ya,yb: profile_bcs(ya,yb,s,p),
						 solinit,options_F)
		Sxint,_ = deval(s['solution'],xint)

		options_eigfF.fjacobian = lambda x,y: double_eigfF_jacobian(x,y,s,p)
		options_eigfF.bcjacobian = lambda x,y: eigfF_bcs_jacobian(x,y,s,p)
		if j ==0:
			solinit  = bvpinit(np.linspace(0,s['I'],len(s['guess'][1])),s['guess'])
		else:
			solinit  = bvpinit(xint,eigf_Sxint)
		s['eigf_solution'] = bvp6c(lambda x,y: eigf_F(x,y,s,p),
						 		   lambda ya,yb: eigf_bcs(ya,yb,s,p),
						 		   solinit,options_eigfF)
		
		
		lmbda = s['eigf_solution'].y[16,-1] + 1.j*s['eigf_solution'].y[17,-1]
		end = time.time()
		total_time += end-start
		start = end
		
		# print "beta = ", p['beta']
		# print "lambda = ", lmbda
		data.append((p['beta'],lmbda))
		print "total_time = ", total_time
		print data
		eigf_Sxint,_ = deval(s['eigf_solution'],xint)
		plot_soln(xint,Sxint,s,p,save=True)
		plot_soln_eigf(xint,eigf_Sxint,s,p,save=True)
		
	
	return s,p
	
	
	
	# s['flag'] = 'NA'
	# s['flag'] = 'restarting'
	# File = open('combustion/combustion_Data/restart_combustion.pkl','rb')
	# comb_Dict = pickle.load(File)
	# p['beta'] = comb_Dict['beta']
	# s['I'] = comb_Dict['I'];
	# s['R'] = 8.4*np.exp(0.54*p['beta']);  s['L'] = -s['R']
	# s['solution'] = comb_Dict['solution']
	# p['speed'] = s['solution'](0)[3]
	#
	# s['eigf']=False
	#
	#
	# Inputs initial guess from Matlab
	# #########################################################################
	# input = open(r'../stablab20/Continuation_Testing/matlab/eigf_guess_real.txt','r')
	# aList = input.readlines()
	# s['matlab_guess'] = np.zeros((9,p['temp_n']),dtype='complex')
	# for j in range(0,p['temp_n']):
	# 	line = aList[j].rstrip().split()
	# 	for k in range(0,9):
	# 		s['matlab_guess'][k,j] = np.float64(line[k])
	#
	# input = open(r'../stablab20/Continuation_Testing/matlab/eigf_guess_imag.txt','r')
	# aList = input.readlines()
	# for j in range(0,p['temp_n']):
	# 	line = aList[j].rstrip().split()
	# 	for k in range(0,9):
	# 		s['matlab_guess'][k,j] = s['matlab_guess'][k,j] + 1j*np.float64(line[k])
	# np.set_printoptions(precision=12)
	# s['guess'] = s['matlab_guess']
	# print "s['guess'] = ", s['guess'].shape
	# print "s['matlab_guess'] = ", s['matlab_guess'].shape




if __name__ == "__main__":
	# # Both working
	combustion()
	# combustion_profile_continuation()
	# combustion_roottracking()




