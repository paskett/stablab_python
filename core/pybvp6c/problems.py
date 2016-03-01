from __future__ import print_function, division
import math, sys
from math import cos,pi

import numpy as np
np.set_printoptions(precision=15)
from scipy.io import loadmat, savemat,whosmat
port_path = '/Users/joshualytle/bin/projects/pystablab/core/pybvp6c/'

from core.pybvp6c.bvp6c import bvp6c, bvpinit, deval
from core.pybvp6c.structure_variable import *

import matplotlib.pyplot as plt

import contextlib
import cStringIO, pdb





@contextlib.contextmanager
def nostdout():
	save_stdout = sys.stdout
	sys.stdout = cStringIO.StringIO()
	yield
	sys.stdout = save_stdout

def prob1(flag=False):
	# BVP parameters
	epsilon = .01
	# ------------------------------------------------------------
	def analytic_sol(x): 
		out = np.exp(-x/math.sqrt(epsilon)) - np.exp((x-2)/math.sqrt(epsilon));
		return out/(1. - np.exp(-2./math.sqrt(epsilon)));

	# ------------------------------------------------------------
	def ode(x,y):
		return np.array([  y[1],
						   y[0]/epsilon ]);

	# ------------------------------------------------------------
	def f_jacobian(x,y):
		return np.array([ [0, 1],
							  [1./epsilon,0] ])

	# ------------------------------------------------------------
	def bc_jacobian(x,y):
		dGdya = np.array([ [1, 0],
						   [0, 0] ]);
		dGdyb = np.array([ [0, 0],
						   [1, 0] ]);
		return dGdya,dGdyb

	# ------------------------------------------------------------
	def bcs(ya,yb):
		return np.array([  ya[0]-1,
						   yb[0]		]);

	# ------------------------------------------------------------
	def init(x):
		return np.array([  1.-x,
							-1.	  ]);

	# ------------------------------------------------------------
	
	options = struct()
	# options include abstol, reltol, singularterm, stats, vectorized, maxnewpts,slopeout,xint
	options.abstol, options.reltol = 1e-8, 1e-7
	options.fjacobian, options.bcjacobian = f_jacobian, bc_jacobian
	options.nmax = 20000
	
	solinit = bvpinit(np.linspace(0,1,10),init)
	with nostdout():
		sol = bvp6c(ode,bcs,solinit,options)
		xint = np.linspace(0,1,100); Sxint,_ = deval(sol,xint)
	# savemat('np_vector.mat', {'x':sol.x,'y':sol.y,'xint':xint,'Sxint':Sxint})

	m_data = loadmat(port_path+'test_data/bvp_prob1.mat')
	try:
		grid_diff = np.max(np.abs(sol.x-m_data['x']))
		sol_diff = np.max(np.abs(sol.y-m_data['y']))
		grid_eval = np.max(np.abs(xint-m_data['xint']))
		sol_eval = np.max(np.abs(Sxint-m_data['Sxint']))
	except:
		raise ValueError
	if flag==True:
		# plt.plot(xint,Sxint[0],linewidth=2.0)
		# plt.axis([0, 1, -.1, 1.1])
		# plt.title('Numerical Solution')
		# plt.xlabel('x'); plt.ylabel('y'); plt.show()
		assert grid_diff < 1e-14,"Difference in grids = %e"%grid_diff
		assert sol_diff < 1e-14, "Difference in solutions = %e"%sol_diff
		assert grid_eval < 1e-14, "Difference in xint = %e"%grid_eval
		assert sol_eval < 1e-14, "Difference in Sxint = %e"%sol_eval
		print("Difference in grids = %e"%grid_diff)
		print("Difference in solutions = %e"%sol_diff)
		print("Difference in xint = %e"%grid_eval)
		print("Difference in Sxint = %e"%sol_eval)
		print("bvp_prob1 passes\n\n\n")
	return (grid_diff, sol_diff, grid_eval, sol_eval)


def prob4(flag=False):
	exp = math.exp
	# BVP parameters
	epsilon = .01
	# ------------------------------------------------------------
	# def analytic_sol(x):
	#	out = np.exp(-x/math.sqrt(epsilon)) - np.exp((x-2)/math.sqrt(epsilon));
	#	return out/(1. - np.exp(-2./math.sqrt(epsilon)));

	# ------------------------------------------------------------
	def ode(x,y):
		return np.array([  y[1],
						   -y[1]/epsilon +(1.+epsilon)/epsilon*y[0] ]);

	# ------------------------------------------------------------
	def f_jacobian(x,y):
		return np.array([ [0, 1],
							  [(1.+epsilon)/epsilon,-1./epsilon] ])

	# ------------------------------------------------------------
	def bc_jacobian(x,y):
		dGdya = np.array([ [1, 0],
						   [0, 0] ]);
		dGdyb = np.array([ [0, 0],
						   [1, 0] ]);
		return dGdya,dGdyb

	# ------------------------------------------------------------
	def bcs(ya,yb):
		return np.array([  ya[0] - (1.+exp(-2.)),
						   yb[0] - (1.+exp( -2.*(1.+epsilon)/epsilon ))		]);

	# ------------------------------------------------------------
	def init(x):
		return np.array([  1.-x,
							-1.	  ]);

	# ------------------------------------------------------------
	
	options = struct()
	# options include abstol, reltol, singularterm, stats, vectorized, maxnewpts,slopeout,xint
	options.abstol, options.reltol = 1e-8, 1e-7
	options.fjacobian, options.bcjacobian = f_jacobian, bc_jacobian
	options.nmax = 20000
	
	solinit = bvpinit(np.linspace(-1,1,10),init)
	with nostdout():
		sol = bvp6c(ode,bcs,solinit,options)
		xint = np.linspace(-1,1,100); Sxint,_ = deval(sol,xint)
	# savemat('np_vector.mat', {'x':sol.x,'y':sol.y,'xint':xint,'Sxint':Sxint})
	
	m_data = loadmat(port_path+'test_data/bvp_prob4.mat')
	grid_diff = np.max(np.abs(sol.x-m_data['x']))
	sol_diff = np.max(np.abs(sol.y-m_data['y']))
	grid_eval = np.max(np.abs(xint-m_data['xint']))
	sol_eval = np.max(np.abs(Sxint-m_data['Sxint']))
	if flag==True:
		# plt.plot(sol.x,sol.y[0],linewidth=2.0)
		# plt.axis([-1.1, 1.1, 0., 1.2])
		# plt.title('Numerical Solution')
		# plt.xlabel('x'); plt.ylabel('y'); plt.show()
		assert grid_diff < 1e-14,"Difference in grids = %e"%grid_diff
		assert sol_diff < 1e-13, "Difference in solutions = %e"%sol_diff
		assert grid_eval < 1e-14, "Difference in xint = %e"%grid_eval
		assert sol_eval < 1e-13, "Difference in Sxint = %e"%sol_eval
		print("Difference in grids = %e"%grid_diff)
		print("Difference in solutions = %e"%sol_diff)
		print("Difference in xint = %e"%grid_eval)
		print("Difference in Sxint = %e"%sol_eval)
		print("bvp_prob4 passes\n\n\n")

	return (grid_diff, sol_diff,grid_eval, sol_eval)


def prob7(flag=False):
	exp, pi = math.exp, math.pi
	cos, sin = math.cos, math.sin
	# BVP parameters
	epsilon = .0001 # .001, .0001
	# ------------------------------------------------------------
	# def analytic_sol(x):
	#	out = np.exp(-x/math.sqrt(epsilon)) - np.exp((x-2)/math.sqrt(epsilon));
	#	return out/(1. - np.exp(-2./math.sqrt(epsilon)));

	# ------------------------------------------------------------
	def ode(x,y):
		f = -epsilon**(-1.)*( (1. + epsilon*pi**2.)*cos(pi*x) + pi*x*sin(pi*x)	)
		return np.array([  y[1],
						   -x*y[1]/epsilon + 1./epsilon*y[0] + f	]);

	# ------------------------------------------------------------
	def f_jacobian(x,y):
		return np.array([ [0, 1],
							  [1./epsilon,-x/epsilon] ])

	# ------------------------------------------------------------
	def bc_jacobian(x,y):
		dGdya = np.array([ [1, 0],
						   [0, 0] ]);
		dGdyb = np.array([ [0, 0],
						   [1, 0] ]);
		return dGdya,dGdyb

	# ------------------------------------------------------------
	def bcs(ya,yb):
		return np.array([  ya[0] + 1.,
						   yb[0] - 1.		]);

	# ------------------------------------------------------------
	def init(x):
		return np.array([  1.-x,
							-1.	  ]);

	# ------------------------------------------------------------
	
	options = struct()
	# options include abstol, reltol, singularterm, stats, vectorized, maxnewpts,slopeout,xint
	options.abstol, options.reltol = 1e-8, 1e-7
	options.fjacobian, options.bcjacobian = f_jacobian, bc_jacobian
	options.nmax = 20000
	
	solinit = bvpinit(np.linspace(-1,1,10),init)
	with nostdout():
		sol = bvp6c(ode,bcs,solinit,options)
		xint = np.linspace(-1,1,100); Sxint,_ = deval(sol,xint)
	# savemat('np_vector.mat', {'x':sol.x,'y':sol.y,'xint':xint,'Sxint':Sxint})
	
	m_data = loadmat(port_path+'test_data/bvp_prob7.mat')
	grid_diff = np.max(np.abs(sol.x-m_data['x']))
	sol_diff = np.max(np.abs(sol.y-m_data['y']))
	grid_eval = np.max(np.abs(xint-m_data['xint']))
	sol_eval = np.max(np.abs(Sxint-m_data['Sxint']))
	if flag==True:
		# plt.plot(sol.x,sol.y[1],linewidth=2.0)
		# # plt.axis([-1.1, 1.1, 0., 1.2])
		# plt.title('Numerical Solution')
		# plt.xlabel('x'); plt.ylabel('y'); plt.show()
		assert grid_diff < 1e-10,"Difference in grids = %e"%grid_diff
		assert sol_diff < 1e-10, "Difference in solutions = %e"%sol_diff
		assert grid_eval < 1e-10, "Difference in xint = %e"%grid_eval
		assert sol_eval < 1e-10, "Difference in Sxint = %e"%sol_eval
		print("Difference in grids = %e"%grid_diff)
		print("Difference in solutions = %e"%sol_diff)
		print("Difference in xint = %e"%grid_eval)
		print("Difference in Sxint = %e"%sol_eval)
		print("bvp_prob7 passes\n\n\n")
		

	return (grid_diff, sol_diff,grid_eval, sol_eval)



def prob20(flag=False):
	# exp, pi = math.exp, math.pi
	# cos, sin = math.cos, math.sin
	# log, cosh = math.log, math.cosh
	
	# BVP parameters
	epsilon = .06 # .5, .3, .06
	alpha = 1 + epsilon*math.log(math.cosh(-.745/epsilon))
	beta = 1 + epsilon*math.log(math.cosh(.255/epsilon))
	# ------------------------------------------------------------
	# def analytic_sol(x):
	#	out = np.exp(-x/math.sqrt(epsilon)) - np.exp((x-2)/math.sqrt(epsilon));
	#	return out/(1. - np.exp(-2./math.sqrt(epsilon)));

	# ------------------------------------------------------------
	def ode(x,y):
		return np.array([  y[1],
						   -y[1]**2./epsilon + 1./epsilon ]);

	# ------------------------------------------------------------
	def f_jacobian(x,y):
		return np.array([ [0, 1],
							  [0.,-2.*y[1]/epsilon] ])

	# ------------------------------------------------------------
	def bc_jacobian(x,y):
		dGdya = np.array([ [1, 0],
						   [0, 0] ]);
		dGdyb = np.array([ [0, 0],
						   [1, 0] ]);
		return dGdya,dGdyb

	# ------------------------------------------------------------
	def bcs(ya,yb):
		return np.array([  ya[0] - alpha,
						   yb[0] - beta		]);

	# ------------------------------------------------------------
	def init(x):
		return np.array([  1.-x,
							-1.	  ]);

	# ------------------------------------------------------------
	
	options = struct()
	# options include abstol, reltol, singularterm, stats, vectorized, maxnewpts,slopeout,xint
	options.abstol, options.reltol = 1e-8, 1e-7
	options.fjacobian, options.bcjacobian = f_jacobian, bc_jacobian
	options.nmax = 20000
	
	solinit = bvpinit(np.linspace(0,1,10),init)
	with nostdout():
		sol = bvp6c(ode,bcs,solinit,options)
		xint = np.linspace(0,1,100); Sxint,_ = deval(sol,xint)
	# savemat('np_vector.mat', {'x':sol.x,'y':sol.y,'xint':xint,'Sxint':Sxint})
	
	m_data = loadmat(port_path+'test_data/bvp_prob20.mat')
	grid_diff = np.max(np.abs(sol.x-m_data['x']))
	sol_diff = np.max(np.abs(sol.y-m_data['y']))
	grid_eval = np.max(np.abs(xint-m_data['xint']))
	sol_eval = np.max(np.abs(Sxint-m_data['Sxint']))
	if flag==True:
		# print(sol.x.shape,sol.y[0].shape)
		# plt.plot(sol.x,sol.y[0],linewidth=2.0)
		# plt.axis([-0.02, 1.02, .95, 1.8])
		# plt.title('Numerical Solution')
		# plt.xlabel('x'); plt.ylabel('y'); plt.show()
		assert grid_diff < 1e-10,"Difference in grids = %e"%grid_diff
		assert sol_diff < 1e-10, "Difference in solutions = %e"%sol_diff
		assert grid_eval < 1e-10, "Difference in xint = %e"%grid_eval
		assert sol_eval < 1e-10, "Difference in Sxint = %e"%sol_eval
		print("Difference in grids = %e"%grid_diff)
		print("Difference in solutions = %e"%sol_diff)
		print("Difference in xint = %e"%grid_eval)
		print("Difference in Sxint = %e"%sol_eval)
		print("bvp_prob20 passes\n\n\n")
	return(grid_diff, sol_diff,grid_eval, sol_eval)




def prob27(flag=False):
	# Parameters
	epsilon = .005 
	alpha, beta = 1., 1/3.
	# ------------------------------------------------------------
	def ode(x,y):
		return np.array([  y[1],
	         			   1./epsilon*y[0]*(1.-y[1]) ]);
						   
	# ------------------------------------------------------------
	def f_jacobian(x,y):
		return np.array([ [0, 1],
	            		  [1./epsilon*(1-y[1]),-y[0]/epsilon] ])
						  
	# ------------------------------------------------------------
	def bc_jacobian(x,y):
		dGdya = np.array([ [1, 0],
	          		   	   [0, 0] ]);
		dGdyb = np.array([ [0, 0],
	          		   	   [1, 0] ]);
		return dGdya,dGdyb

	# ------------------------------------------------------------
	def bcs(ya,yb):
		return np.array([  ya[0] - alpha,
	         			   yb[0] - beta		]);

	# ------------------------------------------------------------
	def init(x):
		return np.array([  8/3.*x**2.-10/3.*x+1.,
	    		  	    	16/3.*x-10/3.   ]);

	# ------------------------------------------------------------
	
	options = struct()
	# options include abstol, reltol, singularterm, stats, vectorized, maxnewpts,slopeout,xint
	options.abstol, options.reltol = 1e-8, 1e-7
	options.fjacobian, options.bcjacobian = f_jacobian, bc_jacobian
	options.nmax = 20000
	
	solinit = bvpinit(np.linspace(0,1,30),init)
	with nostdout():
		sol = bvp6c(ode,bcs,solinit,options)
		xint = np.linspace(0,1,100); Sxint,_ = deval(sol,xint)
	# savemat('np_vector.mat', {'x':sol.x,'y':sol.y,'xint':xint,'Sxint':Sxint})
	
	m_data = loadmat(port_path+'test_data/bvp_prob27.mat')
	grid_diff = np.max(np.abs(sol.x-m_data['x']))
	sol_diff = np.max(np.abs(sol.y-m_data['y']))
	grid_eval = np.max(np.abs(xint-m_data['xint']))
	sol_eval = np.max(np.abs(Sxint-m_data['Sxint']))
	if flag==True:
		# plt.plot(sol.x,sol.y[0],linewidth=2.0)
		# # plt.axis([-0.02, 1.02, .95, 1.8])
		# plt.title('Numerical Solution')
		# plt.xlabel('x'); plt.ylabel('y'); plt.show()
		assert grid_diff < 1e-10,"Difference in grids = %e"%grid_diff
		assert sol_diff < 1e-10, "Difference in solutions = %e"%sol_diff
		assert grid_eval < 1e-10, "Difference in xint = %e"%grid_eval
		assert sol_eval < 1e-10, "Difference in Sxint = %e"%sol_eval
		print("Difference in grids = %e"%grid_diff)
		print("Difference in solutions = %e"%sol_diff)
		print("Difference in xint = %e"%grid_eval)
		print("Difference in Sxint = %e"%sol_eval)
		print("bvp_prob27 passes\n\n\n")
	
	
	return (grid_diff, sol_diff,grid_eval, sol_eval)







def prob24(flag=False):
	# Parameters
	epsilon, gamma = .008, 1.4
	alpha, beta = .9129, .375
	# ------------------------------------------------------------
	def ode(x,y):
		return np.array([  y[1],
						   1./(1+x**2.)*((1+gamma)/(2*epsilon)-2*x)*y[1]-y[1]*y[0]**(-2.)/(epsilon*(1+x**2.))-
						   	    2.*x/(epsilon*(1+x**2.)**2.)*(y[0]**(-1.)-(gamma-1)/2.*y[0])
							 ]);
						   
	# ------------------------------------------------------------
	def f_jacobian(x,y):
		return np.array([ [0, 1],
	            		  [ 2.*y[1]*y[0]**(-3.)/(epsilon*(1+x**2))+epsilon**(-1)*2*x*(1+x**2)**(-2)*(y[0]**(-2)+(gamma-1)/2.) ,
						   (1+x**2.)**(-1)*(epsilon**(-1)*(1+gamma)/2.-2*x)-y[0]**(-2.)/(epsilon*(1+x**2))  ] 
					  ])
						  
	# ------------------------------------------------------------
	def bc_jacobian(x,y):
		dGdya = np.array([ [1, 0],
	          		   	   [0, 0] ]);
		dGdyb = np.array([ [0, 0],
	          		   	   [1, 0] ]);
		return dGdya,dGdyb

	# ------------------------------------------------------------
	def bcs(ya,yb):
		return np.array([  ya[0] - alpha,
	         			   yb[0] - beta		]);

	# ------------------------------------------------------------
	def init(x):
		return np.array([  (.375-.9129)*x+.9129,
	    		  	    	(.375-.9129)   ]);

	# ------------------------------------------------------------
	
	options = struct()
	# options include abstol, reltol, singularterm, stats, vectorized, maxnewpts,slopeout,xint
	options.abstol, options.reltol = 1e-8, 1e-7
	options.fjacobian, options.bcjacobian = f_jacobian, bc_jacobian
	options.nmax = 20000
	
	solinit = bvpinit(np.linspace(0,1,80),init)
	with nostdout():
		sol = bvp6c(ode,bcs,solinit,options)
		xint = np.linspace(0,1,100); Sxint,_ = deval(sol,xint)
	# savemat('np_vector.mat', {'x':sol.x,'y':sol.y,'xint':xint,'Sxint':Sxint})
	
	m_data = loadmat(port_path+'test_data/bvp_prob24.mat')
	grid_diff = np.max(np.abs(sol.x-m_data['x']))
	sol_diff = np.max(np.abs(sol.y-m_data['y']))
	grid_eval = np.max(np.abs(xint-m_data['xint']))
	sol_eval = np.max(np.abs(Sxint-m_data['Sxint']))
	if flag==True:
		# plt.plot(sol.x,sol.y[0],linewidth=2.0)
		# # plt.axis([-0.02, 1.02, .95, 1.8])
		# plt.title('Numerical Solution')
		# plt.xlabel('x'); plt.ylabel('y'); plt.show()
		assert grid_diff < 1e-10,"Difference in grids = %e"%grid_diff
		assert sol_diff < 1e-10, "Difference in solutions = %e"%sol_diff
		assert grid_eval < 1e-10, "Difference in xint = %e"%grid_eval
		assert sol_eval < 1e-10, "Difference in Sxint = %e"%sol_eval
		print("Difference in grids = %e"%grid_diff)
		print("Difference in solutions = %e"%sol_diff)
		print("Difference in xint = %e"%grid_eval)
		print("Difference in Sxint = %e"%sol_eval)
		print("bvp_prob24 passes\n\n\n")
	
	
	return (grid_diff, sol_diff,grid_eval, sol_eval)



def prob31(flag=False):
	cos, sin, tan = math.cos, math.sin, math.tan
	sec = lambda x: 1./cos(x)
	# Parameters
	epsilon = .05
	alpha, beta = 0., 0.
	# ------------------------------------------------------------
	def ode(x,y):
		return np.array([  sin(y[1]),
						   y[2],
          				   -epsilon**(-1)*y[3],
          				   epsilon**(-1)*( (y[0]-1)*cos(y[1])- y[2]*(sec(y[1]) + epsilon*y[3]*tan(y[1])) )
            		   		]);
						   
	# ------------------------------------------------------------
	def f_jacobian(x,y):
		return np.array([  [0, cos(y[1]), 0, 0],
            [0, 0, 1., 0],
            [0, 0, 0, -epsilon**(-1.)],
            [epsilon**(-1.)*cos(y[1]), 
            epsilon**(-1)*( (1-y[0])*sin(y[1])- y[2]*(sec(y[1])*tan(y[1]) + epsilon*y[3]*(sec(y[1]))**2.) ),  
            -epsilon**(-1)*(sec(y[1]) + epsilon*y[3]*tan(y[1])),
            -y[2]*tan(y[1])]
            ])
						  
	# ------------------------------------------------------------
	def bc_jacobian(x,y):
		dGdya = np.array([ [1, 0, 0, 0],
	            		   [0, 0, 1, 0], 
	            		   [0, 0, 0, 0],
	            		   [0, 0, 0, 0] ]);
		dGdyb = np.array([ [0, 0, 0, 0],
	            		   [0, 0, 0, 0],
	            		   [1, 0, 0, 0],
	            		   [0, 0, 1, 0] ]);
		return dGdya,dGdyb

	# ------------------------------------------------------------
	def bcs(ya,yb):
		return np.array([  ya[0] - alpha,
						   ya[2] - alpha,
	         			   yb[0] - beta,
					       yb[2] - beta
					   		]);

	# ------------------------------------------------------------
	def init(x):
		return np.array([  -(1/5.)*(x-.5)**2.+.05,
	    		  	    	(16/5.)*(x-.5)**3./3.-.8*x+.45,
							(16/5.)*(x-.5)**2.-.8,
							-epsilon*(32/5.)*(x-.5)
						   ]);

	# ------------------------------------------------------------
	
	options = struct()
	# options include abstol, reltol, singularterm, stats, vectorized, maxnewpts,slopeout,xint
	options.abstol, options.reltol = 1e-8, 1e-7
	options.fjacobian, options.bcjacobian = f_jacobian, bc_jacobian
	options.nmax = 20000
	
	solinit = bvpinit(np.linspace(0,1,10),init)
	with nostdout():
		sol = bvp6c(ode,bcs,solinit,options)
		xint = np.linspace(0,1,100); Sxint,_ = deval(sol,xint)
	# savemat('np_vector.mat', {'x':sol.x,'y':sol.y,'xint':xint,'Sxint':Sxint})
	
	m_data = loadmat(port_path+'test_data/bvp_prob31.mat')
	grid_diff = np.max(np.abs(sol.x-m_data['x']))
	sol_diff = np.max(np.abs(sol.y-m_data['y']))
	grid_eval = np.max(np.abs(xint-m_data['xint']))
	sol_eval = np.max(np.abs(Sxint-m_data['Sxint']))
	if flag==True:
		# for j in range(4):
		# 	plt.plot(sol.x,sol.y[j],linewidth=2.0)
		# # plt.axis([-0.02, 1.02, .95, 1.8])
		# plt.title('Numerical Solution')
		# plt.xlabel('x'); plt.ylabel('y'); plt.show()
		assert grid_diff < 1e-10,"Difference in grids = %e"%grid_diff
		assert sol_diff < 1e-10, "Difference in solutions = %e"%sol_diff
		assert grid_eval < 1e-10, "Difference in xint = %e"%grid_eval
		assert sol_eval < 1e-10, "Difference in Sxint = %e"%sol_eval
		print("Difference in grids = %e"%grid_diff)
		print("Difference in solutions = %e"%sol_diff)
		print("Difference in xint = %e"%grid_eval)
		print("Difference in Sxint = %e"%sol_eval)
		print("bvp_prob31 passes\n\n\n")
	
	
	return (grid_diff, sol_diff,grid_eval, sol_eval)







def test_ndtools_prob31(flag=False):
	# prob31 using numdifftools to construct a Jacobian
	import numdifftools as nd
	cos, sin, tan = math.cos, math.sin, math.tan
	sec = lambda x: 1./cos(x)
	# Parameters
	epsilon = .05
	alpha, beta = 0., 0.
	# ------------------------------------------------------------
	def ode(x,y):
		return np.array([  sin(y[1]),
						   y[2],
          				   -epsilon**(-1)*y[3],
          				   epsilon**(-1)*( (y[0]-1)*cos(y[1])- y[2]*(sec(y[1]) + epsilon*y[3]*tan(y[1])) )
            		   		]);
						   
	# ------------------------------------------------------------
	# def f_jacobian(x,y):
	# 	return np.array([  [0, cos(y[1]), 0, 0],
	#             [0, 0, 1., 0],
	#             [0, 0, 0, -epsilon**(-1.)],
	#             [epsilon**(-1.)*cos(y[1]),
	#             epsilon**(-1)*( (1-y[0])*sin(y[1])- y[2]*(sec(y[1])*tan(y[1]) + epsilon*y[3]*(sec(y[1]))**2.) ),
	#             -epsilon**(-1)*(sec(y[1]) + epsilon*y[3]*tan(y[1])),
	#             -y[2]*tan(y[1])]
	#             ])
	def f_jacobian(x,y):
		g = nd.Jacobian(lambda z: ode(x,z) )
		return g(y)
	# ------------------------------------------------------------
	def bc_jacobian(x,y):
		dGdya = np.array([ [1, 0, 0, 0],
	            		   [0, 0, 1, 0], 
	            		   [0, 0, 0, 0],
	            		   [0, 0, 0, 0] ]);
		dGdyb = np.array([ [0, 0, 0, 0],
	            		   [0, 0, 0, 0],
	            		   [1, 0, 0, 0],
	            		   [0, 0, 1, 0] ]);
		return dGdya,dGdyb

	# ------------------------------------------------------------
	def bcs(ya,yb):
		return np.array([  ya[0] - alpha,
						   ya[2] - alpha,
	         			   yb[0] - beta,
					       yb[2] - beta
					   		]);

	# ------------------------------------------------------------
	def init(x):
		return np.array([  -(1/5.)*(x-.5)**2.+.05,
	    		  	    	(16/5.)*(x-.5)**3./3.-.8*x+.45,
							(16/5.)*(x-.5)**2.-.8,
							-epsilon*(32/5.)*(x-.5)
						   ]);

	# ------------------------------------------------------------
	
	options = struct()
	# options include abstol, reltol, singularterm, stats, vectorized, maxnewpts,slopeout,xint
	options.abstol, options.reltol = 1e-8, 1e-7
	options.fjacobian, options.bcjacobian = f_jacobian, bc_jacobian
	options.nmax = 20000
	
	solinit = bvpinit(np.linspace(0,1,10),init)
	with nostdout():
		sol = bvp6c(ode,bcs,solinit,options)
		xint = np.linspace(0,1,100); Sxint,_ = deval(sol,xint)
	# savemat('np_vector.mat', {'x':sol.x,'y':sol.y,'xint':xint,'Sxint':Sxint})
	
	m_data = loadmat(port_path+'test_data/bvp_prob31.mat')
	grid_diff = np.max(np.abs(sol.x-m_data['x']))
	sol_diff = np.max(np.abs(sol.y-m_data['y']))
	grid_eval = np.max(np.abs(xint-m_data['xint']))
	sol_eval = np.max(np.abs(Sxint-m_data['Sxint']))
	if flag==True:
		for j in range(4):
			plt.plot(sol.x,sol.y[j],linewidth=2.0)
		# plt.axis([-0.02, 1.02, .95, 1.8])
		plt.title('Numerical Solution')
		plt.xlabel('x'); plt.ylabel('y'); plt.show()
		assert grid_diff < 1e-10,"Difference in grids = %e"%grid_diff
		assert sol_diff < 1e-10, "Difference in solutions = %e"%sol_diff
		assert grid_eval < 1e-10, "Difference in xint = %e"%grid_eval
		assert sol_eval < 1e-10, "Difference in Sxint = %e"%sol_eval
		print("Difference in grids = %e"%grid_diff)
		print("Difference in solutions = %e"%sol_diff)
		print("Difference in xint = %e"%grid_eval)
		print("Difference in Sxint = %e"%sol_eval)
		print("bvp_prob31 passes\n\n\n")
	
	
	return (grid_diff, sol_diff,grid_eval, sol_eval)








def measles_seir(flag=False):
	# prob31 using numdifftools to construct a Jacobian
	# cos, sin, tan = math.cos, math.sin, math.tan
	# sec = lambda x: 1./cos(x)
	beta0 = 1575
	beta1 = 1.
	eta = 0.01
	lmbda = .0279
	mu = .02
	# print(mu)
	def beta(t):
		return beta0*(1. + beta1*cos(2*pi*t))
	
	# ------------------------------------------------------------
	def ode(x,y):
		return np.array([  mu - beta(x)*y[0]*y[2],
						   beta(x)*y[0]*y[2] - y[1]/lmbda,
						   y[1]/lmbda - y[2]/eta,
						   0.,
						   0., 
						   0.
							]);
						   
	# ------------------------------------------------------------
	def f_jacobian(x,y):
		out = np.array([ [ -beta(x)*y[2], 0, -beta(x)*y[0],	 0,0,0	  ],
						 [ beta(x)*y[2], -1./lmbda, beta(x)*y[0],  0,0,0	],
						 [ 0., 1./lmbda, -1/eta,  0,0,0	   ],
						 [0.,0.,0.,0.,0.,0.,],
						 [0.,0.,0.,0.,0.,0.,],
						 [0.,0.,0.,0.,0.,0.,]	])
		return out
	
	
	# ------------------------------------------------------------
	def bc_jacobian(x,y):
		dGdya = np.array([ [1, 0, 0, -1, 0, 0],
						   [0, 1, 0, 0, -1, 0], 
						   [0, 0, 1, 0, 0, -1],
						   [0, 0, 0, 0, 0, 0],
						   [0, 0, 0, 0, 0, 0],
						   [0, 0, 0, 0, 0, 0]	 ]);
		dGdyb = np.array([ [0, 0, 0, 0, 0, 0],
						   [0, 0, 0, 0, 0, 0],
						   [0, 0, 0, 0, 0, 0],
						   [1, 0, 0, -1, 0, 0],
						   [0, 1, 0, 0, -1, 0], 
						   [0, 0, 1, 0, 0, -1]	 ]);
		return dGdya, dGdyb

	# ------------------------------------------------------------
	def bcs(ya,yb):
		return np.array([  ya[0] - ya[3],
						   ya[1] - ya[4],
						   ya[2] - ya[5],
						   yb[0] - yb[3],
						   yb[1] - yb[4],
						   yb[2] - yb[5]
							]);

	# ------------------------------------------------------------
	# def init(x):
	#	return np.array([  -(1/5.)*(x-.5)**2.+.05,
	#						(16/5.)*(x-.5)**3./3.-.8*x+.45,
	#						(16/5.)*(x-.5)**2.-.8,
	#						-epsilon*(32/5.)*(x-.5)
	#					   ]);
	def init(x):
		S = .1 + .05 * np.cos(2. * pi * x)
		out = np.ones((6,len(x)))
		out[0] = .65 + .4*S
		out[1] = .05 * (1. - S)-.042
		out[2] = .05 * (1. - S)-.042
		out[3] = .075*out[3]
		out[4] = .005*out[4]
		out[5] = .005*out[5]
		return out
	# ------------------------------------------------------------
	if flag==False:
		x = np.linspace(0,1,100)
		guess = init(x)
		plt.plot(x,guess[0],'-k')
		# plt.plot(x,guess[1],'-k')
		# plt.plot(x,guess[2],'-k')
		# plt.plot(x,guess[3],'-r')
		# plt.plot(x,guess[4],'-r')
		# plt.plot(x,guess[5],'-r')
		plt.show()
	
	def init(x):
		S = .1 + .05 * cos(2. * pi * x)
		# out = np.ones((6,len(x)))
		# out[0] = .5*S
		# out[1] = .05 * (1. - S)
		# out[2] = .05 * (1. - S)
		# out[3] = .08*out[3]
		# out[4] = .08*out[4]
		# out[5] = .08*out[5]
		return np.array([.65 + .4*S, .05 * (1. - S)-.042, .05 * (1. - S)-.042, .075,.005,.005])
		
	options = struct()
	# options include abstol, reltol, singularterm, stats, vectorized, maxnewpts,slopeout,xint
	options.abstol, options.reltol = 1e-8, 1e-7
	options.fjacobian, options.bcjacobian = f_jacobian, bc_jacobian
	options.nmax = 20000

	solinit = bvpinit(np.linspace(0,1,100),init)
	# with nostdout():
	sol = bvp6c(ode,bcs,solinit,options)
	xint = np.linspace(0,1,100); Sxint,_ = deval(sol,xint)

	if flag==True:
		  for j in range(3):
			  plt.plot(sol.x,sol.y[j],linewidth=2.0)
		  plt.title('Numerical Solution')
		  plt.xlabel('x'); plt.ylabel('y'); plt.show()

	return

if __name__=="__main__":
	pass
	# prob1(flag=True)
	# prob4(flag=True)
	# prob7(flag=True)
	# prob20(flag=True)
	# prob27(flag=True)
	# prob24(flag=True)
	# prob31(flag=True)
	# test_ndtools_prob31(flag=True)
	# measles_seir(flag=True)