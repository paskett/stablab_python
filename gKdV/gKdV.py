from __future__ import division
import sys

import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt

from core.contour import semicirc2, winding_number, Evans_plot
from core.evans import emcset, Evans_compute
from core.bin import projection2

from core.pybvp6c.bvp6c import bvp6c, bvpinit, deval
from core.pybvp6c.structure_variable import *
package_directory = '/Users/joshualytle/bin/projects/pystablab'
import gKdVEvansSystems as EvansSystems

def gKdV(domain=[-5,5], verbose=True):
	''' System Parameters '''
	p = {'p':10}
	
 
	''' Numerical Infinity '''
	assert domain[0] < domain[1]
	s = {'I':1,'R':domain[1],'L':domain[0]}
	
	''' Set STABLAB structures to local default values '''
	s['EvansSystems'] = EvansSystems # Must be called before calling emcset
	s,e,m,c = emcset(s,'front',2,1,'adj_reg_polar') 
	
	'''
	gKdVEvans.py contains profile/ode functions specific to 
	computing the Evans function for the gKdV equation
	'''
	
	#  
	# # refine the Evans function computation to achieve set relative error
	# c['refine'] = 'on'
	
	
	'''Create the Preimage Contour'''
	# circpnts=5; imagpnts=5; innerpnts = 5
	# r=10; spread=4; zerodist=10**(-2)
	# ksteps = 32; lambda_steps = 0
	# preimage=semicirc2(circpnts,imagpnts,innerpnts,ksteps,r,
	# 					spread,zerodist,lambda_steps)
	
	points = 50;
	preimage = (5.5+5*np.exp(
				2*np.pi*1j*np.linspace(0,0.5,points+(points-1)*c['ksteps']))
				)
	
	'''Compute the Evans function'''
	out = Evans_compute(preimage,c,s,p,m,e)
	
	# 
	# Display Evans function output and statistics
	# 
	w = np.concatenate(( out,np.flipud(np.conj(out)) ))
	if verbose: 
		pass
		# print 'Evans Computation Successful'
		# print 'The winding number is ', winding_number(w)
	
	titlestring = ('Parameters for the gKdV equation: \n '+
					'p = ' +str(p['p'])+', I = '+str(s['I'])  )
	filestring = (package_directory + '/gKdV/data_gKdV/'+'Parameters_'+
					str(p['p'])+'_'+str(s['I']))
	labelstring = 'Evans Output'
	plot_flag = False
	format = '.pdf' 		# Possible formats: png, pdf, ps, eps and svg.
	Evans_plot(w,labelstring,titlestring, filestring, format,Plot_B=plot_flag)
	return 


def gKdV_profile():
	''' System Parameters '''
	p = {'p':5.2}
	
 
	''' Numerical Infinity '''
	domain = [-20,20]
	assert domain[0] < domain[1]
	# s = {'I':domain[1],'R':domain[1],'L':domain[0],'side':1,
	# 		'larray':np.array([0,1,2,3]), 'rarray':np.array([4,5,6,7])}
	s = {'I':domain[1],'R':domain[1],'L':domain[0],'side':1,
			'larray':np.array([0,1]), 'rarray':np.array([2,3])}
	
	''' Set STABLAB structures to local default values '''
	s['EvansSystems'] = EvansSystems # Must be called before calling emcset
	s,e,m,c = emcset(s,'front',2,1,'adj_reg_polar') 
	
	'''
	gKdVEvans.py contains profile/ode functions specific to 
	computing the Evans function for the gKdV equation
	'''
	
	n=2
	# g = guess(0,p)
	ph = [1,0]#g[1]]
	sech = lambda x: 1./np.cosh(x)
	
	# % parameters
	# p.p = 5.2; %p.rooot = 0.098034924900383
	# p.target_p = 10;
	# p.s =1;  % Wave speed is scaled out. See Blake's thesis.

	# %
	# %
	# %
	# s['I'] = 20;
	# s['R']=s['I']; s['L']=-s['I'];
	# s.larray = 3:4; s.rarray = 1:2;
	# s.side=1; 
	# s.F=@F;
	# s.UL = [0;0]; s.UR = [0;0];

	[s,e,m,c] = emcset(s,'front',2,1,'reg_reg_polar');


	# m.method = @drury;
	# m.options['R']elTol = 1e-6; m.options.AbsTol = 1e-7;
	#
	# p_span = [0,0,p.p: .005: p.target_p];
	# %rec contains the parameter p, the eigenvalue lambda, and numerical
	# %infinity
	# S_I = linspace(s['I'],10,length(p_span));
	# rec = [p_span ; zeros(size(p_span)); S_I];
	# rec(2,1) = .08; rec(2,2) = .09;
	# % rec(2,3) %0.098034924900383;
	
	
	def double_F(x,y,s,p):
		# out = double_F(x,y,s,p)
		#
		# Returns the split domain for the ode given in the function F.
		#
		# Input "x" and "y" are provided by the ode solver.Note that s.rarray
		# should be [1,2,...,k] and s.larray should be [k+1,k+2,...,2k]. See
		# STABLAB documentation for more inforamtion about the structure s.
		print "y = ",y
		print "y[s['rarray']] = ",y[s['rarray']]
		print "y[s['larray']] = ",y[s['larray']]
		out = np.zeros(2*n)
		out[0:2] = (s['R']/s['I'])*F(x,y[s['rarray']],s,p)
		out[2:] = (s['L']/s['I'])*F(x,y[s['larray']],s,p)
		# out = np.array([(s['R']/s['I'])*F(x,y[s['rarray']],s,p),(s['L']/s['I'])*F(x,y[s['rarray']],s,p)])
		return out 
	
	def F(x,y,s,p):
		 return np.array([y[1], y[0]*(1 - (y[0]**(p['p']-1))/p['p'])])
		 
	def double_F_jacobian(x,y,p):
		out = np.zeros((4,4))
		out[0:2,0:2] = (s['R']/s['I'])*np.array([ [0,                  1],
						  						  [1-y[0]**(p['p']-1), 0]
					     					  	 ])
		out[2:,2:] = (s['L']/s['I'])*np.array([ [0,                  1],
						  						  [1-y[2]**(p['p']-1), 0]
					     					  	 ])
		return out
	# import numdifftools as nd
	#
	# def double_F_jacobian(x,y):					# Testing numdifftools
	# 	try:
	# 		g = nd.Jacobian(lambda z:double_F(x,z,s=s,p=p),delta=np.array([1e-9,1e-9]))
	# 		out = g(y)
	# 	except:
	# 		# print sys.exc_info()
	# 		# g = nd.Jacobian(lambda z:double_F(x,z,s=s,p=p),step_nom=.0005)
	# 		g = nd.Jacobian(lambda z:double_F(x,z,s=s,p=p),delta=np.array([1e-9,1e-9]))
	# 		out = g(y)
	#
	# 	return out
	
	def guess(x,p):
		gamma = 0
		out = np.array([(p['p']*(p['p']+1)/2)**(1/(p['p']-1))*(sech((1-p['p'])/2*x + gamma))**(2/(p['p']-1)),
	    (p['p']*(p['p']+1)/2)**(1/(p['p']-1))*(sech((1-p['p'])/2*x + gamma))**(2/(p['p']-1))*(np.tanh((1-p['p'])/2*x + gamma)),
	    (p['p']*(p['p']+1)/2)**(1/(p['p']-1))*(sech(-(1-p['p'])/2*x + gamma))**(2/(p['p']-1)),
	    (p['p']*(p['p']+1)/2)**(1/(p['p']-1))*(sech(-(1-p['p'])/2*x + gamma))**(2/(p['p']-1))*(np.tanh(-(1-p['p'])/2*x + gamma))])
		return out
	
	def Flinear(y,p):

		 return np.array([ [0,  1], 
	        			   [1,  0]   ])
	
	def bc2(ya,yb,s,p):
		AM = Flinear((ya + yb)/2.,p)
		LM = linalg.orth(projection2(AM,-1,10**(-6))[0])
	
		AP = Flinear(yb,p)
		LP = linalg.orth(projection2(AP,+1,10**(-6))[0])
		
		out = np.zeros((4,))#,dtype='complex128')
		out[0] = np.real(ya[s['rarray'][0:-1]]-ya[s['larray'][0:-1]])#matching conditions: only satisfying one
		out[1] = np.real(ya[ph[0]] - ph[1])							 #phase condition: 1
		# print LP.T.dot( yb[s['rarray']] )
		out[2] = np.real(LP.T.dot( yb[s['rarray']] )	)			 #two growth modes at +infty
		out[3] = np.real(LM.T.dot( yb[s['larray']] ))
		return out
		
		
	def bc2_jacobian(x,y,p):
		A = np.array([ [0,  1], 
	        		   [1,  0]   ])
		LM = linalg.orth(projection2(A,-1,10**(-6))[0])
		LP = linalg.orth(projection2(A,+1,10**(-6))[0])
		
		dGdya = np.array([  [1, 0, -1, 0],
							[0, 1, 0, 0],
							[0, 0, 0, 0],
							[0, 0, 0, 0] ])
		dGdyb = np.zeros((4,4))
		dGdyb[2,0:2] = np.real(LP.T)
		dGdyb[3,2:] = np.real(LM.T)
		print dGdya
		print dGdyb
		return dGdya,dGdyb
	
	pre_double_F= lambda x,y: double_F(x,y,s,p)
	pre_bc=lambda ya,yb: bc2(ya,yb,s,p)
	ode_jacobian=lambda x,y: double_F_jacobian(x,y,p)
	# ode_jacobian=lambda x,y: double_F_jacobian(x,y) #Testing numdifftools
	bc_jacobian=lambda x,y: bc2_jacobian(x,y,p)
	pre_guess = lambda x: guess(x,p)
	
	options = struct()
	# options include abstol, reltol, singularterm, stats, vectorized, maxnewpts,slopeout,xint
	options.abstol, options.reltol = 1e-7, 1e-6
	options.fjacobian, options.bcjacobian = ode_jacobian, bc_jacobian
	options.nmax = 5000
	options.stats='on'

	solinit = bvpinit(np.linspace(0,s['I'],100),pre_guess)
	s['sol'] = bvp6c(pre_double_F,pre_bc,solinit,options)
	xint = np.linspace(0,20,1600); Sxint,_ = deval(s['sol'],xint)
	plt.plot(xint,Sxint[0],'-k',linewidth=2.0)
	plt.plot(-xint,Sxint[2],'-k',linewidth=2.0)
	plt.plot(xint,Sxint[1],'-b',linewidth=2.0)
	plt.plot(-xint,Sxint[3],'-b',linewidth=2.0)
	# x = np.linspace(s['L'],s['R'],1600)
	# p = p['p']
	# y_xint = (.5*p*(p+1))**(1./(p-1))*(sech(.5*(1-p)*x))**(2/(p-1))
	# plt.plot(x,y_xint,'-r',linewidth=1.2)
	plt.show()
	plt.clf()
	
	
	return



if __name__ == "__main__":
	# pass
	gKdV()
	# gKdV_profile()