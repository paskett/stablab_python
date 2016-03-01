from bvp6c import bvp6c
from deval import deval
from bvpinit import bvpinit
from structure_variable import *
import matplotlib.pyplot as plt


class bvp:
	def __init__(self,ode,interval,bcs,init_guess):
		self.ode = ode
		self.interval = interval
		self.bcs = bcs
		self.init_guess = init_guess
	
	def set(self):
		pass
	
	def solve(self):
		pass
	
	def eval(self):
		pass
	


class Bvp6c(bvp):
	def set(self,points=30,options=[]):
		self.solinit = bvpinit(np.linspace(self.interval[0],self.interval[1],points),
								self.init_guess)
		self.options = options
	
	def solve(self):
		# return bvp6c(*args,**kargs)
		self.sol = bvp6c(self.ode,self.bcs,self.solinit,self.options)
		return self.sol
		
	def eval(self,pts):
		return deval(self.sol,pts)
	

if __name__ =="__main__":
	
	# print bvp.__name__
	epsilon = .01
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
	
	interval = [0,1]
	options = struct()
	# options include abstol, reltol, singularterm, stats, vectorized, maxnewpts,slopeout,xint
	options.abstol, options.reltol = 1e-8, 1e-7
	options.fjacobian, options.bcjacobian = f_jacobian, bc_jacobian
	options.nmax = 20000
	
	prob1 = Bvp6c(ode,interval,bcs,init_guess=init)
	prob1.set(10,options)
	prob1.solve()
	xint = np.linspace(0,1,100)
	Sxint,_ = prob1.eval(xint)
	
	plt.plot(xint,Sxint[0],'-k',linewidth=2.0)
	plt.axis([-.02,1.02,-.02,1.02])
	plt.show()
	
	
	
	