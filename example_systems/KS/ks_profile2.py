def  ks_profile2(s,p):
	#
	# function [sol,delta]=ks_profile(s,p)
	#
	#
	# bvp solver for KS periodic profile equation. Here p is a structure
	# containing the parameters X=period (guess), q=integration constant, 
	# beta = parameter, epsilon = parameter, delta=paramter.
	# The peridoc X is treated as a free variable. 
	# The structure s either contains a matlab ode/bvp solution structure,
	# s.sol which provides an initial guess for the bvp solver, or if it does
	# not, then an initial guess is provided which corresponds to q=6,
	# epsilon=0.2, X=6.3.
	
	# change defalut tolerance settings if not specified.
	if not ('options' in s):
		options=bvpset('AbsTol',10**(-6), 'RelTol',10**(-8));
	
	# check if an initial guess is provided.
	if not ('sol' in s):
		ld=load('profile_starter');
		s.sol=ld.s.sol;
	
	# define an anomynuous function to allow the passing of the parameter p
	# without using a global variable.
	ode = lambda x,y,q: (per_ode(x,y,q,p));
	
	# define anonymous functions to avoid global variables
	guess=lambda x: (interpolate(x,s.sol));
	
	# call bvp solver
	solinit = bvpinit(np.linspace(0,1,30),guess,p.X);
	sol = bvp5c(ode,bc,solinit,options);
	X = sol.parameters;
	
	######################################################
	return [sol,X] 

def bc(ya,yb,X):
	# boundary conditions
	
	out=[
	[ya[0]-yb[0]], # periodice boundary condition
	[ya[1]-yb[1]], # periodice boundary condition
	[ya[2]-yb[2]], # periodice boundary condition
	[ya[1]] # phase condition
	]
	 
	######################################################
	return out

def per_ode(x,y,X,p):
	# ks ode
	
	u=y[0];
	u_x=y[1];
	u_xx=y[2];
	
	out=[
	[u_x],
	[u_xx],
	[-p.epsilon*X*u_xx-p.delta*X**2*u_x-X**3*f(u)+X**3*p.q]
	]
	  
	######################################################
	return out

def  interpolate(x,sol):
	# interpolate intial guess
	
	Xtemp = sol.x[end];
	out=np.real;
	out[1] = out[1]*Xtemp;
	out[2] = out[2]*Xtemp**2;
	
	# sol is the solution in terms of u(x), whereas the needed guess is for
	# \bar u(\bar x) where \bar x = x/X and X is the period. Hence, the
	# derivatives of the sol solution needs to be multiplied by powers of X.
	
	
	
	
	
	######################################################
	return out 
