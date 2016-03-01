from __future__ import print_function, division
import pdb
from bvpget import bvpget     # Must have
from scipy.linalg import norm, lu, solve
from scipy.sparse.linalg import spsolve,splu
from scipy.sparse import spdiags, lil_matrix, csc_matrix
# from scipy.io import loadmat, savemat
# import matplotlib.pyplot as plt
from structure_variable import *

from bvpinit import bvpinit
from deval import deval

def bvp6c(ode, bc, solinit, options=[], *varargin):
	# BVP6C	 Solve boundary value problems for ODEs by collocation.
	#		 (6th order extension of BVP4C by Kierzenka and Shampine)
	#	 SOL = BVP6C(ODEFUN,BCFUN,SOLINIT) integrates a system of ordinary
	#	 differential equations of the form y' = f(x,y) on the interval [a,b],
	#	 subject to general two-point boundary conditions of the form
	#	 bc(y(a),y(b)) = 0. ODEFUN is a function of two arguments: a scalar X
	#	 and a vector Y. ODEFUN(X,Y) must return a column vector representing
	#	 f(x,y). BCFUN is a function of two vector arguments. BCFUN(YA,YB) must
	#	 return a column vector representing bc(y(a),y(b)). SOLINIT is a structure
	#	 with fields named	 
	#		 x -- ordered nodes of the initial mesh with 
	#			  SOLINIT.x(1) = a, SOLINIT.x(end) = b
	#		 y -- initial guess for the solution with SOLINIT.y(:,i)
	#			  a guess for y(x(i)), the solution at the node SOLINIT.x(i)	   
	# 
	#	 BVP6C produces a solution that is continuous on [a,b] and has a
	#	 continuous first derivative there. The solution is evaluated at points
	#	 XINT using the output SOL of BVP6C and the function DEVAL:
	#	 YINT = DEVAL(SOL,XINT). The output SOL is a structure with 
	#		 SOL.x	-- mesh selected by BVP6C
	#		 SOL.y	-- approximation to y(x) at the mesh points of SOL.x
	#		 SOL.solver -- 'bvp6c'
	#	 If specified in BVPSET, SOL may also contain
	#		 SOL.yp -- approximation to y'(x) at the mesh points of SOL.x
	#		 sol.ypmid approximations to y'(x) at interior points of SOL.x
	#		 
	#	 SOL = BVP6C(ODEFUN,BCFUN,SOLINIT,OPTIONS) solves as above with default
	#	 parameters replaced by values in OPTIONS, a structure created with the
	#	 BVPSET function. To reduce the run time greatly, use OPTIONS to supply 
	#	 a function for evaluating the Jacobian and/or vectorize ODEFUN. 
	#	 See BVPSET for details and SHOCKBVP for an example that does both.
	# 
	#	 SOL = BVP6C(ODEFUN,BCFUN,SOLINIT,OPTIONS,P1,P2...) passes constant, known
	#	 parameters P1, P2... to the functions ODEFUN and BCFUN, and to all 
	#	 functions specified in OPTIONS. Use OPTIONS = [] as a place holder if
	#	 no options are set.   
	#	 
	#	 Some boundary value problems involve a vector of unknown parameters p
	#	 that must be computed along with y(x):
	#		 y' = f(x,y,p)
	#		 0	= bc(y(a),y(b),p) 
	#	 For such problems the field SOLINIT.parameters is used to provide a guess
	#	 for the unknown parameters. On output the parameters found are returned
	#	 in the field SOL.parameters. The solution SOL of a problem with one set 
	#	 of parameter values can be used as SOLINIT for another set. Difficult BVPs 
	#	 may be solved by continuation: start with parameter values for which you can 
	#	 get a solution, and use it as a guess for the solution of a problem with 
	#	 parameters closer to the ones you want. Repeat until you solve the BVP 
	#	 for the parameters you want.
	# 
	#	 The function BVPINIT forms the guess structure in the most common 
	#	 situations:  SOLINIT = BVPINIT(X,YINIT) forms the guess for an initial mesh X
	#	 as described for SOLINIT.x and YINIT either a constant vector guess for the
	#	 solution or a function that evaluates the guess for the solution
	#	 at any point in [a,b]. If there are unknown parameters, 
	#	 SOLINIT = BVPINIT(X,YINIT,PARAMS) forms the guess with the vector PARAMS of 
	#	 guesses for the unknown parameters.  
	# 
	#	 BVP6C solves a class of singular BVPs, including problems with 
	#	 unknown parameters p, of the form
	#		 y' = S*y/x + f(x,y,p)
	#		 0	= bc(y(0),y(b),p) 
	#	 The interval is required to be [0, b] with b > 0. 
	#	 Often such problems arise when computing a smooth solution of 
	#	 ODEs that result from PDEs because of cylindrical or spherical 
	#	 symmetry. For singular problems the (constant) matrix S is
	#	 specified as the value of the 'SingularTerm' option of BVPSET,
	#	 and ODEFUN evaluates only f(x,y,p). The boundary conditions
	#	 must be consistent with the necessary condition S*y(0) = 0 and
	#	 the initial guess should satisfy this condition.	
	# 
	#	 BVP6C can solve multipoint boundary value problems.  For such problems
	#	 there are boundary conditions at points in [a,b]. Generally these points
	#	 represent interfaces and provide a natural division of [a,b] into regions.
	#	 BVP6C enumerates the regions from left to right (from a to b), with indices 
	#	 starting from 1.  In region k, BVP6C evaluates the derivative as 
	#	 YP = ODEFUN(X,Y,K).  In the boundary conditions function, 
	#	 BCFUN(YLEFT,YRIGHT), YLEFT(:,k) is the solution at the 'left' boundary
	#	 of region k and similarly for YRIGHT(:,k).	 When an initial guess is
	#	 created with BVPINIT(XINIT,YINIT), XINIT must have double entries for 
	#	 each interface point. If YINIT is a function, BVPINIT calls Y = YINIT(X,K) 
	#	 to get an initial guess for the solution at X in region k. In the solution
	#	 structure SOL returned by BVP6C, SOL.x has double entries for each interface 
	#	 point. The corresponding columns of SOL.y contain the 'left' and 'right' 
	#	 solution at the interface, respectively. See THREEBVP for an example of
	#	 solving a three-point BVP.
	# 
#	 Example
#		   solinit = bvpinit([0 1 2 3 4],[1 0]);
#		   sol = bvp6c(@twoode,@twobc,solinit);
#	   solve a BVP on the interval [0,4] with differential equations and 
#	   boundary conditions computed by functions twoode and twobc, respectively.
#	   This example uses [0 1 2 3 4] as an initial mesh, and [1 0] as an initial 
#	   approximation of the solution components at the mesh points.
#		   xint = linspace(0,4);
#		   yint = deval(sol,xint);
#	   evaluate the solution at 100 equally spaced points in [0 4]. The first
#	   component of the solution is then plotted with 
#		   plot(xint,yint(1,:));
#	 For more examples see TWOBVP, FSBVP, SHOCKBVP, MAT4BVP, EMDENBVP, THREEBVP.
# 
#	 See also BVPSET, BVPGET, BVPINIT, DEVAL, @.
#  
#	 BVP6C is a finite difference code that implements the Cash-Singhal 6
#	 formula. This is a collocation formula and the collocation 
#	 polynomial provides a C1-continuous solution that is sixth order
#	 accurate uniformly in [a,b]. (For multipoint BVPs, the solution is 
#	 C1-continuous within each region, but continuity is not automatically
#	 imposed at the interfaces.) Mesh selection and error control are based
#	 on the residual of the continuous solution. Analytical condensation is
#	 used when the system of algebraic equations is formed.
# 
#	 BVP4C
#	  Jacek Kierzenka and Lawrence F. Shampine
#	  Copyright 1984-2003 The MathWorks, Inc. 
#	  $Revision: 1.21.4.8 $	 $Date: 2003/12/26 18:08:47 $
#	 BVP6C Modification
#	  Nick Hale	 Imperial College London
#	  $Date: 12/06/2006 $
	
	
	# print(3*'\n')
	# print(" -------------------------------------------")
	# print(" --------- Startup Parameters ----------")
	# print(" -------------------------------------------")
	# print(3*'\n')
	sol = struct()
	# check input parameters
	nargin = len(list( locals() )) # Must be placed before defining new variables
	MaxNewPts = bvpget(options,'MaxNewPts',2) 
	solnpts = bvpget(options,'Xint',[]) 
	if len(solnpts)==0: 
		slopeout = ( bvpget(options,'SlopeOut','on')=='on') 
	else:
		slopeout = 0 
		if bvpget(options,'SlopeOut') == 'on':
			print('MATLAB:bvp6c:SolnPtsNoSlopeOut' + '\n',
				  'SOL will not contain slope data \n',
				  '(solution points were specified in BVPSET)')
			return
	
	x = np.copy( (solinit.x).T )	   # row vector
	y = np.copy( solinit.y ) 
	# if (len(x)==0) or (len(x)<2):
	#	print 'MATLAB:bvp6c:SolinitXNotEnoughPts', '\ must contain at least the two end points.'
	#	raise SystemError
	# else:
	#	N = len(x)		# number of mesh points
	N = len(x)
	xdir = np.sign(x[-1]-x[0])
	# if np.any(xdir * np.diff(x) < 0):
	#	print 'MATLAB:bvp6c:SolinitXNotMonotonic\n','The entries in X must increase or decrease.'
	#	raise SystemError
	
	# a multi-point BVP?
	mbcidx = np.where(np.diff(x) == 0)	# locate internal interfaces
	if len(mbcidx[0])>0: ismbvp = True
	else: ismbvp = False
	
	# if y.shape ==():
	#	print 'MATLAB:bvp6c:SolinitYEmpty\n', 'No initial guess provided in Y'
	#	raise SystemError
	#
	# if len(y[0,:]) != N:
	#	print 'MATLAB:bvp6c:SolXSolYSizeMismatch\n',' Y not consistent with X.'
	#	raise SystemError
	
	n = y.shape[0] 
	nN = n*N
	
	# stats
	nODEeval = 0
	nBCeval = 0
	# # set the options		#This code is redundant
	# if nargin<4:
	#	options = []
	# 
	# parameters
	knownPar = varargin 
	unknownPar = hasattr(solinit,'parameters') 
	
	# We have no unknown parameters
	npar = 0
	ExtraArgs = knownPar
	nExtraArgs = len(ExtraArgs)
	
	# Check the argument functions
	# Combustion problem is not a multi-point bvp
	nBCs = n + npar
	testBC = bc(y[:,0],y[:,-1])
	
	nBCeval = nBCeval + 1
	if len(testBC) != nBCs:
		print('Python:bvp6c:BCfunOuputSize\n'+ 'The boundary condition function should return a column vector of'+ 
			' length %i'%nBCs)
		raise SystemError
	
	if ismbvp: pass
	else: testODE = ode(x[0],y[:,0]) 
	
	nODEeval += 1
	if len(testODE) != n:
		assert ValueError,('Python:bvp6c:ODEfunOutputSize\n'+
		'The derivative function should return a column vector of' +
			' length %i' %n)
	
	# Get options and set the defaults
	rtol = bvpget(options,'RelTol',1e-3)
	if (type(rtol)==list and len(rtol) != 1) or (rtol<=0):
		assert ValueError,('Python:bvp6c:RelTolNotPos\nRelTol must be a positive scalar.')
	
	if rtol < 100*eps:
		rtol = 100*eps
		print('Python:bvp6c:RelTolIncrease\nRelTol has been increased to %f.'%rtol)
	
	atol = bvpget(options,'AbsTol',1e-6)
	if isnumeric(atol): atol = atol*np.ones((n,1))
	else:
		if len(atol) != n:
			assert ValueError,('Python:bvp6c:SizeAbsTol\nSolving %s requires a scalar AbsTol, or a vector AbsTol of length %d.'%( 'ode',n))
		atol = atol[:]
		
	if np.any(atol<=0):
		assert ValueError,('Python:bvp6c:AbsTolNotPos\nAbsTol must be positive.')
	
	threshold = atol/rtol
	
	# analytical Jacobians
	Fjac = bvpget(options,'FJacobian')
	BCjac = bvpget(options,'BCJacobian')  
	
	averageJac = True if (Fjac ==[]) else False 
	
	Nmax = bvpget(options,'Nmax',np.floor(2500./n))
	printstats = (bvpget(options,'Stats','off')=='on')
	
	# vectorized (with respect to x and y)
	xyVectorized = (bvpget(options,'Vectorized','off')=='on')
	if xyVectorized: vectVars = [1,2] # input to odenumjac 
	else: vectVars = []
	
	# Deal with a singular BVP.
	singularBVP = False
	S = bvpget(options,'SingularTerm',[])
	# if not (S==[]): # len(S)>0: # ~isempty(S):
	#	if (x[0] != 0) or (x[-1] <= x[0]):
	#		print 'MATLAB:bvp6c:SingBVPInvalidInterval\nSingular BVP must be on interval [0, b] with 0 < b.'
	#		raise SystemError
	#
	#	singularBVP = true
	#	# Compute matrix for imposing necessary BC, Sy(0) = 0,
	#	# and impose on guess for solution.
	#	PBC = eye(size(S)) - pinv(S)*S
	#	y[:,0] = PBC*y[:,0]
	#	# Compute matrix for proper definition of y'(0).
	#	PImS = pinv(eye(size(S)) - S)
	#	# Augment ExtraArgs with information about singular BVP.
	#	ExtraArgs += [ode, Fjac, S, PImS] # ExtraArgs = [ ExtraArgs {ode Fjac S PImS}]
	#	ode = Sode		 # Function Assignment
	#	if ~isempty(Fjac):
	#		if npar > 0:	 #Function Assignment
	#			Fjac = SFjacpar
	#		else:
	#			Fjac = SFjac
	
	
	maxNewtIter = 4
	maxProbes = 4	 # weak line search 
	needGlobJac = True
	
	done = False
	# print(3*'\n')
	# print(" -------------------------------------------")
	# print(" --------- Starting the Main Loop ----------")
	# print(" -------------------------------------------")
	# print(3*'\n')
	# THE MAIN LOOP:
	iterate_this = 0
	while not done:
		iterate_this +=1
		Y =	 y[:]
		
		RHS,Xmid,Ymid,yp,Fmid,NF = colloc_RHS(n,x,Y,ode,bc,npar,xyVectorized, mbcidx,nExtraArgs,ExtraArgs)	
		nODEeval = nODEeval + NF
		nBCeval = nBCeval + 1
		
		for iterin in xrange(maxNewtIter):
			
			if needGlobJac:
				# setup and factor the global Jacobian	
				
				dPHIdy,NF,NBC = colloc_Jac(n,x,Xmid,Y,Ymid,yp,Fmid,ode,bc,Fjac,BCjac,npar,
											vectVars,averageJac,mbcidx,nExtraArgs,ExtraArgs); 
											
				needGlobJac = False						
				nODEeval = nODEeval + NF			   
				nBCeval = nBCeval + NBC		
				# explicit row scaling	
				# plt.spy(dPHIdy)
				# plt.show()
				# plt.clf()
				dense_dPHIdy = dPHIdy.todense()
				wt = np.amax(np.abs(dense_dPHIdy),axis=1)
				
				singJac = False
				if any(wt == 0) or not np.all(np.isfinite(dense_dPHIdy[np.nonzero(dense_dPHIdy)])):
					singJac = True
					print("singJac World")
				else:
					# Unoptimized
					# scalMatrix = spdiags(1./wt.T,np.array([0]),nN+npar,nN+npar).todense()
					# dPHIdy = scalMatrix.dot(dense_dPHIdy)
					# P, L,U = lu(dPHIdy)
					# P = P.T
					
					# To use with a sparse solver
					# scalMatrix = spdiags(1./wt.T,np.array([0]),nN+npar,nN+npar)
					# dPHIdy = scalMatrix.dot(dPHIdy)
					# P, L,U = lu(csc_matrix(dPHIdy))
					# P = P.T
					
					# Using a sparse solver + sparse LU decomposition; most efficient
					scalMatrix = spdiags(1./wt.T,np.array([0]),nN+npar,nN+npar)
					dPHIdy = scalMatrix.dot(dPHIdy)
					invA = splu(csc_matrix(dPHIdy))
					
					# singJac = check_singular(dPHIdy,L,U,P,warnstat,warnoff)
				
				if singJac:
					msg = 'Unable to solve the collocation equations -- a singular Jacobian encountered';
					print('Python:bvp6c:SingJac', msg)
					raise SystemError
					
				# scalMatrix = P.dot(scalMatrix)
						
			# find the Newton direction	   
			# Unoptimized
			# delY = solve(U,solve( L,scalMatrix.dot(RHS) ))
			# distY = norm(delY)
			
			# Using a sparse solver
			# U = csr_matrix(U)
			# L = csr_matrix(L)
			# delY = spsolve(U,spsolve( L,scalMatrix.dot(RHS) ))
			# distY = norm(delY)	
			
			# Using a sparse solver + sparse LU decomposition; most efficient
			delY = invA.solve(scalMatrix.dot(RHS) )
			distY = norm(delY)
			
			# weak line search with an affine stopping criterion
			lmbda = 1
			
			
			Y= np.copy(np.reshape(Y,(n*N,),order='F') )
			# print(Y.shape)
			# print(delY.shape)
			for probe in range(maxProbes):		
				Ynew = Y - lmbda*np.squeeze(delY)	
				# print(Ynew.shape)
				Ynew = Ynew.reshape((n,N),order='F' )
				RHS,Xmid,Ymid,yp,Fmid,NF = colloc_RHS(n,x,Ynew,ode,bc,npar,xyVectorized,mbcidx,nExtraArgs,ExtraArgs)
				nODEeval = nODEeval + NF
				nBCeval = nBCeval + 1
				
				# distYnew = norm(solve(U,solve( L,scalMatrix.dot(RHS) )) )
				# distYnew = norm(spsolve(U,spsolve( L,scalMatrix.dot(RHS) )) )
				distYnew = norm(invA.solve(scalMatrix.dot(RHS) ) )
				
				if (distYnew < 0.9*distY): break		
				else: lmbda = 0.5*lmbda		
			
			needGlobJac = (distYnew > 0.1*distY)
			
			if distYnew < 0.1*rtol: break
			else: Y = Ynew
			
		y = np.reshape(Ynew[0:nN],(n,N),order='F') #reshape(Ynew(1:nN),n,N); yp, ExtraArgs, and RHS are consistent with y
		
		res,NF,Yip05,Fmid[:,:,1] = residual(ode,x,y,yp,Fmid,RHS,threshold,xyVectorized,nBCs,
											mbcidx,ExtraArgs)
		
		nODEeval = nODEeval + NF
		# print("Length of x is ",len(x))
		# if len(x)==30:
		# 	savemat('prob27_30.mat', {'x':x,'y':y,'res':res})
		# 	raise SystemError
	
		if max(res) < rtol:
			done = True
		else:	# redistribute the mesh 
			
			N,x,y,mbcidx = new_profile(n,x,y,Yip05,yp,Fmid,res,mbcidx,rtol,Nmax,MaxNewPts);
			
			if N > Nmax:   
				warning('MATLAB:bvp6c:RelTolNotMet', 
			  [ 'Unable to meet the tolerance without using more than %d '
				'mesh points. \n The last mesh of %d points and ' 
				'the solution are available in the output argument. \n ', 
				'The maximum residual is %g, while requested accuracy is %g.'], 
				Nmax,length(x),max(res),rtol)
				sol.solver = 'bvp6c' 
				sol.maxres = max(res)
				if ~isempty(solnpts):
					sol.note = 'max residual before interpolation'
					sol.x=solnpts
					sol.y=ntrp6c(ode,sol.x,x,y,yp,Fmid)
				else:
					sol.x = x
					sol.y = y
					if slopeout:
						sol.yp = yp
						sol.ypmid = Fmid
				sol.fevals = nODEeval
				sol.meshexceeded= 1
		  
				if printstats:
					fprintf('The solution was obtained on a mesh of %g points.\n',N)
					fprintf('The maximum residual is %10.3e. \n',sol.maxres)
					fprintf('There were %g calls to the ODE function. \n',nODEeval)
					fprintf('There were %g calls to the BC function. \n',nBCeval)
				return sol
		
			nN = n*N
			needGlobJac = True
	# end of while loop
	
	# Output
	sol.solver = 'bvp6c'
	if not solnpts== []:
		sol.note = 'max residual before interpolation'
		sol.x=solnpts; sol.y=ntrp6c(ode,sol.x,x,y,yp,Fmid)
	else:
		sol.x = x; sol.y = y
		if slopeout: 
			sol.yp = yp; sol.ypmid = Fmid
	
	
	sol.meshexceeded = 0
	sol.fevals = nODEeval
	
	# Stats
	if printstats:
		print('\nData for ode '+ode.func_name)
		print('The solution was obtained on a mesh of %g points.'%N)
		print('The maximum residual is %10.3e.'%max(res))
		print('There were %g calls to the ODE function.'%nODEeval)
		print('There were %g calls to the BC function.'%nBCeval)
	
	sol.stat = struct()
	sol.stat.mesh= len(sol.x)
	sol.stat.ode=nODEeval
	sol.stat.bc=nBCeval
	sol.stat.max_error=max(res)
	
	return sol

#--------------------------------------------------------------------------
#	
def condaux(flag,X,L,U,P):
	#CONDAUX  Auxiliary function for estimation of condition.
	
	if flag=='dim': 
		f = max(L.shape)
	elif flag=='real':
		f = 1
	elif flag=='notransp':
		f = solve(U, solve( L,(P.dot(X)) )	 )
	elif flag =='transp':
		f = P.dot(solve(np.conj(L.T), solve(np.conj(U.T), X) ))
	#This function is probably finished. Need to test.
	
	# switch flag
	# case 'dim'
	#	  f = max(size(L));
	# case 'real'
	#	  f = 1;
	# case 'notransp'
	#	  f = U \ (L \ (P * X));
	# case 'transp'
	#	  f = P * (L' \ (U' \ X));
	# end
	return f	


#--------------------------------------------------------------------------
#	
def check_singular(A,L,U,P,warnstat,warnoff):
	#CHECK_SINGULAR	 Check A (L*U*P) for 'singularity'; mute certain warnings.
	
	Ainv_norm = normest1(condaux,[],[],L,U,P)
	singular = (Ainv_norm*norm(A,inf)*eps > 1e-5)
	return singular


#---------------------------------------------------------------------------

def interp_Hermite(w,h,y,yp,yp_ip025,yp_ip05,yp_ip075,both=True):
	#INTERP_HERMITE	 use the 6th order Hermite Interpolant presented by Cash
		#and Wright to find y and y' at abscissas for Quadrature.
	# As written, this requires most of its arguments to be 2d numpy arrays, and it returns 2d numpy arrays: (n,1)
	def A66(w):
		return w**2.*np.polyval(np.array([-24, 60, -50, 15.]),w)	  # w^2*(15-50*w+60*w^2-24*w^3);

	def B66(w):
		return w**2.*np.polyval(np.array([12, -26, 19, -5.])/3.,w)	   # w^2/3*(w-1)*(12*w^2-14*w+5);

	def C66(w):
		return w**2.*np.polyval(np.array([-8, 16, -8.])/3.,w)		   # -w^2*8/3*(1-w)^2;

	def D66(w):
		return w**2.*np.polyval(np.array([16, -40, 32, -8.]),w)	  # w^2*8*(1-w)^2*(2*w-1);


	def Ap66(w):
		return w*np.polyval(np.array([-120, 240, -150, 30.]),w)	 #w*(30-150*w+240*w^2-120*w^3);

	def Bp66(w):
		return w*np.polyval(np.array([20, -104/3., 19, -10/3.]),w)	  #w*(w*(20*w^2+19)-(104*w^2+10)/3);

	def Cp66(w):
		return -16/3.*w*np.polyval(np.array([2, -3, 1.]),w)		  #-16/3*w*(1-3*w+2*w^2);

	def Dp66(w):
		return w*np.polyval(np.array([80, -160, 96, -16.]),w)		 #w*(80*w^3-160*w^2+96*w-16);


	N=y.shape[1]
	diagscal = spdiags(h.T,0,N-1,N-1)

	Sx =	  A66(w)*y[:,1:] + A66(1-w)*y[:,:-1]	  + \
		( B66(w)*yp[:,1:] - B66(1-w)*yp[:,:-1] + \
		  C66(w)*(yp_ip075-yp_ip025) + D66(w)*yp_ip05 )*diagscal

	if both:
		diagscal = spdiags(1./h.T,0,N-1,N-1)
		Spx=( Ap66(w)*y[:,1:]	- Ap66(1-w)*y[:,:-1] )*diagscal + \
			( Bp66(w)*yp[:,1:] + Bp66(1-w)*yp[:,:-1] + \
			  Cp66(w)*(yp_ip075-yp_ip025) + Dp66(w)*yp_ip05 )
	else:
		Spx = []
	return Sx,Spx

	
#---------------------------------------------------------------------------
# @print_this
def residual(Fcn, x, y, yp, Fmid, RHS, threshold, xyVectorized, nBCs, mbcidx, ExtraArgs):
	#RESIDUAL  Compute L2-norm of the residual using 7-point Lobatto quadrature.
	# multi-point BVP support
	if len(mbcidx[0])>0: 
		ismbvp = True
	else: 
		ismbvp = False # This case for the combustion problem
	nregions = len(mbcidx[0]) + 1
	# Lidx = [1, mbcidx+1]
	# Ridx = [mbcidx, len(x)]
	Lidx = [1]			 # since mbcidx = [] for the combustion system
	Ridx = [len(x)]
	# if ismbvp:
	#	FcnArgs = {0,ExtraArgs{:}}	#pass region idx
	# else:
	#	FcnArgs = ExtraArgs
	lob, lobw = 7*[0], 7*[0]
	lob[2]= 0.0848880518607165
	lobw[2]=0.276826047361566
	lob[3]= 0.265575603264643
	lobw[3]=0.431745381209863

	lob[5]= 0.734424396735357
	lobw[5]=lobw[3]
	lob[6]= 0.9151119481392835
	lobw[6]=lobw[2]

	n,N = y.shape

	Yp05=np.zeros((n,N-1))	
	Y05=np.zeros((n,N-1)) 

	res = np.zeros(N-1)
	nfcn = 0
	NewtRes = np.zeros((n,N-1))
	# Do not populate the interface intervals for multi-point BVPs.
	# intidx = setdiff[1:N-1,mbcidx]
	intidx = np.arange(N-1)
	NewtRes[:,intidx] = np.reshape(RHS[nBCs:],(n,-1),order='F')
	# print "\n Starting for loop \n"
	for region in xrange(nregions):
		if ismbvp: pass # FcnArgs{1} = region	   # Pass the region index to the ODE function.

		xidx = np.arange(Lidx[region]-1,Ridx[region]) # mesh point index
		Nreg = len(xidx)
		xreg = x[xidx]
		yreg = y[:,xidx]
		ypreg = yp[:,xidx]
		hreg = np.diff(xreg)
		iidx = xidx[:-1]	  # mesh interval index
		Nint = len(iidx)
		thresh  = threshold + np.zeros((n,Nint))
		yp_ip025=Fmid[:,iidx,0]
		yp_ip075=Fmid[:,iidx,2]

		diagscal = spdiags(hreg.T,0,Nint,Nint)
		xip05 = 0.5*(xreg[iidx] + xreg[iidx+1])
		#more accurate estimate of y_ip05 than used in Cash-Singhal
		Yip05 = (0.5*(yreg[:,iidx+1]+ yreg[:,iidx]) - 
						(ypreg[:,iidx+1]-yp[:,iidx]+4*(yp_ip075-yp_ip025))*diagscal/24	)
						
		if xyVectorized:
			Yp_ip05 = Fcn(xip05,Yip05)
			nfcn = nfcn + 1
		else: # not vectorized
			Yp_ip05 = np.zeros((n,Nint))
			for i in xrange(Nint):
				Yp_ip05[:,i] = Fcn(xip05[i],Yip05[:,i])
			nfcn = nfcn + Nint
		Y05[:,iidx]=Yip05
		Yp05[:,iidx]=Yp_ip05

		res_reg=np.zeros((1,Nint))

		#Sum contributions from other points
		for j in [2,3,5,6]:
			xLob = xreg[:Nreg-1] + lob[j]*hreg
			yLob, ypLob = interp_Hermite(lob[j],hreg,yreg,ypreg,yp_ip025,Yp_ip05,yp_ip075)
			if xyVectorized:
				fLob = Fcn(xLob,yLob)
				nfcn = nfcn + 1
			else:
				fLob = np.zeros((n,Nint))
				for i in xrange(Nint):
					fLob[:,i] = Fcn(xLob[i],yLob[:,i])
				nfcn = nfcn + Nint
			temp = (ypLob - fLob)/np.maximum(np.abs(fLob),thresh)
			res_reg = res_reg + lobw[j]*np.sum(temp.conj()*temp,axis=0)
		# scaling
		
		res_reg = np.sqrt( np.abs(hreg/2.)*res_reg)
		res[iidx] = res_reg
	return res,nfcn,Y05,Yp05


#--------------------------------------------------------------------------
# @print_this
def new_profile(n,x,y,y05,F,Fmid,res,mbcidx,rtol,Nmax,MaxNewPts):
	#NEW_PROFILE  Redistribute mesh points and approximate the solution.
	# multi-point BVP support
	nregions = len(mbcidx[0]) + 1
	Lidx = [1]
	Ridx = [len(x)]

	# mbcidxnew = []; xx = []; yy = []   #Screwed the code up
	NN = 0
	# if len(x)==189: pdb.set_trace()
	for region in xrange(nregions):
		xidx = np.arange(Lidx[region]-1,Ridx[region])
		xreg = x[xidx]
		yreg = y[:,xidx]
		Freg = F[:,xidx]
		hreg = np.diff(xreg)
		Nreg = len(xidx)

		iidx = xidx[:-1]	# mesh interval index
		resreg = res[iidx]

		F025reg = Fmid[:,iidx,0]
		F05reg	= Fmid[:,iidx,1]
		F075reg = Fmid[:,iidx,2]
		i1 = np.where(resreg > rtol)[0]
		i2 = np.where(resreg[i1] > 100*rtol)[0]
		NNmax = Nreg + len(i1) + len(i2)
		xxreg = np.zeros((NNmax))
		yyreg = np.zeros((n,NNmax))
		last_int = Nreg - 1-1

		xxreg[0] = xreg[0]
		yyreg[:,0] = yreg[:,0]
		NNreg = 1
		i = 0
		while i <= last_int:
			if resreg[i] > rtol:	 # introduce points
				# print("Introducing points")
				if resreg[i] > 100*rtol:
					Ni = MaxNewPts
					hi = hreg[i] / (Ni+1)
					j = np.arange(1,Ni+1) # 1:Ni
					xxreg[NNreg+j-1] = xxreg[NNreg-1] + j*hi
					for j in xrange(Ni):
						temp, _= interp_Hermite((j+1)/(Ni+1),hreg[i],yreg[:,i:i+2],
						 						Freg[:,i:i+2],F025reg[:,i,np.newaxis],F05reg[:,i,np.newaxis],F075reg[:,i,np.newaxis],both=False)
						yyreg[:,NNreg+j] = temp[:,0]
				else:
					# print("Entering else clause")
					Ni = 1
					xxreg[NNreg+1-1] = xxreg[NNreg-1] + hreg[i]/2
					yyreg[:,NNreg+1-1] = y05[:,i]
				NNreg = NNreg + Ni
			else:				 # try removing points
				# print("Removing points")
				if (i <= last_int-2) and (np.max(resreg[i+1:i+3]) < rtol):
					hnew = (hreg[i]+hreg[i+1]+hreg[i+2])/2.
					C1 = resreg[i]/(hreg[i]/hnew)**(11/2.)
					C2 = resreg[i+1]/(hreg[i+1]/hnew)**(11/2.)
					C3 = resreg[i+2]/(hreg[i+2]/hnew)**(11/2.)
					pred_res = max(np.array([C1,C2,C3]))

					if pred_res < 0.05 * rtol:	 # replace 3 intervals with 2
						xxreg[NNreg] = xxreg[NNreg-1] + hnew
						temp,_ = interp_Hermite(0.5,hreg[i],yreg[:,(i+1):(i+3)],
						 					Freg[:,(i+1):(i+3)],F025reg[:,i+1,np.newaxis],F05reg[:,i+1,np.newaxis],F075reg[:,i+1,np.newaxis],both=False)
						yyreg[:,NNreg] = temp[:,0]
						
						NNreg = NNreg + 1
						i = i + 2
			xxreg[NNreg] = xreg[i+1]   # preserve the next mesh point
			yyreg[:,NNreg] = yreg[:,i+1]
			NNreg = NNreg + 1
			i = i + 1
	#	end
		NN = NN + NNreg
		if (NN > Nmax):
			# return the previous solution
			xx = x
			yy = y
			mbcidxnew = mbcidx
			break
		else:
			xx = xxreg[0:NNreg]
			yy = yyreg[:,0:NNreg]
			mbcidxnew = mbcidx
			if region < nregions-1:	 # possible only for multipoint BVPs
				mbcidxnew = [mbcidxnew, NN]
	# end
		
		
		
		# if N==189:
		# 	savemat('pybvp6c/compare.mat',{'n':n,'x':x,'y':y,'Yip05':Yip05,'yp':yp,'Fmid':Fmid,'res':res,\
		# 	                        'mbcidx':mbcidx,'rtol':rtol,'Nmax':Nmax,'MaxNewPts':MaxNewPts})
		# 	raise SystemError
	return NN,xx,yy,mbcidxnew

	
#--------------------------------------------------------------------------
# @print_this
def colloc_RHS(n, x, Y, Fcn, Gbc, npar, xyVectorized, mbcidx, nExtraArgs, ExtraArgs):
	#COLLOC_RHS	 Evaluate the system of collocation equations Phi(Y).  
	#	The derivative approximated at the midpoints and returned in Fmid is
	#	used to estimate the residual. 
	# multi-point BVP support
	if len(mbcidx[0])>0: 
		ismbvp = True
	else: 
		ismbvp = False # This case for the combustion problem
	nregions = len(mbcidx[0]) + 1
	Lidx = [1]			 # since mbcidx = [] for the combustion system
	Ridx = [len(x)]
	
	if npar ==0:
		y = np.reshape(Y,(n,-1))
	else:
		y = np.reshape(Y[:,-1-npar],(n,-1)) 
	
	y = Y
	n, N = y.shape
	nBCs = n*nregions + npar
	
	F = np.zeros((n,N))
	Fmid = np.zeros((n,N-1,3))	 # include interface intervals
	Phi = np.zeros((nBCs+n*(N-nregions),1))	  # exclude interface intervals
	
	# Boundary conditions
	
	Phi[:nBCs,0] = Gbc(y[:,Lidx[0]-1],y[:,Ridx[0]-1])	
	phiptr = nBCs	 # active region of Phi
	for region in range(nregions):
		# mesh point index
		xidx = range(Lidx[0]-1,Ridx[0] )
		Nreg = len(xidx)
		xreg = x[xidx]
		yreg = y[:,xidx]
	  
		iidx = xidx[:-1]   # mesh interval index
		Nint = len(iidx)
		# derivative at the mesh points
		if xyVectorized:
			Freg = Fcn(xreg,yreg)
			nfcn = 1
		else:
			Freg = np.empty((n,Nreg)) # np.zeros vs. np.empty
			for i in range(Nreg):  
				Freg[:,i] = Fcn(xreg[i],yreg[:,i])
			nfcn = Nreg
		
		# mesh point data
		h = np.diff(xreg)
		H = np.diag(h[:],0)
		xi = xreg[:-1]
		yi = yreg[:,:-1]
		xip1 = xreg[1:]
		yip1 = yreg[:,1:]
		Fi = Freg[:,:-1]
		Fip1 = Freg[:,1:]
		
		#interior points & derivative
		xip025 = 0.25*(3*xi + xip1)
		xip075 = 0.25*(xi + 3*xip1)
		yip025 = (54*yi + 10*yip1 + (9*Fi - 3*Fip1).dot(H))/64.
		yip075 = (10*yi + 54*yip1 + (3*Fi - 9*Fip1).dot(H))/64.
		if xyVectorized:
			Fip025 = Fcn(xip025,yip025)
			Fip075 = Fcn(xip075,yip075)
			nfcn = nfcn + 3	   
		else: # not vectorized 
			Fip025 = np.empty((n,Nint))  #np.zeros vs np.empty
			Fip075 = np.empty((n,Nint))
			for i in xrange(Nint):
				Fip025[:,i] = Fcn(xip025[i],yip025[:,i])
				Fip075[:,i] = Fcn(xip075[i],yip075[:,i])
			nfcn = nfcn + 2*Nint
			
		# mid points & derivative  
		xip05 = 0.5*(xi + xip1)
		yip05 = 0.5*(yi + yip1) - (5*Fi - 16*Fip025 + 16*Fip075 - 5*Fip1).dot(H/24.)
		if xyVectorized:
			Fip05 = Fcn(xip05,yip05) 
			nfcn = nfcn + 1	   
		else: # not vectorized 
			Fip05 = np.empty((n,Nint))  #np.zeros vs np.empty
			for i in xrange(Nint):
				Fip05[:,i] = Fcn(xip05[i],yip05[:,i]) 
			nfcn = nfcn + Nint
		
	# the Cash-Singhal formula 
		Phireg = yip1 - yi - (7*Fi + 32*Fip025 + 12*Fip05 + 32*Fip075 + 7*Fip1).dot(H/90.)
		# output assembly 
		temp = Phireg.shape
		Phi[(phiptr):(phiptr+n*Nint)] = np.reshape(Phireg,(temp[0]*temp[1],1),order='F') 
		phiptr = phiptr + n*Nint
		Xmid = np.empty((1,Nint,3))
		Ymid = np.empty((n,Nint,3))
		Fmid = np.empty((n,Nint,3))
		
		Xmid[:,iidx,0] = xip025
		Xmid[:,iidx,1] = xip05
		Xmid[:,iidx,2] = xip075
	  
		Ymid[:,iidx,0] = yip025
		Ymid[:,iidx,1] = yip05
		Ymid[:,iidx,2] = yip075
	  
		F[:,xidx] = Freg
		Fmid[:,iidx,0] = Fip025
		Fmid[:,iidx,1] = Fip05
		Fmid[:,iidx,2] = Fip075
	return Phi,Xmid,Ymid,F,Fmid,nfcn

	
#---------------------------------------------------------------------------
# @print_this
def colloc_Jac(n, x, Xmid, Y, Ymid, F, Fmid, ode, bc, Fjac, BCjac, npar, vectVars, averageJac, mbcidx, nExtraArgs, ExtraArgs):
	# multi-point BVP support
	if len(mbcidx[0])>0: ismbvp = True
	else: ismbvp = False 
		
	nregions = len(mbcidx[0]) + 1
	Lidx = [1]
	Ridx = [len(x)]
	
	N = len(x)
	nN = n*N			  
	In = np.eye(n)
	nfcn = 0   
	nbc = 0
	y = np.copy(np.reshape(Y[:nN],(n,N)) )
	# BC points
	ya = np.copy(y[:,Lidx[0]-1])
	yb = np.copy(y[:,Ridx[0]-1])
	if Fjac ==[]:	# use numerical approx
		threshval = 1e-6
		Joptions = struct()
		Joptions.diffvar = 2  # dF(x,y)/dy
		Joptions.vectvars = vectVars
		Joptions.thresh = threshval*np.ones((n,1)) 
		if npar > 0:   # unknown parameters
			if ismbvp:
				dPoptions.diffvar = 4  # dF(x,y,region,p)/dp
			else:
				dPoptions.diffvar = 3  # dF(x,y,p)/dp
			dPoptions.vectvars = vectVars
			dPoptions.thresh = threshval*np.ones((npar,1)) # threshval(ones(npar,1))
			dPoptions.fac = []		  
		
		# Collocation equations
	nBCs = n*nregions + npar
	
	rows = np.arange(nBCs,nBCs+n)  # define the action area
	cols = np.arange(n)			   # in the global Jacobian
	if npar == 0:	# no unknown parameters -----------------------------
	  
		# dPHIdy = spalloc(nN,nN,2*n*nN) #	Do this later in Python
		dPHIdy = lil_matrix( (nN,nN) )	
		if BCjac==[]:	 # use numerical approx
			pass
			# dGdya,dGdyb,nbc = BCnumjac(bc,ya,yb,n,npar,nExtraArgs,ExtraArgs)
		# elif isa(BCjac,'cell'):	  # Constant partial derivatives {dGdya,dGdyb}
			# pass # Not used in combustion problem	   
		else:  # use analytical Jacobian
			dGdya,dGdyb = BCjac(ya,yb)
	  
		# Collocation equations 
		for region in range(nregions):
		
			# Left BC
			# print("\n")
			# print("region = ", region)
			# print("n = ", n)
			# print("nBCs = ", nBCs)
			# print("cols = ", cols)
			# print("x1.shape = ", dGdya[:,(region-1)*n+np.arange(n)].shape)
			# print("x2.shape = ", dPHIdy[:nBCs,cols].shape)
			dPHIdy[:nBCs,cols] = dGdya[:,(region-1)*n+np.arange(n)]
			left_bc = dPHIdy[:nBCs,cols]
			xidx = np.arange(Lidx[region]-1,Ridx[region])	 # mesh point index 
			xreg = x[xidx]	 
			yreg = y[:,xidx]
			Freg = F[:,xidx]
			hreg = np.diff(xreg)
			
			iidx = xidx[:-1]	 # mesh interval index
			Nint = len(iidx)
	
			[X1qtrreg, Xmidreg, X3qtrreg] = midptreg(iidx,Xmid)
			[Y1qtrreg, Ymidreg, Y3qtrreg] = midptreg(iidx,Ymid)
			[F1qtrreg, Fmidreg, F3qtrreg] = midptreg(iidx,Fmid)
			
			# Collocation equations
			if Fjac == []:	# use numerical approx
				pass
	#			Joptions.fac = []
	#			Ji,Joptions.fac,ignored,nFcalls = odenumjac(ode,[xreg[0],yreg[:,0]],Freg[:,0],Joptions)
	#
	#			nfcn = nfcn+nFcalls
	#			nrmJi = norm(Ji,1)
	#			for i in range(Nint):
	#				hi = hreg[i]
	#				# the right mesh point
	#				xip1 = xreg[i+1]
	#				yip1 = yreg[:,i+1]
	#				Fip1 = Freg[:,i+1]
	#
	#				Jip1,Joptions.fac,ignored,nFcalls = odenumjac(ode,[xip1,yip1],Fip1,Joptions)
	#				nfcn = nfcn + nFcalls
	#				nrmJip1 = norm(Jip1,1)
	#
	#				#the interior points
	#				if averageJac and ( norm(Jip1 - Ji,1) <= 0.125*(nrmJi + nrmJip1) ):
	#					Jip025 = 0.25*(3*Ji + Jip1)
	#					Jip05 = 0.5*(Ji + Jip1)
	#					Jip075 = 0.25*(Ji + 3*Jip1)
	#				else:
	#					xip025, xip05, xip075 = midpti(i,X1qtrreg, Xmidreg, X3qtrreg)
	#					yip025, yip05, yip075 = midpti(i,Y1qtrreg, Ymidreg, Y3qtrreg)
	#					Fip025, Fip05, Fip075 = midpti(i,F1qtrreg, Fmidreg, F3qtrreg)
	#
	#					# [Jip025,Joptions.fac,ignored,nFcalls025] =  odenumjac(ode,{xip025,yip025,FcnArgs{:}},Fip025,Joptions)
	#					# [Jip05,Joptions.fac,ignored,nFcalls05] =	odenumjac(ode,{xip05,yip05,FcnArgs{:}},Fip05,Joptions)
	#					# [Jip075,Joptions.fac,ignored,nFcalls075] = odenumjac(ode,{xip075,yip075,FcnArgs{:}},Fip075,Joptions)
	#					Jip025,Joptions.fac,ignored,nFcalls025 =  odenumjac(ode,[xip025,yip025],Fip025,Joptions)
	#					Jip05,Joptions.fac,ignored,nFcalls05 =	odenumjac(ode,[xip05,yip05],Fip05,Joptions)
	#					Jip075,Joptions.fac,ignored,nFcalls075 = odenumjac(ode,[xip075,yip075],Fip075,Joptions)
	#					nfcn = nfcn + nFcalls025 + nFcalls05 + nFcalls075
	# #				end
	#
	#				Jip05Jip025=Jip05.dot(Jip025)
	#				Jip05Jip075=Jip05.dot(Jip075)
	#				# assembly
	#				dPHIdy[rows[0]:rows[-1]+1,cols[0]:cols[-1]+1]=calc_dPHYdy1(hi,In,Ji,Jip025,Jip05,Jip075,Jip05Jip025,Jip05Jip075)
	#				cols = cols + n
	#				dPHIdy[rows[0]:rows[-1]+1,cols[0]:cols[-1]+1] = calc_dPHYdy2(hi,In,Jip1,Jip025,Jip05,Jip075,Jip05Jip025,Jip05Jip075)
	#				# if  i ==(Nint-1):
	#				rows = rows + n	  # next equation
	#
	#				Ji = Jip1
	#				nrmJi = nrmJip1
			#
			# elif isa(Fjac,'numeric'): # constant Jacobian
			#	pass
			#
			else: # use analytical Jacobian		   
				Ji = Fjac(xreg[0],yreg[:,0]);
	  
				for i in range(Nint):
					hi = hreg[i]
					# the right mesh point
					xip1 = xreg[i+1]
					yip1 = yreg[:,i+1]
					Jip1 = Fjac(xip1,yip1)
					
					# the interior points
					xip025, xip05, xip075 = midpti(i,X1qtrreg, Xmidreg, X3qtrreg)
					yip025, yip05, yip075 = midpti(i,Y1qtrreg, Ymidreg, Y3qtrreg)
					
					Jip025 = Fjac(xip025,yip025)
					Jip05 = Fjac(xip05,yip05)
					Jip075 = Fjac(xip075,yip075)
					
					Jip05Jip025=Jip05.dot(Jip025)
					Jip05Jip075=Jip05.dot(Jip075)
					# assembly
					hereamI = calc_dPHYdy1(hi,In,Ji,Jip025,Jip05,Jip075,Jip05Jip025,Jip05Jip075)
					dPHIdy[np.ix_(rows,cols)]=calc_dPHYdy1(hi,In,Ji,Jip025,Jip05,Jip075,Jip05Jip025,Jip05Jip075)
					
					cols = cols + n
					dPHIdy[np.ix_(rows,cols)]=calc_dPHYdy2(hi,In,Jip1,Jip025,Jip05,Jip075,Jip05Jip025,Jip05Jip075)
					
					rows = rows + n	  # next equation  

					Ji = Jip1
			
		# Right BC
			dPHIdy[:nBCs,cols] = dGdyb[:,(region-1)*n+np.arange(n)]
			right_BC = dPHIdy[:nBCs,cols]
			cols = cols + n
			#	end # regions
	
	else:  # there are unknown parameters --------------
		pass
#		# # Not touched by the combustion problem
	
	return dPHIdy,nfcn,nbc

		
#--------------------------------------------------------------------------
		
def calc_dPHYdy1(hi,In,Ji,Jip025,Jip05,Jip075,Jip05Jip025,Jip05Jip075):
	
	dPHIdy	 = - In -			  (
			  hi/90*	   ( 7*Ji+27*Jip025+6*Jip05+5*Jip075   ) + 
			  hi*hi/360*   ( 27*Jip05Jip025-5*Jip05Jip075)		 + 
			  ( hi*hi/360*(18*Jip025-10*Jip05+6*Jip075) +
				hi*hi*hi/240*(3*Jip05Jip025-Jip05Jip075) ).dot(Ji)
									 )
	
	return dPHIdy

#-------------------------------------------------------------------------

def calc_dPHYdy2(hi,In,Jip1,Jip025,Jip05,Jip075,Jip05Jip025,Jip05Jip075):
	dPHIdy	 = In -				  (
			 hi/90*		  ( 5*Jip025+6*Jip05+27*Jip075+7*Jip1  ) + 
			 hi*hi/360*	  ( 5*Jip05Jip025-27*Jip05Jip075 )		 - 
			 ( hi*hi/360*(6*Jip025-10*Jip05+18*Jip075) +		 
			   hi*hi*hi/240*(Jip05Jip025-3*Jip05Jip075)	 ).dot(Jip1)	
									 )
	
	return dPHIdy


#--------------------------------------------------------------------------

def bcaux(Ya,Yb,n,bcfun,*args):
	res = bcfun(Ya,Yb)
	return res

	
#---------------------------------------------------------------------------

def BCnumjac(bc,ya,yb,n,npar, nExtraArgs,ExtraArgs):
#BCNUMJAC	 Numerically compute dBC/dya, dBC/dyb, and dBC/dpar, if needed.
	bcArgs = [ya,yb,n,bc] 
	dBCoptions = struct()
	dBCoptions.thresh = 1.e-6*np.ones((n,1))  # repmat(1e-6,len(ya[:]),1)
	dBCoptions.fac = []
	dBCoptions.vectvars = [] # BC functions not vectorized
	
	bcVal  = bcaux(ya,yb,n,bc) 
	nCalls = 1
	
	dBCoptions.diffvar = 1
	
	# Ji,Joptions.fac,ignored,nFcalls = odenumjac(ode,[xreg[0],yreg[:,0]],Freg[:,0],Joptions)
	
	
	[dBCdya,ignored,ignored1,nbc] = odenumjac(bcaux,bcArgs,bcVal,dBCoptions)
	
	nCalls = nCalls + nbc
	dBCoptions.diffvar = 2
	[dBCdyb,ignored,ignored1,nbc] = odenumjac(bcaux,bcArgs,bcVal,dBCoptions)
	nCalls = nCalls + nbc
	# if npar > 0:
	#	bcArgs = [ya,yb,ExtraArgs[np.arange(nExtraArgs)]]
	#	dBCoptions.thresh = repmat(1e-6,npar,1)
	#	dBCoptions.diffvar = 3 
	#	[dBCdpar,ignored,ignored1,nbc] = odenumjac(bc,bcArgs,bcVal,dBCoptions)
	#	nCalls = nCalls + nbc
	
	return dBCdya,dBCdyb,nCalls # ,dBCdpar	#Optional argument associated with the above code block


# #--------------------------------------------------------------------------

def midptreg(iidx,mid):
	a025 = mid[:,iidx,0]
	a05 = mid[:,iidx,1]
	a075 = mid[:,iidx,2]	
	return a025, a05, a075

	
def midpti(i, b025, b05, b075):
	a025 = b025[:,i]
	a05 = b05[:,i]
	a075 = b075[:,i]	
	return a025, a05, a075

	
