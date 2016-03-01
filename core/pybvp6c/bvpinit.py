from structure_variable import *

def bvpinit(x,v,*args):
	"""
BVPINIT  Form the initial guess for BVP4C.
   SOLINIT = BVPINIT(X,YINIT) forms the initial guess for BVP4C in common
   circumstances. The boundary value problem (BVP) is to be solved on [a,b].
   The vector X specifies a and b as X(1) = a and X(end) = b. It is also
   a guess for an appropriate mesh. BVP4C will adapt this mesh to the solution,
   so often a guess like X = linspace(a,b,10) will suffice, but in difficult
   cases, mesh points should be placed where the solution changes rapidly.

   The entries of X must be ordered. For two-point BVPs, the entries of X
   must be distinct, so if a < b, then X(1) < X(2) < ... < X(end), and
   similarly for a > b. For multipoint BVPs there are boundary conditions
   at points in [a,b]. Generally, these points represent interfaces and
   provide a natural division of [a,b] into regions. BVPINIT enumerates
   the regions from left to right (from a to b), with indices starting
   from 1. You can specify interfaces by double entries in the initial
   mesh X. BVPINIT interprets oneentry as the right end point of region k
   and the other as the left end point of region k+1. THREEBVP exemplifies
   this for a three-point BVP.

   YINIT provides a guess for the solution. It must be possible to evaluate
   the differential equations and boundary conditions for this guess.
   YINIT can be either a vector or a function:

      vector:  YINIT(i) is a constant guess for the i-th component Y(i,:) of
            the solution at all the mesh points in X.

   function:  YINIT is a function of a scalar x. For example, use
              solinit = bvpinit(x,@yfun) if for any x in [a,b], yfun(x)
              returns a guess for the solution y(x). For multipoint BVPs,
              BVPINIT calls Y = YINIT(X,K) to get an initial guess for the
              solution at x in region k.

   SOLINIT = BVPINIT(X,YINIT,PARAMETERS) indicates that the BVP involves
   unknown parameters. A guess must be provided for all parameters in the
   vector PARAMETERS. pybvp6c: PARAMETERS must be a list ?

   SOLINIT = BVPINIT(X,YINIT,PARAMETERS,P1,P2...) passes the additional
   known parameters P1,P2,... to the guess function as YINIT(X,P1,P2...) or
   YINIT(X,K,P1,P2) for multipoint BVPs. Known parameters P1,P2,... can be
   used only when YINIT is a function. When there are no unknown parameters,
   use SOLINIT = BVPINIT(X,YINIT,[],P1,P2...).

   SOLINIT = BVPINIT(SOL,[ANEW BNEW]) forms an initial guess on the interval
   [ANEW,BNEW] from a solution SOL on an interval [a,b]. The new interval
   must be bigger, so either ANEW <= a < b <= BNEW or ANEW >= a > b >= BNEW.
   The solution SOL is extrapolated to the new interval. If present, the
   PARAMETERS from SOL are used in SOLINIT. To supply a different guess for
   unknown parameters use SOLINIT = BVPINIT(SOL,[ANEW BNEW],PARAMETERS).
   Note, this has not been extended for bvp6c (i.e. the new computed
   initial guess from a 6th of bvp6c solution will be only 4th order).

   See also BVPGET, BVPSET, BVP4C, BVP6C, DEVAL, NTRP6C, @.

   Jacek Kierzenka and Lawrence F. Shampine
   Copyright 1984-2003 The MathWorks, Inc.
   $Revision: 1.11.4.2 $  $Date: 2003/05/19 11:15:02 $
   BVP6C Modification
    Nick Hale  Imperial College London
    $Date: 12/06/2006 $
	"""
# Extend existing solution?
	
	nargin = len(list( locals() )) -1 + len(args)
	if nargin>=3:
		parameters = args[0]
		if nargin >3:
			varargin = args[1:]
			extraArgs = varargin
		else: 
			extraArgs = ()
	else: 
		extraArgs = ()
	
	solinit = struct()
	if x.__class__.__name__ =='struct':
		# if (nargin < 2) or (len(v) < 2):
		# 	import sys
		# 	print 'MATLAB:bvpinit:NoSolInterval',\
		#           'Did not specify [ANEW BNEW] in BVPINIT(SOL,[ANEW BNEW]).'
		# 	sys.exit()
		# elif nargin < 3:
		if nargin <3:
			parameters = []
		solinit = bvpxtrp(x,v,parameters)
		return solinit
	
		
# Create a guess structure.
	N = len(x)
	if x[0] == x[-1]:
		raise ValueError, ('Python:bvpinit:XSameEndPts'+
					'The entries of x must satisfy a = x(1) ~= x(end) = b.')
	elif x[0] < x[-1]:
		if True in (np.diff(x) < 0):
			raise ValueError,('Python:bvpinit:IncreasingXNotMonotonic'+
					'The entries of x must satisfy a = x(1) < x(2) < ... < x(end) = b.')
	else:  # x(1) > x(N):
		if True in (np.diff(x) > 0):
			raise ValueError,('Python:bvpinit:DecreasingXNotMonotonic'+
			'The entries of x must satisfy a = x(1) > x(2) > ... > x(end) = b.')	
	
	if nargin>2:
		params = parameters
	else:
		params = []
	
	
	
	mbcidx = np.where(np.diff(x) == 0)[0]  # locate internal interfaces
	# ismbvp = ~isempty(mbcidx) 
	if len(mbcidx)>0:
		Lidx = np.concatenate( (np.array([0]), mbcidx) ) # [1, mbcidx+1]
		Ridx = np.concatenate( (mbcidx,np.array( [len(x)-1] ) ) ) # [mbcidx, len(x)]
	
	if isinstance(v,np.ndarray):
		# combustion does not enter here
		m,n = v.shape
		if n==1 and m==1:        # must have at least two BPts and two functions
			raise TypeError, ('Python:bvpinit:SolGuessNotVector' +
					'The guess for the solution must return a vector.')
		elif n==1 or m==1:
			L=len(v)
			yinit=np.zeros( (L,len(x)) )
			for i in range(0,L):
				yinit[i,:]=v[i]*np.ones( (1,len(x)) )
		elif m == len(x):
			yinit=np.copy(v.T)
		elif n == len(x):
			yinit=np.copy(v)
		else:
			raise TypeError, ('Python:bvpinit:SolGuessNotVector' +
						'The guess for the solution must return a vector.')
	else:
  #checking  
		if len(mbcidx)>0:
			# combustion does not enter here
			w = v(x[0],1,extraArgs)# check region 1, only.
		else:
			# combustions enters here
			w = v(x[0])
		# w = np.array([[0]]) # temporary
		# m,n = w.shape
		m = w.shape
		# print "w.shape = \n", w.shape
		if not (len(m)==1) and not (m[0]>0):
			raise TypeError, ('Python:bvpinit:SolGuessNotVector' +
			'The guess for the solution must return a vector.')
			
  #assigning
		if len(mbcidx)>0:
			# combustion does not enter here
			for region in range(0,nregions):
				for i in range(Lidx(region)-1,Ridx(region)):
					yinit[:,i] = v(x[i],region, extraArgs)
       	
		else:
			# combustion does enter here
			length = len(v(x[0]))
			yinit = np.empty((length,N))
			yinit[:,0] = w[:]
			for i in xrange(1,N):
				yinit[:,i] = v(x[i])
			
	
	solinit.x = np.copy(x[:].T) # row vector
	solinit.y = np.copy(yinit)
	if len(params)>0:
		solinit.parameters = params
	
	return solinit


#---------------------------------------------------------------------------

def bvpxtrp(sol,v,parameters):
# Extend a solution SOL on [sol.x(1),sol.x(end)]
# to a guess for [v(1),v(end)] by extrapolation.
	
	a, b = sol.x[0], sol.x[-1]
	
	anew, bnew = v[0], v[-1]
	
	if (a < b and (anew > a or bnew < b)) or (a > b and (anew < a or bnew > b)):
		raise ValueError,('Python:bvpinit:bvpxtrp:SolInterval'+
					'The call BVPINIT(SOL,[ANEW BNEW]) must have\n'+
					'ANEW <= SOL.x(1) < SOL.x(end) <= BNEW  or \n'+
					'ANEW >= SOL.x(1) > SOL.x(end) >= BNEW. \n')
					
	
	solxtrp.x = np.copy(sol.x)
	solxtrp.y = np.copy(sol.y)
	eps = np.finfo(np.float).eps
	if abs(anew - a) <= 100.*eps*max(abs(anew),abs(a)):
		solxtrp.x[0] = anew
	else:
		S = Hermite(sol.x(1),sol.y[:,1],sol.yp[:,1],sol.x(2),sol.y[:,2],sol.yp[:,2],anew)
		solxtrp.x = np.concatenate( (anew, solxtrp.x)) # [anew solxtrp.x]				# What structure should this be?
		solxtrp.y = np.concatenate( (S, solxtrp.y)) #[S solxtrp.y]							# What structure should this be?
		
	if abs(bnew - b) <= 100.*eps*max(abs(bnew),abs(b)):
		solxtrp.x[-1] = bnew
	else:
		S = Hermite(sol.x[-2],sol.y[:,-2],sol.yp[:,-2],sol.x[-1],sol.y[:,-1],sol.yp[:,-1],bnew)
		solxtrp.x = np.concatenate( (solxtrp.x, bnew) )				# What structure should this be?
		solxtrp.y = np.concatenate( (solxtrp.y, S) )						# What structure should this be?
	
	if len(parameters)>0:
		solxtrp.parameters = parameters
	elif hasattr(sol,'parameters'):
		solxtrp.parameters = sol.parameters
		
	return solxtrp

		
#---------------------------------------------------------------------------

def Hermite(x1,y1,yp1,x2,y2,yp2,xi):
# Evaluate cubic Hermite interpolant at xi.
	h = x2 - x1
	s = (xi - x1)/h
	A1 = (s - 1.)**2. * (1. + 2.*s)
	A2 = (3. - 2.*s)*s**2.
	B1 = h*s*(s - 1.)**2.
	B2 = h*s**2. *(s - 1.)
	S = A1*y1 + A2*y2 + B1*yp1 + B2*yp2
	
	return S

