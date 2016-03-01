from __future__ import print_function
import sys
import numpy as np
from ntrp6h import ntrp6h
from structure_variable import print_this

# @print_this
def deval(sol,xint,idx=(),Spxint_requested=False):
	"""
	Jacek Kierzenka and Lawrence F. Shampine
	Copyright 1984-2004 The MathWorks, Inc.
	BVP6C Modification
	Nick Hale	 Imperial College London
	$Date: 12/06/2006 $
	"""
	
	try: 
		t, y = sol.x, sol.y 
	except:
		raise TypeError, 'Python:deval:SolNotFromDiffEqSolver'
	
	# if idx==():
	# 	idx = np.arange(y.shape[0])
	# else:
	# 	raise SystemError
	idx = np.arange(y.shape[0])
	idx = idx[:]
	
	# Spxint_requested = (nargout > 1)	 # Evaluate the first derivative?
	
	n, Nxint = len(idx), len(xint)
	Sxint = np.zeros((n,Nxint))
	if Spxint_requested: 
		Spxint = np.zeros((n,Nxint))
	else: 
		Spxint = []
	
	# Make tint a row vector and if necessary, 
	# sort it to match the order of t.
	tint = xint[:].T
	tdir = np.sign(t[-1] - t[0])
	had2sort = np.where(tdir*np.diff(tint) < 0)
	had2sort = np.nonzero(had2sort)[0].size
	if had2sort > 0:
		print(np.nonzero(had2sort)[0].size)
		# print("Need to sort")
		tint,tint_order = np.sort(tdir*tint)
		tint = tdir*tint 
	# print("tdir =",tdir); print("tint =",tint);
	# # Using the sorted version of tint, test for illegal values.
	# if (tdir*(tint[0] - t[0]) < 0) or (tdir*(tint[-1] - t[-1]) > 0):
	#	error('MATLAB:deval:SolOutsideInterval',...
	#	['Attempting to evaluate the solution outside the interval\n'...
	#	 '[%e, %e] where it is defined.\n'],t(1),t(end))
	assert (not (tdir*(tint[0] - t[0]) < 0) or (tdir*(tint[-1] - t[-1]) > 0) )
	# Select appropriate interpolating function.
	# if ~isfield(sol,'yp') or ~isfield(sol,'ypmid'):
	#	error('MATLAB:deval:nsufficientData',...
	#	'bvp6c deval requires the option BVPSET(''SLOPEOUT'',''ON'')')
	interpfcn = ntrp6h
	
	evaluated = 0
	bottom = 0
	count = 0
	while evaluated < Nxint:
		
		# Find right-open subinterval [t(bottom), t(bottom+1))
		# containing the next entry of tint. 
		Index = np.where( tdir*(tint[evaluated] - t[bottom:]) >= 0 )[0]
		bottom += Index[-1]
		
		# Is it [t(end), t[-1]]?
		at_tend = (t[bottom] == t[-1])
		if at_tend:
			# Use (t[bottom-1], t[bottom]] to interpolate y(t[-1]) and yp(t[-1]).
			index = np.where(tint[evaluated:] == t[bottom])[0]
			bottom = bottom - 1
		else:
			# Interpolate inside [t[bottom], t[bottom+1]).
			index = np.where( tdir*(tint[evaluated:] - t[bottom+1]) < 0 )[0]
			
		interpdata = (sol.yp[:,bottom], sol.yp[:,bottom+1], sol.ypmid[:,bottom,:])
		
	# Evaluate the interpolant at all points from [t(bottom), t(bottom+1)).
		# if Spxint_requested:
		#	[yint,ypint] = feval(interpfcn,tint(evaluated+index),t(bottom),y(:,bottom),...
		#				 t(bottom+1),y(:,bottom+1),*interpdata)
		# else:
		#	yint = interpfcn(tint[evaluated+index],t[bottom],y[:,bottom],
		#			t[bottom+1],y[:,bottom+1],interpdata)	 
		
		
		yint, ypint = interpfcn(tint[evaluated+index],t[bottom],y[:,bottom],
											t[bottom+1],y[:,bottom+1],*interpdata)
		if at_tend: bottom += 1
		
		# Purify the solution at t[bottom].
		index1 = np.where(tint[evaluated+index] == t[bottom])[0]
		if len(index1)>0: 
			yint[idx,index1] = y[idx,bottom]
		
		# Accumulate the output.
		Sxint[:,evaluated+index] = yint[idx,:]	
		if Spxint_requested: Spxint[:,evaluated+index] = ypint[idx,:]
		evaluated += len(index)
		count +=1
		
		
	# End of the while loop.
	
	if had2sort:	 # Restore the order of tint in the output.
		Sxint[:,tint_order] = Sxint
		if Spxint_requested: Spxint[:,tint_order] = Spxint
	return Sxint,Spxint
