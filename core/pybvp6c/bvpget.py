import sys

def bvpget(options,name,default=[]):
#BVPGET  Get BVP OPTIONS parameters.
#   VAL = BVPGET(OPTIONS,'NAME') extracts the value of the named property
#   from integrator options structure OPTIONS, returning an empty matrix if
#   the property value is not specified in OPTIONS. It is sufficient to type
#   only the leading characters that uniquely identify the property. Case is
#   ignored for property names. [] is a valid OPTIONS argument. 
#   
#   VAL = BVPGET(OPTIONS,'NAME',DEFAULT) extracts the named property as
#   above, but returns VAL = DEFAULT if the named property is not specified
#   in OPTIONS. For example 
#   
#       val = bvpget(opts,'RelTol',1e-4);
#   
#   returns val = 1e-4 if the RelTol property is not specified in opts.
#   
#   See also BVPSET, BVPINIT, BVP4C, DEVAL.
#
#   Jacek Kierzenka and Lawrence F. Shampine
#   Copyright 1984-2003 The MathWorks, Inc. 
#   $Revision: 1.11.4.2 $  $Date: 2003/10/21 11:55:35 $
#   BVP6C Modification
#    Nick Hale  Imperial College London
#    $Date: 12/06/2006 $
	
	
	if (not options.__class__.__name__ == 'struct'):
		print options.__class__.__name__
		print 'Python:bvpget:OptsNotStruct\n','First argument must be an options structure created with BVPSET.'
		sys.exit()
	
	Names = ['abstol',
    		'reltol' ,
    		'singularterm',
    		'fjacobian',
    		'bcjacobian',
    		'stats',
    		'nmax',
    		'vectorized',
    		'maxnewpts' ,
    		'slopeout' ,
    		'xint',
    		]
	
	# if name.rstrip().lower() not in Names:
	# 	import sys
	# 	print 'Python:bvpget:InvalidPropName'
	# 	print 'Unrecognized property name %s. See BVPSET for possibilities.'% name
	# 	sys.exit()
	# o = getattr(options,name.rstrip().lower())
	try:
		o = getattr(options,name.rstrip().lower())
	except: 
		o = default
	
	return o
