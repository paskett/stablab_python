'''
Implements two functions: boussinesq and continuation_boussinesq. 

boussinesq computes the Evans function along a contour for a fixed 
value of the parameter S (wave speed).
continuation_boussinesq is not yet fully implemented, but will track
the unstable eigenvalue of the Boussinesq system as S varies. 
'''
import numpy as np
import matplotlib.pyplot as plt

from core.contour import semicirc2, winding_number, Evans_plot
from core.evans import emcset, Evans_compute

import BoussinesqEvansSystems as EvansSystems
package_directory = '/Users/joshualytle/bin/projects/pystablab'

def boussinesq(p = {'S':0.04}, domain=[-10,10], verbose=True, Plot_Evans=False):
	'''
	 parameters contained in dictionary p
	'''
	
	# 
	#  numerical infinity
	# 
	if domain[0] < domain[1]:
		s = {'I':1,'R':domain[1],'L':domain[0]}
	else: 
		raise TypeError, "Error in the specified domain"
	
	# 
	# set STABLAB structures to local default values
	# 
	# Must be called before calling emcset / Boussinesq-specific
	
	s['EvansSystems'] = EvansSystems
	[s,e,m,c] = emcset(s,'front',2,2,'reg_reg_polar') 
	
	#  
	# # refine the Evans function computation to achieve set relative error
	# c['refine'] = 'on'
	
	
	'''Create the Preimage Contour'''
	# circpnts=5; imagpnts=5; innerpnts = 5
	# r=10; spread=4; zerodist=10**(-2)
	# ksteps = 32; lambda_steps = 0
	# preimage=semicirc2(circpnts,imagpnts,innerpnts,ksteps,
	# 					r,spread,zerodist,lambda_steps)
	
	points = 10;
	preimage = (0.44 + 0.05*np.exp(
				2*np.pi*1j*np.linspace(0,0.5,points+(points-1)*c['ksteps']))
				)
	
	'''Compute the Evans function'''
	out = Evans_compute(preimage,c,s,p,m,e)
	w = np.concatenate(( out,np.flipud(np.conj(out)) ))
	
	# 
	# Display Evans function output and statistics
	# 
	if verbose: 
		pass
		# print 'Evans Computation Successful'
		# print 'The winding number is ', winding_number(w)
	
	titlestring = ('Parameters for the Boussinesq equation: \n ' + 
					'S = ' +str(p['S'])+', I = '+str(s['I']))
	filestring = (package_directory + '/Boussinesq/data_Boussinesq/'+'Parameters_'+
					str(p['S'])+'_'+str(s['I'])        )
	labelstring = 'Evans Output'
	format = '.pdf'		# Possible formats: png, pdf, ps, eps and svg.
	Evans_plot(w,labelstring,titlestring, filestring, format,Plot_Evans)
	return 




def continuation_boussinesq(p = {'S':0.04}, domain=[-10,10], verbose=True):
	'''
	 parameters contained in dictionary p
	'''
	
	# 
	#  numerical infinity
	# 
	if domain[0] < domain[1]:
		s = {'I':1,'R':domain[1],'L':domain[0]}
	
	# 
	# set STABLAB structures to local default values
	# 
	# Must be called before calling emcset / Boussinesq-specific
	s['EvansSystems'] = EvansSystems
	[s,e,m,c] = emcset(s,'front',2,2,'reg_reg_polar') 
	
	'''
	BoussinesqEvans.py contains profile/ode functions specific to 
	computing the Evans function for the Boussinesq equation
	'''
	
	#  
	# # refine the Evans function computation to achieve set relative error
	# c['refine'] = 'on'
	
	
	'''Create the Preimage Contour'''
	# circpnts=5; imagpnts=5; innerpnts = 5
	# r=10; spread=4; zerodist=10**(-2)
	# ksteps = 32; lambda_steps = 0
	# preimage=semicirc2(circpnts,imagpnts,innerpnts,ksteps,
	# 					r,spread,zerodist,lambda_steps)
	
	points = 10;
	preimage = (0.44 + 0.05*np.exp(
				2*np.pi*1j*np.linspace(0,0.5,points+(points-1)*c['ksteps']))
				)
	
	'''Compute the Evans function'''
	out = Evans_compute(preimage,c,s,p,m,e)  # Standard approach with shooting
	
	# 
	# Display Evans function output and statistics
	# 
	if verbose:
		print 'Evans Computation Successful'
	w = np.concatenate(( out,np.flipud(np.conj(out)) ))
	if verbose:
		print 'The winding number is ', winding_number(w)
	
	titlestring = ('Continuation; Parameters for the Boussinesq equation: \n ' + 
					'S = ' +str(p['S'])+', I = '+str(s['I']))
	filestring = (package_directory+ '/Boussinesq/data_Boussinesq/'+'Continuation_Parameters_'+
					str(p['S'])+'_'+str(s['I'])        )
	labelstring = 'Evans Output'
	format = '.pdf'		# Possible formats: png, pdf, ps, eps and svg.
	plot_B = True
	Evans_plot(w,labelstring,titlestring, filestring, format,plot_B)
	return 



if __name__ == "__main__":
	boussinesq(Plot_Evans=False)#domain = [10,-10])
	# continuation_boussinesq()
	# pass
