import sys
import pickle
import matplotlib.pyplot as plt

import numpy as np
from scipy import linalg
from scipy.interpolate import interp1d
import scikits.bvp_solver

from comb_bvp_solvers import init_prof_guess, prof_solver
from comb_bvp_solvers import eigf_solver

# def file_name(s,p):
# 	filestring = 'comb_' + 'beta' + '_'+str(p['beta']).replace('.','_')+ '_'+ \
# 				'Ifty' +'_'+ ('%-4.2f'% (s['I'])).replace('.','_')
#
# 	return filestring
#
#
# def plot_this_profile(solution, problem,n_points,titlestring,filestring,s):
# 	x = np.linspace(
# 				problem.boundary_points[0],
# 				problem.boundary_points[1],
# 				200)
#
# 	n = s['n']
# 	for j in range(0,n-1):
# 		plt.plot( (s['R']/s['I'])*x, solution(x)[j,:],'k')
# 		plt.plot( (s['L']/s['I'])*np.flipud(x),
# 				  np.flipud(solution(x)[j+n,:]) ,'k')
#
#
# 	plt.title('Profile wave for the combustion equation'+
# 			  '\n Computed by bvp_solver')
# 	plt.savefig('combustion/combustion_Data/'+filestring+'.eps')
# 	plt.show()
# 	return


def profilecontinuation(s,p): 
	
	if not s['restarting']:
		# beta_vals=np.array([4.0,5,6,7,8,9,10,10.25])
		# beta_vals=np.array([4.0,5,6,7,8,9,10,10,11.5395,11.5395,11.5395])
		#,11.54,11.55,11.6,11.7,11.8,11.9,12.0])
		# np.arange(4,17,0.5)
		beta_vals = np.array([4.0,5,6,7,7.1,7.2])
	else:
		# beta_vals=np.array([10.25,10.25,10.30,10.35])
		beta_vals=np.arange(p['beta'],17,.05)
	p['beta'] = beta_vals[0]
	S_R = 8.4*np.exp(0.54 * beta_vals)
	# s.update({
	# 			'R': S_R[0],
	# 			'L': -S_R[0]
	# 			})
	
	
	'''Creates REC to record results of continuation'''
	REC = np.concatenate( 
							( np.array([beta_vals]).T,
							np.zeros(np.array([beta_vals]).T.shape),
							np.array([S_R]).T 
							),
							axis = 1
						)
	
	
	
	ARR = np.linspace(0.,s['I'],400)
	'''Creates initial guesses '''
	if s['restarting']: 
		prof_guess = interp1d(ARR,s['solution'](ARR),kind='cubic')
		# prof_guess = lambda X: s['solution'](X).T[0] if X<=s['I'] else s['solution'](s['I']).T[0]
	else: 
		prof_guess = lambda x,s=s,p=p: init_prof_guess(x,s,p)
	'''Solves the boundary value problem as p['beta'] increments '''
	for j in range(0, len(beta_vals)): 
		# Update variables
		p['beta'] = REC[j,0]
		s.update({
					'R':REC[j,2],
					'L':-REC[j,2]
					})
		
		problem_prof, solution_prof = prof_solver(prof_guess,s,p)
		prof_guess = lambda X: (1.0*beta_vals[j-1]/beta_vals[j])*solution_prof(X).T[0] if X<=s['I'] else (beta_vals[j-1]/beta_vals[j])*solution_prof(s['I']).T[0]
		s['solution'] = interp1d(ARR,solution_prof(ARR),kind='cubic')
		if s['eigf']:
			problem_eigf, solution_eigf = eigf_solver(s,p)
			eigf_guess = interp1d(ARR,solution_eigf(ARR),kind='cubic')
			s['eigfsolution'] = interp1d(ARR,solution_eigf(ARR),kind='cubic')
		
		
		REC[j,1] = solution_prof(0)[3,0] # Records wave speed
		print '\n'*2
		print 'j = '+str(j)+':'
		print '[beta, wave speed, domain length] = \n', REC[j,:]
		if s['eigf']: 
			print "Eigenvalue lambda ="
			print solution_eigf(0)[-2]+ 1j*solution_eigf(0)[-1]
		
		comb_Dict = {
						'beta':p['beta'],
						'I':s['I'],
						'solution':solution_prof,
						'REC':REC
						}
		
		
		File = open('combustion/combustion_Data/comb_restart2.pkl','wb')
		pickle.dump(comb_Dict,File); File.close()
#########################################################
		
	print "REC = ", REC
	return s,p,solution_prof, problem_prof

if __name__ == "__main__":
	s = {'restarting':False}
	p = {}
	profilecontinuation(s,p)



