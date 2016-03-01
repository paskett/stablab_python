import sys
import pickle

import matplotlib.pyplot as plt

from bin import EvansBin
from bin.EvansDF import *
from bin.Boussinesq_eigf_profilesolver import eigf_profilesolver


# # parameters
p = {'S':0.25, 'rooot':3.247596e-1,'target_S':0.48}

L = 23.0
s = {'I':L,'R':L,'L':-L,'side':1,'F':'F','larray':[4,5,6,7],'rarray':[0,1,2,3]}
s['UL'], s['UR'] = [0,0,0,0], [0,0,0,0]

s['PFunctions'] = 'bin.BoussinesqEvans' # Must be called before calling emcset
[s,e,m,c] = emcset(s,'front',2,2,'reg_reg_polar')
m['method'] = EvansBin.mod_drury
m['options'] = {'RelTol':1e-8,'AbsTol':1e-9}
p['temp_n'] = 100

# 
# Finding initial eigenpair
#
s, x1, x2 = eigf_init_guess(s,e,m,c,p)


fig = plt.figure() 
for j in range(0,m['n']):
	plt.plot(x1,np.flipud(s['guess'][j+m['n']-1,:]) ,'k')
	plt.plot(np.flipud(x2), s['guess'][j,:],'k')
	
plt.title('Eigenfunction for Boussinesq equation')
plt.xlabel('x')
plt.ylabel('W (Initial Guess)')
plt.savefig('Boussinesq_Data/' + 'Boussinesq_eigenfunction_guess' + '.eps')
plt.show()


s['ph'] = [1,s['guess'][1,0]]

'''Uses bvp_solver to refine initial estimate 
of the eigenfunction '''
s['flag'] = 'NA' # or 

'''Commence with saved data'''
# File = open('gKdV/gKdV_Data/restart_gKdV.pkl','rb')
# gKdV_Dict = pickle.load(File)
# p['rooot'] = gKdV_Dict['solution'](0)[-1,0]
# s['solution'] = gKdV_Dict['solution']
# p['p'] = gKdV_Dict['p']
# s['I'] = gKdV_Dict['I']
# s['R'] = s['I']; s['L'] = -s['I']
# s['flag'] = 'restarting'; p['target_p'] = 6.2

out = eigf_profilesolver(s,p,m) 
