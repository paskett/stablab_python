import sys
import pickle

import matplotlib.pyplot as plt
# Must be imported here
from bin import EvansBin
from bin.EvansDF import *
from bin.gKdV_eigf_profilesolver import eigf_profilesolver


# # parameters
p = {'p':5.2,'rooot':0.098034924900383,'target_p':10.0}

# 
# Finding initial eigenpair
# 

L = 20.0
s = {'I':L,'R':L,'L':-L,'side':1,'F':'F','larray':[3,4,5,6],'rarray':[0,1,2,6]}
s['UL'], s['UR'] = [0,0,0], [0,0,0]

s['PFunctions'] = 'bin.gKdVEvans' # Must be called before calling emcset
[s,e,m,c] = emcset(s,'front',2,1,'reg_reg_polar')
m['method'] = EvansBin.mod_drury
m['options'] = {'RelTol':1e-8,'AbsTol':1e-9}
p['temp_n'] = 100

s, x1, x2 = eigf_init_guess(s,e,m,c,p)
# print x1.shape, x2.shape,s['guess'].shape

# sys.exit()


fig = plt.figure() 
for j in range(0,m['n']):
	plt.plot(x1,np.flipud(s['guess'][j+m['n']-1,:]) ,'k')
	plt.plot(np.flipud(x2), s['guess'][j,:],'k')
	
plt.title('Eigenfunction for gKdV equation')
plt.xlabel('x')
plt.ylabel('W (Initial Guess)')
plt.savefig('gKdV_Data/' + 'gKdV_eigenfunction_guess' + '.eps')
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
