import pickle

import numpy as np
from scipy import linalg
from scipy.interpolate import interp1d
import scikits.bvp_solver
import matplotlib.pyplot as plt

import gKdVEvans
import EvansBin


def eigf_profilesolver(s,p,m): 
	
	n=3 #Dimension of first order eigenvalue equation
	
	print '\n' + 'gKdV_eigf_profilesolver' + '\n'

	if s['flag']!='restarting':
		x = np.linspace(0, s['I'], p['temp_n'])
		y = range(0,7)
		for j in range(0,7):
			f1 = interp1d(x, np.real(s['guess'][j,:]), kind='cubic')
			y[j] = lambda z,j=j:interp1d(x, np.real(s['guess'][j,:]), kind='cubic')(z)

		
		guess = lambda x:np.array([f(x) for f in y])
		print "guess = \n", guess
		# import sys; sys.exit()
	else:
		solution = s['solution']
		guess = lambda X: solution(X).T[0] if X<=s['I'] else solution(s['I']).T[0]
	
	##########################################################
	def F(x,y):
		return (gKdVEvans.A(x,y[-1],s,p)).dot(y[:-1])

	##########################################################
	def function(x,y):
		yr = list(y)[0:n]; yr.append(y[-1])
		yl = list(y)[n:2*n]; yl.append(y[-1])
		out = np.concatenate( ( (s['R']/s['I'])*F( (s['R']/s['I'])*x,np.array(yr) ),
		(s['L']/s['I'])*F( (s['L']/s['I'])*x,np.array(yl) ),np.array([0]) ) )

	 	return out

	##########################################################
	def boundary_conditions(Ya,Yb):
		aa = [0, 0, 0]; aa.append((Ya[-1] + Yb[-1])/2.0)
		AM = Flinear( np.array(aa) ); AP = AM;

		P,Q = EvansBin.projection2(AM,-1,1e-6)
		LM = linalg.orth(P.T)
		P,Q = EvansBin.projection2(AP,1,1e-6)
		LP = linalg.orth(P.T)
	 
		BCa = list(Ya[0:n]-Ya[n:2*n]) #matching conditions = 3

		BCa.append( Ya[s['ph'][0]] - s['ph'][1] )      # phase condition = 1 
		BCb = list(LP.T.dot(Yb[0:n])) #at +infty; = 2
		BCb.append( (LM.T.dot(Yb[n:2*n]))[0] )#at -infty; = 1
		
		return np.array(BCa,dtype = float), np.array(BCb,dtype = float)
			    
	##########################################################
	def Flinear(y):
		out = np.array([[0,      1,  0],\
						[0,      0,  1],\
						[-y[-1], 1,  0]])
		return out
	##########################################################
	
	def plot_eigf(solution, problem,n_points,filestring): 
	
		x = np.linspace(problem.boundary_points[0],problem.boundary_points[1], 100)

		fig = plt.figure() 
		for j in range(0,n):
			plt.plot(x, solution(x)[j,:],'k')
			plt.plot(-np.flipud(x),np.flipud(solution(x)[j+m['n'],:]) ,'k')
			
		plt.title('Eigenfunction for gKdV equation\n Computed by bvp_solver')
		plt.xlabel('x'); pylab.ylabel('W '); format = '.eps'
		# plt.savefig('gKdV/gKdV_Data/'+filestring+format)
		# plt.show()
		# plt.clf()
		
		return 0
	##########################################################
	
	def my_bpv_solver():
			
		problem = scikits.bvp_solver.ProblemDefinition(num_ODE = 7,
                                  num_parameters = 0,
                                  num_left_boundary_conditions = 4,
                                  boundary_points = (0, s['I']),
                                  function = function,
                                  boundary_conditions = boundary_conditions)

		solution = scikits.bvp_solver.solve(problem,
                            solution_guess = guess,
                            trace = 0)
#                             parameter_guess = np.array([15.0]),

		return problem, solution
	
	##########################################################
	
	def file_name(): 
		filestring = 'gKdV_' + 'p' + '_'+str(p['p']).replace('.','_')+ '_'+ \
					'Ifty' +'_'+ ('%-4.2f'% (s['I'])).replace('.','_')
					
		return filestring
	##########################################################
	


	'''Creates rec to record results of continuation'''
	incr_p = 0.05; LengTH = int(round((p['target_p']-p['p'])/incr_p) + 1)

	param = np.linspace(p['p'],p['p'] + incr_p*(LengTH-1),LengTH);
	param = np.array([param]).T
	rec = np.concatenate( ( param,np.zeros(param.shape) ),axis = 1)
	
	target_I = 10.0 #(P,I) = (5.2,20), (10,10): Now creating a linear relation
	S_I = target_I*np.ones(param.shape) -(10.0/4.8)*(param - 10*np.ones(param.shape) ) 
	rec = np.concatenate( (rec,S_I),axis = 1); rec[0,1] = p['rooot']; 
	del S_I; del param

	'''Solves the boundary value problem as p['p'] increments '''
	for j in range(0,LengTH): 
		problem, solution = my_bpv_solver()
                
		guess = lambda X: solution(X).T[0] if X<=s['I'] else solution(s['I']).T[0]

		y = solution(0); rec[j,1] = y[-1,0]
		
		
# 		p['rooot'] = y[-1,0]; rec[j,1] = p['rooot']
		
		print '\n'*2
		print 'j = '+str(j)+':'
		print 'Parameter p = ', p['p']
		print 'lambda = ', rec[j,1]
		print 'Numerical Infinity = ', s['I']
		
		filestring = file_name()

		if abs(round(p['p'])- p['p']) < 0.001: 
			gKdV_Dict = {'p':p['p'],'I':s['I'],'solution':solution}
		
			File = open('gKdV_Data/Solution_'+filestring+ '.pkl','wb')
			pickle.dump(gKdV_Dict,File); File.close()
		
			File = open('gKdV_Data/restart_gKdV.pkl','wb')
			pickle.dump(gKdV_Dict,File); File.close()
# 			plot_eigf(solution, problem,100,filestring)
		
		# File = open('gKdV/gKdV_Data/rec.pkl','wb')
		File = open('gKdV_Data/rec.pkl','wb')

		pickle.dump(rec,File); File.close()
		
		if j !=LengTH-1:
			p['p'] = rec[j+1,0]
			s['I'] = rec[j+1,2]; s['R'] = s['I']; s['L'] = -s['I']

	File = open('gKdV_Data/rec.pkl','rb')
	REC = pickle.load(File)
	File.close()
	print '\n'*2,'REC = \n', REC,'\n'*3
	
	plt.clf()
	plt.plot(rec[:,0],rec[:,1],'*k')
	plt.plot(rec[:,0],rec[:,1],'-k')
	plt.show()
	return 0