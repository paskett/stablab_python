import numpy as np
from scipy import linalg
import pylab
from combustion.Evans_comb import A
import bin.EvansBin as EvansBin
import scikits.bvp_solver
import pickle
import sys


def profile_F(x,y,p): 
	if y[0]>=0.0001: e = np.exp(-1.0/y[0])
	else: e = 0.0
	z=y[3]; ve = (z - p['beta']*z*y[0]- p['beta']*y[1] - p['tau']*y[2] )*e
	out = np.array([y[1], \
			  -z*y[1] - ve/z, \
    		-z*y[2]/p['tau'] + p['beta']*ve/(p['tau']*z), \
				 0.0]);
	return out



def Flinear(y,speed,p):
	beta=p['beta']; tau=p['tau'];  
	
	if y[0] ==0: e = 0.
	elif y[0]>0: e = np.exp(-beta)
	
	out = np.array([[ 0.,                1.,                    0. ],
       [1.*beta*e,           -speed+1.*beta*e/speed,        1.*tau*e/speed], 
       [-1.*beta**2*e/tau,  -1.*beta**2*e/(tau*speed),  -1.*speed/tau - 1.*beta*e/speed ]])
	return out



def double_profile_F(x,y,s,p):
	n=s['n']; yr = list(y)[0:n]; yl = list(y)[n:2*n]
	
	out = np.concatenate( ( (s['R']/s['I'])*profile_F( x,np.array(yr),p ),
		(s['L']/s['I'])*profile_F( x,np.array(yl),p ) ) )
 	return out



def init_prof_guess(x,s,p):
	alpha = .45
	speed = np.sqrt(2./(p['beta']*p['tau']))*np.exp(-p['beta']/2.)
	
	UL = [1./p['beta'], 0, 0]; UR = [0, 0, 0]
	NInfty=s['L']; PInfty=s['R']; Infty=s['I']
	a = (UL[0]+UR[0])/2.; c = (UL[0]-UR[0])/2.
	
	out = np.zeros(8)
	out[0] = a-c*np.tanh(alpha*(PInfty/Infty)*x)
	out[1:3] = -c*alpha*np.cosh(alpha*(PInfty/Infty)*x)**(-2)
	out[3]=speed
	out[4] = a-c*np.tanh(alpha*(NInfty/Infty)*x)
	out[5:6] = -c*alpha*np.cosh(alpha*(NInfty/Infty)*x)**(-2)   
	out[7] = speed
	return out



def profile_boundary_conditions(Ya,Yb,s,p):
	n=s['n']
	speed=Ya[3]
	UL = [1.0/p['beta'], 0, 0]; UR = [0, 0, 0]
	
	AM = Flinear( np.array(UL) ,speed,p)
	AP = Flinear( np.array(UR) ,speed,p)
	
	P,Q = EvansBin.projection1(AM,-1,0.0) # Decay manifold at - infty
	LM = linalg.orth(P.T)
	P,Q = EvansBin.projection1(AP,0,1e-8) # Growth manifold at + infty
	LP = linalg.orth(P.T)
	
	BCa = list(Ya[0:n]-Ya[n:2*n]) #matching conditions = 4
	BCa.append( Ya[0] - 0.5/p['beta'])      # phase condition = 1 
	
	BCb = list(LM.T.dot(Yb[n:2*n-1] - UL)) # at -infty; = 2
	BCb.append( (LP.T.dot(Yb[0:n-1]-UR))[0]) # at +infty; = 1
	
	return np.real(np.array(BCa,dtype = complex)), np.real(np.array(BCb,dtype = complex))



def my_prof_bpv_solver(prof_guess,s,p):
	
	pro_problem = scikits.bvp_solver.ProblemDefinition(num_ODE = 8,
                                 num_parameters = 0,
                                 num_left_boundary_conditions = 5,
                                 boundary_points = (0, s['I']),
                                 function = lambda x,y,s=s,p=p:double_profile_F(x,y,s,p),
                                 boundary_conditions = lambda Ya,Yb,s=s,p=p:profile_boundary_conditions(Ya,Yb,s,p))
	
	pro_solution = scikits.bvp_solver.solve(pro_problem,
                           solution_guess = prof_guess,
                           trace = 0,
                           tolerance=1e-8)
#                           parameter_guess = np.array([15.0]),
	return pro_problem, pro_solution



def file_name(s,p): 
	filestring = 'comb_' + 'beta' + '_'+str(p['beta']).replace('.','_')+ '_'+ \
				'Ifty' +'_'+ ('%-4.2f'% (s['I'])).replace('.','_')
	
	return filestring



def plot_this_profile(solution, problem,n_points,titlestring,filestring,s):
	x = np.linspace(problem.boundary_points[0],problem.boundary_points[1], 200)	
	n = s['n']
	fig = pylab.figure() 
	for j in range(0,n-1):
		pylab.plot((s['R']/s['I'])*x, solution(x)[j,:],'k')
		pylab.plot((s['L']/s['I'])*np.flipud(x),np.flipud(solution(x)[j+n,:]) ,'k')
	
	pylab.title('Profile wave for the combustion equation\n Computed by bvp_solver')
# 	pylab.xlabel('x'); pylab.ylabel('u '); 
	format = '.eps'
	fig.savefig('combustion/combustion_Data/'+filestring+format)
	pylab.show()
	
	return 0



def profilesolver(s,p): 
	
	if s['flag'] != 'restarting':
		beta_vals=np.array([4.0,5,6,7,7.1])
# 		beta_vals=np.array([4.0,5,6,7,8,9,10,10.25])
# 		beta_vals=np.array([4.0,5,6,7,8,9,10,11.5395])
		# beta_vals = np.arange(4.,17.,1.)
		p['beta'] = beta_vals[0]
		S_R= 8.4*np.exp(0.54*beta_vals); LengTH = len(S_R)
		s['R'] = S_R[0]; s['L'] = -s['R']
	else: 
# 		beta_vals=np.array([10.25,10.25,10.30,10.35])
		beta_vals=np.arange(p['beta'],17,.5)
		S_R= 8.4*np.exp(0.54*beta_vals); LengTH = len(S_R)	
		s['R'] = S_R[0]; s['L'] = -s['R']
		p['beta'] = beta_vals[0]
	
	'''Creates a REC to record results of continuation'''
	REC = np.concatenate( ( np.array([beta_vals]).T,np.zeros(np.array([beta_vals]).T.shape),
	np.array([S_R]).T ),axis = 1); del S_R
	
	'''Solves the boundary value problem as p['beta'] increments '''
	print '\n'*2
	
	for j in range(0,LengTH): 
		if j==0: 
			if s['flag'] == 'restarting': 
				prof_guess = lambda X: s['solution'](X).T[0] if X<=s['I'] else s['solution'](s['I']).T[0]
			else:
				prof_guess = lambda x,s=s,p=p: init_prof_guess(x,s,p)
			problem_prof, solution_prof = my_prof_bpv_solver(prof_guess,s,p)
			print "Continuation completed successfully for beta = ", p['beta']#,', I = ',s['R']
		else: 
			prof_guess = lambda X: (1.0*beta_vals[j-1]/beta_vals[j])*solution_prof(X).T[0] if X<=s['I'] else (beta_vals[j-1]/beta_vals[j])*solution_prof(s['I']).T[0]
			problem_prof, solution_prof = my_prof_bpv_solver(prof_guess,s,p)
			print "                                               ", p['beta']#,', I = ',s['R']
			
# 		s['solution'] = solution_prof; p['speed'] = solution_prof(0)[3,0]
		
# 		REC[j,1] = solution_prof(0)[3,0] # Records wave speed
		print 'wave speed = ',solution_prof(0)[3,0]
		File = open('combustion/combustion_Data/REC.pkl','wb')
		pickle.dump(REC,File); File.close()
		
# 		plot_this_profile(solution_prof, problem_prof,200,'hello','testfile',s)
		if j !=LengTH-1:             # Here beta and the domain length are updated
			p['beta'] = REC[j+1,0]
			s['R'] = REC[j+1,2];  s['L'] = -s['R']
		####################################################
	print '\n'*2
	
	comb_Dict = {'beta':p['beta'],'I':s['I'],'solution':solution_prof}
	File = open('combustion/combustion_Data/restart_combustion.pkl','wb')
	pickle.dump(comb_Dict,File); File.close()
	filestring = file_name(s,p)
	plot_this_profile(solution_prof, problem_prof,200,'hello','testfile',s)
	print "wave speed = ", REC[j,1]
	return s,p,solution_prof, problem_prof



	
	
	
	
# 		sys.exit()
# 		filestring = file_name(s,p)
# 		plot_this_profile(solution_prof, problem_prof,200,'hello','testfile',s)
#######################################################
# 		comb_Dict = {'beta':p['beta'],'I':s['I'],'solution':solution_prof}
		
# 		File = open('combustion/combustion_Data/Solution_'+filestring+ '.pkl','wb')
# 		pickle.dump(comb_Dict,File); File.close()
# 		
# 		File = open('combustion/combustion_Data/restart_combustion.pkl','wb')
# 		pickle.dump(comb_Dict,File); File.close()
	
	