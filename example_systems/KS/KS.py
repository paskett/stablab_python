import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp

#Import from stablab
from stablab import Evans_compute, winding_number, Evans_plot, emcset, Struct
from stablab.contour import semicirc, semicirc2
from stablab.evans import Aadj
from stablab.periodic_contour import periodic_contour
from stablab.wave_profile import deval, soln
from stablab.root_finding import moments_roots

# import the local functions.
from ks_guess import ks_guess
from ks_profile2 import ks_profile2
from ks_taylor import ks_taylor

def A(x,lamda,s,p):
    # Evans functin matrix for KS equation
    #
    # Here we are taking f(u)=u**2/2 and c=0.

    temp = deval(x,s.sol)
    u = temp[0]
    u_x = temp[1]

    out = np.array([
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [-lamda-u_x, -u, -p.delta, -p.epsilon]
            ])

    return out

def f(u):

	fu = u**2/2
	fu_x = u
	return np.array([fu,fu_x])


def ks_profile(s,p):
        # bvp solver for KS periodic profile equation. Here p is a structure
        # containing the parameters X=period, q=integration constant,
        # beta = parameter, epsilon = parameter, detla=paramter.
        # The parameter beta is treated as a free variable.
        # The structure s either contains a matlab ode/bvp solution structure,
        # s.sol which provides an initial guess for the bvp solver, or if it does
        # not, then an initial guess is provided which corresponds to q=6,
        # epsilon=0.2, X=6.3.

        # change defalut tolerance settings if not specified.
        if 'options' not in s:
                s.options = Struct()
                s.options.AbsTol = 10**(-6)
                s.options.RelTol = 10**(-8)

        # check if an initial guess is provided.
        if 'sol' not in s:
                #ld=load('profile_starter');
                #s.sol=ld.s.sol
                pass

        # define an anomynuous function to allow the passing of the parameter p
        # without using a global variable.
        ode = lambda x,y,delta: (periodic_ode(x,y,delta,p));

        # define anonymous functions to avoid global variables
        guess = lambda x: (interpolate(x,s.sol))

        #Define the default guess function if none is loaded.
        initialGuessFunction = lambda x: np.array([np.sin(2*np.pi/p.X*x),np.cos(2*np.pi/p.X*x)*2*np.pi/p.X,-np.sin(2*np.pi/p.X*x)*4*np.pi**2/p.X**2])

        #Create the initial space and the initial guess.
        initSpace = np.linspace(0,p.X,83)
        initialGuess = initialGuessFunction(initSpace)
        initialGuess = ks_guess() #loads from the file the guess.

        #Solve the bvp with the initial guess and return the solutions.
        sol = solve_bvp(ode,bc,initSpace,initialGuess,p=[p.delta])
        delta = sol.p
        return sol, delta

#Define the periodic bc function, three periodic boundary conditions
#And a phase condition.
def bc(ya,yb,delta):
        out=np.array(
        [ya[0]-yb[0],
        ya[1]-yb[1],
        ya[2]-yb[2],
        ya[1]
        ])
        return out

#Define the periodic ode function
def periodic_ode(x,y,delta,p):
        u = y[0]
        u_x = y[1]
        u_xx = y[2]
        out = np.vstack([
            u_x,
            u_xx,
            -p.epsilon*u_xx-delta*u_x-f(u)[0]+p.q
            ])
        return out

#Interpolate x through the
def interpolate(x,sol):
        out = np.real(deval(sol,x))
        return out


#Demonstrates stability for the KS system via a high frequency
# study and a low frequency study utilizing a Taylor coefficient
# expansion.
if __name__ == "__main__":

    # Declare the dictionary p.
    p = Struct()
    p.X = 6.3
    p.epsilon = 0
    p.delta = 1
    p.q = 5.5

    # -------------------------------------------------------------------------
    # compute profile
    # -------------------------------------------------------------------------

    # Declare the dictionaries s and d
    s = Struct()
    d = Struct()
    s.something = 1

    # find profile solution
    s.sol, p.delta = ks_profile(s,p)
    d.s = s # record parameters in data structure
    d.p = p # record parameters in data structure

    #Plot the profile solution
    #plt.plot(s.sol.y[0])
    #plt.plot(s.sol.y[1])
    #plt.plot(s.sol.y[2])
    #plt.show()

    # Find the L infinity norm of the profile u and u'
    x = np.linspace(0,p.X,1000)
    temp = s.sol.sol(x)
    Linf_bar_u = max(np.abs(temp[0,:]))
    Linf_bar_u_prime = max(np.abs(temp[1,:]))

    # record the L infinity norms in the data structure
    d.Linf_bar_u = Linf_bar_u
    d.Linf_bar_u_prime = Linf_bar_u_prime

    # STABLAB structuress
    s.X = p.X
    # s,e,m,c = emcset(s,'periodic',[2,2],'balanced_polar_periodic',A)
    s,e,m,c = emcset(s,'periodic',[2,2],'balanced_polar_scaled_periodic',A)
    # s,e,m,c = emcset(s,'periodic',[2,2],'regular_periodic',A)
    c.ksteps = 100000
    c.refine = 'on'
    c.stats = 'print'
    m['options']['nsteps'] = 500 # max steps for each iteration of ODE solver

    # Determine the HF bound and record it.
    R = 0.5*(Linf_bar_u_prime+Linf_bar_u**2+p.delta**2+0.5*(1+2*p.epsilon**2)**2)
    R1 = max(R+0.1,2)
    R2 = 1
    d.R = R
    d.R1 = R1
    d.R2 = R2

    kappa = np.linspace(-np.pi/p.X,np.pi/p.X,1000);
    circpnts = 10; imagpnts = 10; innerpnts = 5; spread = 4;
    # circpnts=50; imagpnts=50; innerpnts=20; spread=4;
    preimage = semicirc2(circpnts,imagpnts,innerpnts,c.ksteps,R1,spread,R2)
    preimage = np.concatenate((preimage[0:-1], np.flipud(np.conj(preimage))))

    # compute the Evans fucntion for HF study
    startTime = time.time()
    D, preimage2 = periodic_contour(kappa,preimage,c,s,p,m,e)
    totTime = time.time() - startTime

    # HF study output and statistics
    d.D1 = Struct()
    d.D1.time = totTime
    d.D1.points = len(preimage2)
    d.D1.D = D
    d.D1.preimage2 = preimage2

    # Determine the maximum winding number for different values of kappa
    max_wnd = 0
    for j in np.arange(1,len(kappa)):
    	max_wnd = max(max_wnd,winding_number(D[j-1,:]))
    d.D1.max_wnd = max_wnd

    #--------------------------------------------------------------------------
    # Compute the Taylor Coefficients and find eigenvalue expansion
    # coefficients
    #--------------------------------------------------------------------------

    p,s,d,st = ks_taylor(s,p,m,c,e,d)

    # -------------------------------------------------------------------------
    # Find the maximum value of lambda(k), |k|=R_remainder
    # -------------------------------------------------------------------------

    c.ksteps = 10000
    c.tol = 0.2
    c.refine = 'on'
    e.evans = 'balanced_polar_periodic'

    R_remainder = 0.5
    kappa = R_remainder*np.exp(1j*np.linspace(-np.pi,np.pi,1000))

    points = 100
    R_lambda = 2
    preimage = R_lambda*np.exp(1j*
                    np.linspace(0,2*np.pi,points+(points-1)*c.ksteps))

    print('\nFinding the maximum value of lambda')

    # Find max lambda(k) on ball of radius R_remainder
    s_time = time.time()
    D,preimage2 = periodic_contour(kappa,preimage,c,s,p,m,e)
    currTime = (time.time()-s_time)

    max_rt = 0
    for j in np.arange(1,len(kappa)):
        if winding_number(D[j-1,:]) > 1:
            rts = moments_roots(preimage2,D[j-1,:])
            max_rt = max(max_rt,max(np.abs(rts)))
        else:
            raise ValueError('Contour radius not big enough')

    beta1 = d.beta1
    beta2 = d.beta2
    alpha1 = d.alpha1
    alpha2 = d.alpha2

    # determine r satisfying the requirements
    r1 = -np.real(beta1)*R_remainder**3/(max_rt-R_remainder**2*np.real(beta1))
    r2 = -np.real(beta2)*R_remainder**3/(max_rt-R_remainder**2*np.real(beta2))
    r3 = -R_remainder**3*np.real(beta1)/(16*max_rt)
    r4 = -R_remainder**3*np.real(beta2)/(16*max_rt)
    r5 = -np.real(beta1)*R_remainder**3/(R_lambda-R_remainder**2*np.real(beta1))
    r6 = -np.real(beta2)*R_remainder**3/(R_lambda-R_remainder**2*np.real(beta2))
    r7 = -R_remainder**3*np.real(beta1)/(16*R_lambda)
    r8 = -R_remainder**3*np.real(beta2)/(16*R_lambda)
    r_remainder = min([r1,r2,r3,r4,r5,r6,r7,r8])

    # Raise some errors on invalid values
    if r_remainder > 0.5*R_remainder:
        raise ValueError('r bigger than R/2')
    if abs(np.imag(alpha1))*r_remainder+np.sqrt(2)*np.abs(beta1)*r_remainder**2 > R2:
        raise ValueError('imaginary part too big')
    if abs(np.imag(alpha2))*r_remainder+np.sqrt(2)*np.abs(beta2)*r_remainder**2 > R2:
        raise ValueError('imaginary part too big')
    else:
        d.test = ( abs(np.imag(alpha2)) * r_remainder + np.sqrt(2) * np.abs(beta2)
                    * r_remainder**2 )

    d.max_rt = max_rt
    d.lambda_points = len(preimage2)
    d.R_remainder = R_remainder
    d.r_remainder = r_remainder

    #--------------------------------------------------------------------------
    # Compute the Evans fucntion on semicircular contour of radius 2R2 for
    # r<|k|<pi/X.
    #--------------------------------------------------------------------------

    e.evans = 'balanced_polar_scaled_periodic'
    c.ksteps = 10000
    c.tol = 0.2
    c.refine = 'on'
    kappa = np.concatenate([np.linspace(-np.pi/p.X,-r_remainder,500),
                            np.linspace(r_remainder,np.pi/p.X,500)])

    circpnts = 20; imagpnts = 20; spread = 4; inner_gap = 10**(-6);
    preimage = semicirc(circpnts,imagpnts,c.ksteps,2*R2,spread,inner_gap)
    preimage = np.concatenate((preimage[0:-1], np.flipud(np.conj(preimage))))

    s_time = time.time()
    D,preimage2 = periodic_contour(kappa,preimage,c,s,p,m,e)
    currTime = (time.time()-s_time)

    max_wnd = 0
    for j in np.arange(1,len(kappa)):
        max_wnd = max(max_wnd,winding_number(D[j-1,:]))

    # record Evans function output in data structure
    d.D2 = Struct()
    d.D2.max_wnd = max_wnd
    d.D2.points = len(preimage2)
    d.D2.preimage2 = preimage2
    d.D2.D = D
    d.D2.time = currTime

    # -------------------------------------------------------------------------
    # Compute Evans function for semi-circle shifted left and for
    # |k|<r
    # -------------------------------------------------------------------------

    e.evans = 'balanced_polar_periodic'
    c.ksteps = 10000
    c.tol = 0.2
    c.refine = 'on'
    kappa = np.linspace(-r_remainder,r_remainder,1000)

    shift = 2*max(abs(np.real(beta1)),abs(np.real(beta2)))*r_remainder**2

    circpnts = 20; imagpnts = 20; spread = 4; inner_gap = 10**(-6);
    preimage = semicirc(circpnts,imagpnts,c.ksteps,2*R2,spread,inner_gap)
    preimage = np.concatenate((preimage[0:-1], np.flipud(np.conj(preimage))))
    preimage = preimage - shift

    # compute Evans function
    s_time = time.time()
    D, preimage2 = periodic_contour(kappa,preimage,c,s,p,m,e)
    currTime = (time.time()-s_time)

    # verify the winding number is 2
    for j in np.arange(1,len(kappa)):
        if not (winding_number(D[j-1,:]) == 2):
            # raise ValueError('winding number not two');
            print('\n\nWARNING: Winding number is: ',winding_number(D[j-1,:]),
                                                                        '\n\n')

    # record Evans function output in data structure
    d.shift = shift
    d.D3 = Struct()
    d.D3.preimage2 = preimage2
    d.D3.D = D
    d.D3.points = len(preimage2)
    d.D3.time = currTime
