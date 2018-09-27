from scipy.integrate import solve_bvp
import numpy as np
from ks_stability import f
from ks_guess import ks_guess

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
        if not ('options' in s):
                s['options'] = {}
                s['options']['AbsTol'] = 10**(-6)
                s['options']['RelTol'] = 10**(-8)

        # check if an initial guess is provided.
        if not ('sol' in s):
                #ld=load('profile_starter');
                #s['sol']=ld['s']['sol']
                True

        # define an anomynuous function to allow the passing of the parameter p
        # without using a global variable.
        ode = lambda x,y,delta: (per_ode(x,y,delta,p));

        # define anonymous functions to avoid global variables
        guess=lambda x: (interpolate(x,s['sol']));

        #Define the default guess function if none is loaded.
        initialGuessFunction = lambda x: np.array([np.sin(2*np.pi/p['X']*x),np.cos(2*np.pi/p['X']*x)*2*np.pi/p['X'],-np.sin(2*np.pi/p['X']*x)*4*np.pi**2/p['X']**2])

        #Create the initial space and the initial guess.
        initSpace = np.linspace(0,p['X'],83)
        initialGuess = initialGuessFunction(initSpace)
        initialGuess = ks_guess() #loads from the file the guess.

        #Solve the bvp with the initial guess and return the solutions.
        sol = solve_bvp(ode,bc,initSpace,initialGuess,p=[p['delta']]);
        delta = sol['p'];
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
def per_ode(x,y,delta,p):
        u=y[0];
        u_x=y[1];
        u_xx=y[2];
        out = np.vstack([
            u_x,
            u_xx,
            -p['epsilon']*u_xx-delta*u_x-f(u)[0]+p['q']
            ])
        return out

#Interpolate x through the
def  interpolate(x,sol):
        out=np.real(deval(sol,x))
        return out
