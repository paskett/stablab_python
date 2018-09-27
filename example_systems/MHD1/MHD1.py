import numpy as np
import matplotlib.pyplot as plt
from stablab import (semicirc, winding_number, Evans_plot, emcset,
                       Evans_compute, Struct, reflect_image)
from stablab.wave_profile import profile_flux,soln

def profile_jacobian(y, p):

    J = np.array([((2*p.mu+p.eta)**(-1)*(y-1+p.a*(y**(-p.gamma)-1))+
         (2*p.mu+p.eta)**(-1)*y*(1-p.a*p.gamma*y**(-p.gamma-1)))])

    return J

def profile_ode(x, y, sol, p):

    out = (2*p.mu+p.eta)**(-1)*y*(y-1+p.a*(y**(-p.gamma)-1))

    return out

def A(x,lam,sol,p):

    v = soln(x,sol)
    v = v[0]
    mu = p['mu']
    sigma = p['sigma']
    B = p['B']
    mu0 = p['mu0']

    out = np.array([[0, 1/mu, 0, 0 ],
                    [lam*v, v/mu, 0, -sigma*B*v ],
                    [0, 0, 0, v*sigma*mu0 ],
                    [0, -B*v/mu, lam*v, v**2*sigma*mu0 ]],dtype=np.complex)

    return out

def Ak(x,lam,sol,p):

    v = soln(x,sol)
    v = v[0]
    mu = p['mu']
    sigma = p['sigma']
    B = p['B']
    mu0 = p['mu0']

    out = np.array([
        [ v/mu, 0, -B*sigma*v, 0, 0, 0],
        [ 0, 0, mu0*sigma*v, 1/mu, 0, 0],
        [-B*v/mu, lam*v, mu0*sigma*(v**2), 0, 1/mu, 0],
        [ 0, lam*v, 0, v/mu, mu0*sigma*v, B*sigma*v],
        [ 0, 0, lam*v, lam*v, mu0*sigma*v**2 + v/mu, 0],
        [ 0, 0, 0, B*v/mu, 0, mu0*sigma*(v**2)]],dtype=np.complex)

    return out

if __name__ == "__main__":

    # -------------------------------------------------------------------------
    # parameters
    # -------------------------------------------------------------------------

    p = Struct()

    p.B = 2
    p.gamma = 5/3
    p.mu0 = 1
    p.sigma = 1
    p.vp = 0.0001
    p.mu = 1
    p.eta = -2*p.mu/3

    # -------------------------------------------------------------------------
    # dependent parameters
    # -------------------------------------------------------------------------

    p.a = p.vp**p.gamma*((1-p.vp)/(1-p.vp**p.gamma))

    # Initialising sol, the dict with solution values
    sol = Struct({
        'n': 1, # this is the dimension of the profile ode
        # we divide the domain in half to deal with the
        # non-uniqueness caused by translational invariance
        # sol.side = 1 means we are solving the profile on the interval [0,X]
        'side': 1,
        'F': profile_ode, # F is the profile ode
        'Flinear': profile_jacobian, # J is the profile ode Jacobian
        'UL': np.array([1]), # These are the endstates of the profile and its derivative at x = -infty
        'UR': np.array([p.vp]), # These are the endstates of the profile and its derivative at x = +infty
        'tol': 1e-6
        })
    sol.update({
        'phase': 0.5*(sol['UL']+sol['UR']), # this is the phase condition for the profile at x = 0
        'order': [0], # this indicates to which component the phase conditions is applied
        'stats': 'on', # this prints data and plots the profile as it is solved
        'bvp_options': {'Tol': 1e-6, 'Nmax': 200}
        })

    # Solve the profile
    p,s = profile_flux(p,sol)

    x = np.linspace(s['L'],s['R'],200)
    y = soln(x,s)

    # Plot the profile
    plt.figure("Profile")
    plt.plot(x,y)
    plt.show()

    s, e, m, c = emcset(s,'front',[2,2],'reg_adj_compound',A,Ak)

    circpnts, imagpnts, innerpnts = 30, 30, 32
    r = 1
    spread = 4
    zerodist = 10**(-4)
    # ksteps, lambda_steps = 32, 0
    preimage = semicirc(circpnts, imagpnts, c['ksteps'], r, spread, zerodist)

    out, domain = Evans_compute(preimage,c,s,p,m,e)
    out = out/out[0]
    w = reflect_image(out)
    windnum = winding_number(w)

    print('Winding Number: {:f}\n'.format(windnum))

    Evans_plot(w)
