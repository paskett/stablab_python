
import numpy as np
from stablab import (semicirc2, winding_number, Evans_plot, emcset,
                     Evans_compute, Struct, profile_flux, soln, reflect_image)
import matplotlib.pyplot as plt

#-----------------------------------
# System specific tools
#-----------------------------------

def profile_jacobian(y, p):
    v = y[0]
    e = y[1]
    J = np.array([[(1/p['mu'])*(2*v-1-p['Gamma']*p['e_minus']), p['Gamma']/p['mu']],
                  [(1/p['nu'])*(-((v-1)**2)/2 + e - p['e_minus'] + (v-1)*p['Gamma']*e)+(v/p['nu'])*(-(v-1)+p['Gamma']*p['e_minus']), v/p['nu']]])
    return J

def profile_ode(x, y, sol, p):
    # Note: this is in Lagrangian coordinates
    v = y[0]
    e = y[1]

    y = np.vstack([(1/p['mu'])*(v*(v-1)+p['Gamma']*(e-v*p['e_minus'])),
                   (v/p['nu'])*(-((v-1)**2)/2+e-p['e_minus']+(v-1)*p['Gamma']*p['e_minus'])])

    return y

def A(x,lam,sol,p):

    y = soln([x],sol)
    y = y[0]

    Ux = profile_ode(x,y,sol,p)
    f = 2*y[0]-1-p['Gamma']*p['e_minus']
    g = (p['Gamma']*y[1]-(p['nu']+1)*Ux[0])/p['nu']

    a21 = lam*y[0]/p['nu']
    a22 = y[0]/p['nu']
    a23 = y[0]*Ux[0]/p['nu'] - f*Ux[0] - p['Gamma']*Ux[1]
    a24 = lam*g
    a25 = g+Ux[1]/y[0];
    a53 = lam*y[0]+p['Gamma']*Ux[0];
    a55 = f-lam;


    out = np.array([[0,        1,        0,        0,        0],
                    [a21,      a22,      a23,      a24,      a25],
                    [0,        0,        0,        lam,      1],
                    [0,        0,        0,        0,        1],
                    [0,        p['Gamma'],  a53,  lam*y[0], a55]])
    return out

def A_k(x,lam,s,p):

    y = soln([x],sol)
    y = y[0]

    Ux = profile_ode(x,y,sol,p)
    f = 2*y[0]-1-p['Gamma']*p['e_minus']
    g = (p['Gamma']*y[1]-(p['nu']+1)*Ux[0])/p['nu']

    a21 = lam*y[0]/p['nu']
    a22 = y[0]/p['nu']
    a23 = y[0]*Ux[0]/p['nu'] - f*Ux[0] - p['Gamma']*Ux[1]  # Is the second half correct for U_x_x
    a24 = lam*g
    a25 = g+Ux[1]/y[0];
    a53 = lam*y[0]+p['Gamma']*Ux[0];
    a55 = f-lam;

    out = np.array([[a22, a23, a24, a25, 0, 0, 0, 0, 0, 0],
                    [0, 0, lam, 1, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
                    [p['Gamma'], a53, lam*y[0], a55, 0, 0, 1, 0, 0, 0],
                    [0, a21, 0, 0, a22, lam, 1, -a24, -a25, 0],
                    [0, 0, a21, 0, 0, a22, 1, a23, 0, -a25],
                    [0, 0, 0, a21, a53, lam*y[0], a22 + a55, 0, a23, a24],
                    [0, 0, 0, 0, 0, 0, 0, 0, 1, -1],
                    [0, 0, 0, 0, -p['Gamma'], 0, 0, lam*y[0], a55, lam],
                    [0, 0, 0, 0, 0, -p['Gamma'], 0, -a53, 0, a55]])

    return out


def setParams():
    p = Struct({'Gamma': 2/3,
            'v_plus': 0.7,
            'mu': 1,
            'cnu': 1,
            'kappa': 1 })
    p.update({
            'v_star': p['Gamma']/(p['Gamma']+2) # correct see 2.33
            })
    p.update({
            'e_plus': p['v_plus']*(p['Gamma']+2-p['Gamma']*p['v_plus'])/(2*p['Gamma']*(p['Gamma']+1)), # 2.35
            'e_minus': (p['Gamma']+2)*(p['v_plus']-p['v_star'])/(2*p['Gamma']*(p['Gamma']+1)), # 2.34
            'v_minus': 1,
            'nu': p['kappa']/p['cnu'] # see below 2.25
            })
    return p


if __name__ == "__main__":
    # Inputs
    L = 10
    X_STEPS = 200
    p = setParams()

    # Initialising sol, the dict with solution values
    sol = Struct({
        'n': 2, # this is the dimension of the profile ode
        # we divide the domain in half to deal with the
        # non-uniqueness caused by translational invariance
        # sol.side = 1 means we are solving the profile on the interval [0,X]
        'side': 1,
        'F': profile_ode, # F is the profile ode
        'Flinear': profile_jacobian, # J is the profile ode Jacobian
        'UL': np.array([p['v_minus'],p['e_minus']]), # These are the endstates of the profile and its derivative at x = -infty
        'UR': np.array([p['v_plus'],p['e_plus']]), # These are the endstates of the profile and its derivative at x = +infty
        'tol': 1e-7
        })
    sol.update({
        'phase': 0.5*(sol['UL']+sol['UR']), # this is the phase condition for the profile at x = 0
        'order': [1], # this indicates to which component the phase conditions is applied
        'stats': 'on', # this prints data and plots the profile as it is solved
        'bvp_options': {'Tol': 1e-6, 'Nmax': 200}
        })

    p,s = profile_flux(p,sol)

    x = np.linspace(s['L'],s['R'],200)
    y = soln(x,s)

    plt.figure("Profile")
    plt.plot(x,y)
    plt.show()

    # set STABLAB structures to local default values
    #s, e, m, c = emcset(s, 'front', [2,3], 'reg_adj_compound', A, A_k)
    s, e, m, c = emcset(s,'front',[2,3],'reg_reg_polar',A)

    circpnts, imagpnts, innerpnts = 20, 30, 5
    r = 10
    spread = 4
    zerodist = 10**(-2)
    # ksteps, lambda_steps = 32, 0
    preimage = semicirc2(circpnts, imagpnts, innerpnts, c['ksteps'],
                             r,spread,zerodist,c['lambda_steps'])

    pre_w,preimage = Evans_compute(preimage,c,s,p,m,e)
    pre_w = pre_w/pre_w[0]
    w = reflect_image(pre_w)
    windnum = winding_number(w)

    Evans_plot(w)
    plt.show()
