import numpy as np
import matplotlib.pyplot as plt
from stablab import (semicirc, winding_number, Evans_plot, emcset,
                       Evans_compute, Struct, soln, profile_flux,
                       reflect_image)
from stablab.root_finding import root_solver1

def A(x, lamda, s, p):
    v = soln(x,s)
    b = 1/v[0]
    h = 1-p['a']*p['gamma']*v[0]**(-p['gamma']-1) + v[1]/(v[0]**2) - lamda/v[0]

    out = np.array([[0,         lamda,      1,        0],
                    [0,         0,          1,        0],
                    [0,         0,          0,        1],
                    [lamda/p['d'], lamda/p['d'],  h/p['d'],  (-b/p['d'])]])
    return out

def Ak(x, lamda, s, p):
    v = soln(x,s)
    b = 1/v[0]
    h = 1-p['a']*p['gamma']*v[0]**(-p['gamma']-1) + v[1]/(v[0]**2) - lamda/v[0]

    out = np.array(
          [[0,            1,         0,        -1,          0,        0],
           [0,            0,         1,        lamda,       0,        0],
           [lamda/p['d'], h/p['d'], -b/p['d'],  0,           lamda,   1],
           [0,             0,         0,        0,          1,        0],
           [-lamda/p['d'], 0,         0,        h/p['d'],  -b/p['d'], 1],
           [0,       -lamda/p['d'],   0,    -lamda/p['d'],   0,   -b/p['d']]])
    return out

def profile_ode(x, y, s, p):
    out = np.array([ y[1] ,
                    (y[0]-s['UL'][0]+p['a']*((y[0]**(-p['gamma'])
                        -s['UL'][0]**(-p['gamma']))) - y[1] / y[0])/p['d']])
    return out

def profile_jacobian(U, p):
    p_prime = -p['gamma'] * U[0]**(-p['gamma']-1)
    out = np.array([[0, 1],
                    [(1+p['a']*p_prime)/p['d'],  1/(-U[0]*p['d'])]])
    return out

if __name__ == "__main__":
    # Check out
    # http://math.byu.edu/~jeffh/publications/papers/cap.pdf
    # for details about this system

    # parameters
    p = Struct()
    p.gamma = 1.4
    p.vp = .15
    # This example solves the profile for $d = -0.45$ and then uses
    # continuation thereafter
    d_vals = -np.arange(0.45, 0.451, 0.001)

    # dependent parameters
    p.a = -(1-p.vp)/(1-p.vp**(-p.gamma))

    # solve profile. Use continuation as an example
    s = Struct()
    for j,curr_d_val in enumerate(d_vals):
        p.d = curr_d_val

        # profile
        s.n = 2 # this is the dimension of the profile ode
        # we divide the domain in half to deal with the
        # non-uniqueness caused by translational invariance
        # s.side = 1 means we are solving the profile on the interval [0,X]
        s.side = 1
        s.F = profile_ode # profile_ode is the profile ode
        s.Flinear = profile_jacobian # profile_jacobian is the profile ode Jacobian
        s.UL = np.array([1, 0]) # These are the endstates of the profile and its derivative at x = -infty
        s.UR = np.array([p.vp, 0]) # These are the endstates of the profile and its derivative at x = +infty
        s.phase = 0.5*(s.UL+s.UR) # this is the phase condition for the profile at x = 0
        s.order = [0] # this indicates to which componenet the phase conditions is applied
        s.stats = 'on' # this prints data and plots the profile as it is solved

        if j == 0:
            # there are some other options you specify. You can look in profile_flux to see them
            p,s = profile_flux(p,s) # solve profile for first time
            s_old = s
        else:
            # this time we are using continuation
            p,s = profile_flux(p,s,s_old); # solve profile

    # plot the profile
    x = np.linspace(s.L,s.R,200)
    y = soln(x,s)
    plt.title("Profile")
    plt.plot(x,y[:,0])
    plt.plot(x,y[:,1])
    plt.show()

    # structure variables

    # Here you can choose the method you use for the Evans function, or you can set the option
    # to default and let it choose. [2,2] is the size of the manifold evolving from - and + infy in the
    # Evans solver. 'front' indicates you are solving for a traveling wave front and not a periodic solution
    # s,e,m,c = emcset(s,'front',LdimRdim(A,s,p),'default') # default for capillarity is reg_reg_polar
    # s,e,m,c = emcset(s,'front',[2,2],'reg_adj_polar')
    # s,e,m,c = emcset(s,'front',[2,2],'adj_reg_polar')
    # s,e,m,c = emcset(s,'front',[2,2],'reg_reg_polar')
    
    # This choice solves the right hand side via exterior products
    s,e,m,c = emcset(s, 'front', [2,2], 'adj_reg_compound', A, Ak)

    # display a waitbar
    c.stats = 'print' # 'on', 'print', or 'off'
    c.ksteps = 2**8

    # Preimage Contour
    # This is a semi circle. You can also do a semi annulus or a rectangle
    circpnts = 30
    imagpnts = 30
    R = 10
    spread = 2
    zerodist = 10**(-2)
    preimage = semicirc(circpnts,imagpnts,c.ksteps,R,spread,zerodist)

    # compute Evans function
    halfw, domain = Evans_compute(preimage,c,s,p,m,e)
    w = halfw / halfw[0]

    # We compute Evans function on half of contour then reflect across reals
    w = reflect_image(w)

    # Process and display data
    wnd = winding_number(w) # determine the number of roots inside the contour
    print("Winding Number: ",wnd)

    # plot the Evans function (normalized)
    Evans_plot(w)
