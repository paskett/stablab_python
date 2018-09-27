"""
Implements two functions: boussinesq and continuation_boussinesq.

boussinesq computes the Evans function along a contour for a fixed
value of the parameter S (wave speed).
"""

import numpy as np
import matplotlib.pyplot as plt

from stablab import (semicirc2, winding_number, Evans_plot, emcset,
                     Evans_compute, Struct, reflect_image)
from stablab.root_finding import root_solver1

def A(x,lamda,s,p):
    gamma = .5*np.sqrt(1-p['S']**2);
    u = 1.5*(1-p['S']**2)*(np.cosh(gamma*x))**(-2);
    ux = -2*gamma*u*np.tanh(gamma*x);
    uxx = 2*(gamma**2)*u*( 2-3*(np.cosh(gamma*x))**(-2) );

    a41 = -lamda**2-2*uxx;
    a42 = 2*lamda*p['S']-4*ux;
    a43 = (1-p['S']**2)-2*u;

    out = np.array([[0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                    [a41, a42, a43, 0]])
    return out

def Ak(x, lamda, s, p):

    gamma = .5*np.sqrt(1-p['S']**2)
    u = 1.5*(1-p['S']**2)*(np.cosh(gamma*x))**(-2)

    ux = -2*gamma*u*np.tanh(gamma*x)
    uxx = 2*(gamma**2)*u*( 2-3*(np.cosh(gamma*x))**(-2) )

    a41 = -lamda**2-2*uxx
    a42 = 2*lamda*p['S']-4*ux
    a43 = (1-p['S']**2)-2*u

    out = np.array([[0,       1,      0,    0,      0,    0],
                     [0,      0,      1,    1,      0,    0],
                     [a42,    a43,    0,    0,      1,    0],
                     [0,      0,      0,    0,      1,    0],
                     [(-a41), 0,      0,    a43,    0,    1],
                     [0,      (-a41), 0,    (-a42), 0,    0]])

    return out

if __name__ == "__main__":
    # parameters
    p = Struct({'S':0.4})

    # profile
    s = Struct()
    s.I = 8
    s.R = s.I
    s.L = -s.I
    s.A = A # The function A
    s.Ak = Ak

    # set STABLAB Structs to local default values
    #s,e,m,c = emcset(s,'front',[2,2],'reg_reg_polar')
    #s,e,m,c = emcset(s,'front',[2,2],'reg_adj_polar')
    #s,e,m,c = emcset(s,'front',[2,2],'adj_reg_polar')
    s,e,m,c = emcset(s,'front',[2,2],'reg_adj_compound')

    # display a waitbar
    c.stats = 'print'

    # Preimage
    points = 50
    preimage = 0.16+0.05*np.exp(2*np.pi*1j*np.linspace(0,0.5,points+(points-1)*c.ksteps)) #FIXME: this line could probably be easier to understand

    # Compute the Evans function
    halfw,preimage  = Evans_compute(preimage,c,s,p,m,e)
    w = reflect_image(halfw)
    # Normalize the Evans Output
    w = w/w[0]

    # Process and display data:
    wnd = winding_number(w)
    print("Winding Number: {:f}\n".format(wnd))
    Evans_plot(w)
    plt.show()

    #Set variables in preparation for root solving
    c.lambda_steps = 0
    c.stats = 'off'
    c.pic_stats = 'on'
    c.ksteps = 2**8
    c.moments = 'off'
    c.tol = 0.2
    c.root_fun = Evans_compute
    box = [0.1,-0.1,0.25,0.05]
    tol = 1e-3
    roots = root_solver1(box,tol,p,s,e,m,c)
    print(roots)
