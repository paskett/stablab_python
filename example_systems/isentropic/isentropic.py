import numpy as np
import matplotlib.pyplot as plt
import stablab

"""
def A(x, lamda, s, p):
    v = stablab.soln(x,s)
    f = v-v**(-p['gamma'])*(-v**(p['gamma']+1)+p['a']*(p['gamma']-1)
        +(p['a']+1)*v**(p['gamma']))

    out = np.array([[0, lamda, 1],
                    [0, 0, 1],
                    [lamda*v, lamda*v, f-lamda]],dtype=np.complex)

    return out
"""
def A(x, lamda, s, p):
    gamma = p['gamma']
    a = p['a']

    v = stablab.soln(x,s)
    v_prime = profile_ode(x, v, s, p)
    v_prime_prime = (1 - a*gamma*v**(-gamma-1))*v_prime
    alpha = a*gamma*(v**(-gamma-1)) - v_prime/(v**2)
    A31 = (lamda*alpha*v - a*gamma*(-gamma-1)*(v**(-gamma-1))
            + v_prime_prime/v + 2*v_prime/v**2)

    out = v*np.array([[-lamda, 0, 1],
                    [0,0,1],
                    [A31, lamda*v, v - alpha*v + v_prime/v]])#,dtype=np.complex)
    return out

def Ak(x, lamda, s, p):

    out = A(x, lamda, s, p)

    return out

def profile_ode(x, v, s, p):

    out = np.array(v*(v-1+p['a']*(v**(-p['gamma'])-1)))

    return out

def profile_jacobian(v, p):

    out = np.array([(v-1+p['a']*(v**(-p['gamma'])-1))+v*(1+p['a']
                    *(-p['gamma']*v**(-p['gamma']-1)))])

    return out

if __name__ == "__main__":
    # parameters
    p = stablab.Struct()
    p.gamma = 5.0/3
    p.vp = 0.1
    # dependent parameters
    p.a = -(1-p.vp)/(1-p.vp**(-p.gamma))

    # structure variables
    s = stablab.Struct()
    s.n = 1 # this is the dimension of the profile ode
    # we divide the domain in half to deal with the
    # non-uniqueness caused by translational invariance
    # s.side = 1 means we are solving the profile on the interval [0,X]
    s.side = 1
    s.L = -15
    s.R = 15
    s.F = profile_ode # profile_ode is the profile ode
    s.Flinear = profile_jacobian # profile_jacobian is the profile ode Jacobian
    s.UL = np.array([1]) # These are the endstates of the profile at x = -infty
    s.UR = np.array([p.vp]) # These are the endstates of the profile at x = +infty
    s.phase = 0.5*(s.UL+s.UR) # this is the phase condition for the profile at x = 0
    s.order = [0] # this indicates to which componenet the phase conditions is applied
    s.stats = 'on' # this prints data and plots the profile as it is solved
    s.bvp_options = {'Tol': 1e-9}
    p,s = stablab.profile_flux(p,s) # solve profile for first time

    # plot the profile
    x = np.linspace(s.L,s.R,200)
    y = stablab.soln(x,s)
    plt.plot(x,y)
    plt.show()


    # This choice solves the right hand side via exterior products
    s.A = A
    s.Ak = Ak
    #s,e,m,c = stablab.emcset(s, 'front', [1,2], 'reg_adj_compound', A, Ak)
    s,e,m,c = stablab.emcset(s, 'front', [1,2], 'reg_reg_polar', A)
    #s,e,m,c = emcset(s,'front',[1,2],'reg_adj_polar')
    m.options = {}
    m.options['AbsTol'] = 1e-9
    m.options['RelTol'] = 1e-9

    # display a waitbar
    c.stats = 'print' # 'on', 'print', or 'off'
    c.ksteps = 2**6

    # Preimage Contour
    # This is a semi circle. You can also do a semi annulus or a rectangle
    circpnts = 30
    imagpnts = 10
    innerpts = 10
    R = 10
    spread = 2
    zerodist = 1
    preimage = stablab.semicirc2(circpnts,imagpnts,innerpts,c.ksteps,R,spread,zerodist)

    # compute Evans function
    halfw, domain = stablab.Evans_compute(preimage,c,s,p,m,e)
    w = halfw / halfw[0]
    # We compute the Evans function on half of contour then reflect
    w = stablab.reflect_image(w)

    # process and display data
    wnd = stablab.winding_number(w) # determine the number of roots inside the contour
    print('Winding Number: {:f}\n'.format(wnd))

    # plot the Evans function (normalized)
    stablab.Evans_plot(w)
