"""
    Driver for flux and balanced flux formulation
    for full gas in Lagrangian coordinates
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import complex_ode

import stablab

def full_gas_Hugoniot(p, s, var):
    """
    solves the Hugoniot curve
    """

    if s.H_ind_var == "tau":
        def H(p,tau,S): return s.H_fun(p,tau,S)
    else:
        def H(p,tau,S): return s.H_fun(p,S,tau)

    left = s.left_H

    Hleft = np.sign(H(p,var,left))
    right = s.right_H
    Hright = np.sign(H(p,var,right))

    if np.sign(Hleft) == np.sign(Hright):
        x = np.linspace(left,right,100)
        y = np.zeros((length(x),1))
        for j,x_j in enumerate(x):
            y[j] = H(p,var,x_j);
        # FIXME: plotting may not yet work as intended here -- check it later
        plt.plot(x,y,'.-k')
        plt.plot([left,right],[0,0],'-g')
        plt.show()
        raise ValueError('Both end points have the same sign')

    while (right-left)/left > 1e-14:
        mid = 0.5*(left+right)
        Hmid = np.sign(H(p,var,mid))
        if Hleft == Hmid:
            left = mid
        else:
            right = mid

    out = 0.5*(right+left)
    return out

def H_stable_model(p, tau, S):
    """
    Hugoniot curve equation
    """
    S0 = p.S0
    tau0 = p.tau0
    none = p.none
    mu = p.mu
    kappa = p.kappa
    S_tau = np.exp(S)/tau
    S_tau_sq = np.exp(S)/tau**2
    S0_tau = np.exp(S0)/tau0
    S0_tau_sq = np.exp(S0)/tau0**2
    out = S_tau+tau**2/0.2e1-S0_tau-tau0**2/0.2e1+(S_tau_sq-tau+S0_tau_sq-tau0)*(tau-tau0)/0.2e1
    return out

def RH_stable_model(p):
    """
    Rankine Hugoniot conditions
    """
    st = stablab.Struct()
    st.H_fun = H_stable_model

    st.left_H = 1e-6
    st.right_H = 1000
    st.H_ind_var = 's'
    p.tau_neg = full_gas_Hugoniot(p,st,p.S_neg)

    p.S_plus = p.S0
    p.tau_plus = p.tau0

    none = p.none
    mu = p.mu
    kappa = p.kappa

    S = p.S_neg
    tau = p.tau_neg
    press_neg = np.exp(S) / tau ** 2 - tau
    T_neg = np.exp(S) / tau
    p.T_neg = T_neg

    S = p.S_plus
    tau = p.tau_plus
    press_plus = np.exp(S) / tau ** 2 - tau
    T_plus = np.exp(S) / tau
    p.T_plus = T_plus

    p.spd = -np.sqrt((press_neg-press_plus)/(p.tau_plus-p.tau_neg))

    p.m1 = 0

    p.v_plus = -p.spd*p.tau_plus-p.m1
    p.v_neg = -p.spd*p.tau_neg-p.m1

    return p

def F_stable_model(x,y,s,p):

    tau = y[0]
    S = y[1]
    # wave speed
    spd = p['spd']
    # values at - infinity
    v_neg = p['v_neg']
    tau_neg = p['tau_neg']
    S_neg = p['S_neg']
    # parameters
    m1 = p['m1']
    none = p['none']
    mu = p['mu']
    kappa = p['kappa']
    # other values
    v = v_neg - spd * (tau - tau_neg)
    S_tau = np.exp(S)/tau
    S_tau_sq = np.exp(S)/tau**2

    # profile equation
    out = np.array([
        -tau/spd/mu*(-spd*(v-v_neg)+S_tau_sq-tau-np.exp(S_neg)/
            tau_neg**2+tau_neg) ,
        tau*((-0.1e1/tau/spd/mu-0.1e1/np.exp(S)*tau*v/kappa)*(-spd*(v-v_neg)+
            S_tau_sq-tau-np.exp(S_neg)/tau_neg**2+tau_neg)+0.1e1/np.exp(S)*tau/
            kappa*(-spd*(S_tau+tau**2/0.2e1+v**2/0.2e1-np.exp(S_neg)/tau_neg-
            tau_neg**2/0.2e1-v_neg**2/0.2e1)+v*(S_tau_sq-tau)-v_neg*(
            np.exp(S_neg)/tau_neg**2-tau_neg)))])
    #debug(m1, none, mu, kappa, v, tau, S, out)
    #input("stophere")
    return out

def Flinear_stable_model(y,p):

    tau = y[0]
    S = y[1]
    # wave speed
    spd = p['spd']
    # values at - infinity
    v_neg = p['v_neg']
    tau_neg = p['tau_neg']
    S_neg = p['S_neg']
    # parameters
    m1 = p['m1']
    none = p['none']
    mu = p['mu']
    kappa = p['kappa']
    # Jacobian
    out = np.array([
        [-0.1e1/spd/mu*(spd**2*(tau-tau_neg)+np.exp(S)/tau**2-tau-np.exp(S_neg)/
            tau_neg**2+tau_neg)-tau/spd/mu*(spd**2-0.2e1*np.exp(S)/tau**3-
            0.1e1),
         -0.1e1/tau/spd/mu*np.exp(S) ],
        [(-0.1e1/tau/spd/mu-0.1e1/np.exp(S)*tau*(v_neg-spd*(tau-tau_neg))/
            kappa)*(spd**2*(tau-tau_neg)+np.exp(S)/tau**2-tau-np.exp(S_neg)/
            tau_neg**2 + tau_neg) + 0.1e1 / np.exp(S) * tau / kappa * (-spd *
            (np.exp(S) / tau + tau ** 2 / 0.2e1 + (v_neg - spd * (tau -
            tau_neg)) ** 2 / 0.2e1 - np.exp(S_neg) / tau_neg - tau_neg ** 2 /
            0.2e1 - v_neg ** 2 / 0.2e1) + (v_neg - spd * (tau - tau_neg)) *
            (np.exp(S) / tau ** 2 - tau) - v_neg * (np.exp(S_neg) /tau_neg**2 -
            tau_neg)) + tau * ((0.1e1 / tau ** 2 / spd / mu - 0.1e1 / np.exp(S)
            * (v_neg - spd * (tau - tau_neg)) / kappa + 0.1e1 / np.exp(S) * tau
            * spd / kappa) * (spd ** 2 * (tau - tau_neg) + np.exp(S) / tau ** 2
            - tau - np.exp(S_neg) / tau_neg ** 2 + tau_neg) + (-0.1e1 / tau /
            spd / mu - 0.1e1 / np.exp(S) * tau * (v_neg - spd * (tau -
            tau_neg)) / kappa) * (spd ** 2 - 0.2e1 * np.exp(S) / tau ** 3 -
            0.1e1) + 0.1e1 / np.exp(S) / kappa * (-spd * (np.exp(S) / tau +
            tau ** 2 / 0.2e1 + (v_neg - spd * (tau - tau_neg)) ** 2 / 0.2e1 -
            np.exp(S_neg) / tau_neg - tau_neg ** 2 / 0.2e1 - v_neg ** 2 /
            0.2e1) + (v_neg - spd * (tau - tau_neg)) * (np.exp(S) / tau ** 2 -
            tau) - v_neg * (np.exp(S_neg) / tau_neg ** 2 - tau_neg)) + 0.1e1 /
            np.exp(S) * tau / kappa * (-spd * (-np.exp(S) / tau ** 2 + tau -
            (v_neg - spd * (tau - tau_neg)) * spd) - spd * (np.exp(S) /
            tau ** 2 - tau) + (v_neg - spd * (tau - tau_neg)) * (-0.2e1 *
            np.exp(S) / tau ** 3 - 0.1e1))),
         tau * (0.1e1 / np.exp(S) *
            tau * (v_neg - spd * (tau - tau_neg)) / kappa * (spd ** 2 *
            (tau - tau_neg) + np.exp(S) / tau ** 2 - tau - np.exp(S_neg)/
            tau_neg ** 2 + tau_neg) + (-0.1e1 / tau / spd / mu - 0.1e1 /
            np.exp(S) * tau * (v_neg - spd * (tau - tau_neg)) / kappa)*
            np.exp(S) / tau ** 2 - 0.1e1 / np.exp(S) * tau / kappa * (-spd *
            (np.exp(S) / tau + tau ** 2 / 0.2e1 + (v_neg - spd * (tau -
            tau_neg)) ** 2 / 0.2e1 - np.exp(S_neg) / tau_neg - tau_neg ** 2 /
            0.2e1 - v_neg ** 2 / 0.2e1) + (v_neg - spd * (tau - tau_neg)) *
            (np.exp(S) / tau ** 2 - tau) - v_neg * (np.exp(S_neg) /
            tau_neg ** 2 - tau_neg)) + 0.1e1 / np.exp(S) * tau / kappa *
            (-spd * np.exp(S) / tau + (v_neg - spd * (tau - tau_neg)) *
            np.exp(S) / tau ** 2)) ]
            ])
    return out

def error_check_stable_model(p,s):
    """
    This function error checks our model, looking for values which do not make
    physical sense.
    """
    x = np.linspace(s.L,s.R,200)
    y = np.zeros((len(x),2))
    for j in range(len(x)):
        temp = stablab.stablab.soln(x[j],s)
        y[j,0] = temp[0]
        y[j,1] = temp[1]

    # wave speed
    spd = p.spd

    # values at - infinity
    v_neg = p.v_neg
    tau_neg = p.tau_neg
    S_neg = p.S_neg

    # parameters
    m1 = p.m1
    none = p.none
    mu = p.mu
    kappa = p.kappa
    r = np.zeros((len(x),1))

    for j in range(len(x)):
        tau = y[j,0]
        S = y[j,1]
        v = v_neg - spd * (tau - tau_neg)
        r[j,0] = v
        energy = np.exp(S) / tau + tau ** 2 / 0.2e1
        if energy < 0:
            raise ValueError('unphysical, energy should be non-negative')

        pressure = np.exp(S) / tau ** 2 - tau
        if pressure < 0:
            raise ValueError('unphysical, pressure should be non-negative')

        temperature = np.exp(S) / tau
        if temperature < 0:
            raise ValueError('unphysical, temperature should be non-negative')

def plot_profile_stable_model(p,s):
    """
    Plots the profile equations for the stable model.
    """
    x = np.linspace(s.L,s.R,200)
    y = np.zeros((len(x),2))
    for j in range(len(x)):
        temp = stablab.soln(x[j],s)
        y[j,0] = temp[0]
        y[j,1] = temp[1]

    # wave speed
    spd = p.spd

    # values at - infinity
    v_neg = p.v_neg
    tau_neg = p.tau_neg
    S_neg = p.S_neg

    # parameters
    m1 = p.m1
    none = p.none
    mu = p.mu
    kappa = p.kappa
    r = np.zeros((len(x),1))
    w = np.zeros((len(x),1))

    for j in range(len(x)):
        tau = y[j,0]
        S = y[j,1]
        v = v_neg - spd * (tau - tau_neg)
        r[j,0] = v
        T = np.exp(S) / tau
        w[j,0] = T

    fig = plt.figure("Profile Equations")
    plt.plot(x,r)
    plt.plot(x,y)
    plt.plot(x,w)
    plt.xlabel('x')
    plt.ylabel('profiles')
    plt.legend(['v','tau','S','T'])
    plt.show()

def get_prof(ode,jacobian,restpnt,scl,time,p,s):
    # pre_init=@(x,y)(init2(x,y,box,ode,s,p));
    print("entering get_prof\n")
    def plot_this(ynot,tspan,ode,rp):
        sol = stablab.Struct()
        sol.t = []
        sol.y = []
        def updateSol(tVal,yVals):
            sol.t.append(tVal)
            sol.y.append(yVals)
            return None

        integrator = complex_ode(ode)
        integrator.set_integrator('dopri5',atol=1e-10,rtol=1e-10)
        integrator.set_solout(updateSol)
        integrator.set_initial_value(ynot,tspan[0])
        integrator.integrate(tspan[-1])
        sol.t, sol.y = np.array(sol.t), np.array(sol.y)
        return sol

    def pre_init(x,y): return ode(x,y,s,p)

    linearization = jacobian([restpnt[0,0],restpnt[0,1]],p)
    eigenval, eigenvec = np.linalg.eig(linearization)

    tspan = np.array([0,-time])

    ynot = np.real(-scl*eigenvec[:,0]+restpnt[0,:].T)
    sol = plot_this(ynot,tspan,pre_init,restpnt[1,:])

    diffR = sol.y[0] - np.array([p.tau_plus,p.S_plus])
    diffL = sol.y[-1] - np.array([p.tau_neg,p.S_neg])
    print("diffR:  ",diffR,"\ndiffL:  ",diffL,'\n')

    ind = 0
    minval = 1e10
    tend = 0
    for i,yVal in enumerate(sol.y):
        if abs(yVal[1]-0.5*(p.T_neg+p.T_plus)) < minval:
            ind = i
            minval = abs(yVal[1]-0.5*(p.T_neg+p.T_plus))
        if abs(yVal[0]-p.v_neg) > scl:
            tend = sol.t[i]

    tspan = [0, tend]
    tspan = tspan - np.array([sol.t[ind],sol.t[ind]])


    ynot = np.real(-scl*eigenvec[:,0]+restpnt[0,:].T)

    sol = plot_this(ynot,tspan,pre_init,restpnt[1,:])

    S = sol.y[:,1]
    tau = (sol.y[:,0]-p.spd)/p.m1
    T = np.exp(S)/tau

    plt.plot(sol.t,sol.y[:,0])
    plt.plot(sol.t,sol.y[:,1])
    plt.plot(sol.t,tau)
    plt.plot(sol.t,T)
    plt.legend(['v','S','tau','T'])
    plt.xlabel('x')
    plt.show()
    return sol

def A_stable_model_balflux(x,lamda,s,p):
    # Balanced flux Evans matrix for the system named stable_model.
    #print("entering A_stable_model_balflux")
    # wave speed
    spd = p['spd']

    # parameters
    v_neg = p['v_neg']
    tau_neg = p['tau_neg']
    S_neg = p['S_neg']
    m1 = p['m1']
    none = p['none']
    mu = p['mu']
    kappa = p['kappa']

    if s['solve'] == 'bvp':
        # interpolate profile solution
    	temp = stablab.soln(x,s)
    else:
        temp = stablab.deval(x,s.sol)

    #profile values
    tau = temp[0]
    S = temp[1]

    ynot = np.array([tau,S])

    temp2 = F_stable_model(x,ynot,s,p)

    tau_x = temp2[0]
    S_x = temp2[1]

    v = v_neg - spd * (tau - tau_neg)
    v_x = -spd * tau_x
    T_x = S_x * np.exp(S) / tau - np.exp(S) / tau ** 2 * tau_x

    # Evans matrix -- 5 x 5
    out =  np.array([
            [lamda / spd, 0, 0, -0.1e1 / spd, 0],
            [0, 0, 0, 1, 0],
            [lamda * (-0.2e1 * np.exp(S) / tau ** 2 + tau) / spd, 0, 0, -(-0.2e1 *
                np.exp(S) / tau ** 2 + tau) / spd + v, 1 ],
            [lamda * tau / mu * (v_x * mu / tau ** 2 - 0.3e1 * np.exp(S) / tau **
                3 - 0.1e1) / spd, lamda * tau / mu, 0, -tau / mu * ((v_x * mu /
                tau ** 2 - 0.3e1 * np.exp(S) / tau ** 3 - 0.1e1) / spd + spd),
                0.1e1 / mu],
            [lamda * (-v * tau / kappa * (v_x * mu / tau ** 2 - 0.3e1 *
                np.exp(S) / tau ** 3 - 0.1e1) + tau / kappa * (-spd * (-0.2e1 *
                np.exp(S) / tau ** 2 + tau) + v_x * mu * v / tau ** 2 + T_x *
                kappa / tau ** 2 + v * (-0.3e1 * np.exp(S) / tau ** 3 - 0.1e1))) /
                spd, -lamda * v * tau / kappa, lamda * tau / kappa, v * tau /
                kappa * ((v_x * mu / tau ** 2 - 0.3e1 * np.exp(S) / tau ** 3 -
                0.1e1) / spd + spd) - tau / kappa * ((-spd * (-0.2e1 * np.exp(S) /
                tau ** 2 + tau) + v_x * mu * v / tau ** 2 + T_x * kappa /
                tau ** 2 + v * (-0.3e1 * np.exp(S) / tau ** 3 - 0.1e1)) / spd +
                spd * v + v_x * mu / tau - np.exp(S) / tau ** 2 + tau), -v /
                kappa - tau / kappa * (spd - v / tau)]
            ])
    return out

if __name__ == "__main__":
    # parameters
    s = stablab.Struct()
    p = stablab.Struct()
    #s.solve = 'ode'
    s.solve = 'bvp'

    p.S_neg = -5
    p.none = 1
    p.mu = 1
    p.kappa = 1
    p.S0 = 0
    p.tau0 = 1

    # plus and minus infinity endstates
    p = RH_stable_model(p)

    # phase condition
    s.phase = np.array([0.5*(p.tau_plus+p.tau_neg),0.5*(p.S_plus+p.S_neg)])

    # order in which to apply phase conditions
    s.order = np.array([0,1])

    # profile ode function
    s.F = F_stable_model

    # Jacobian function
    s.Flinear = Flinear_stable_model

    # number of profile equations to integrate
    s.n = 2

    # end states
    s.UL = np.array([p.tau_neg, p.S_neg])
    s.UR = np.array([p.tau_plus, p.S_plus])

    s.stats = 'off'

    if s.solve == 'bvp':
        #tolerance at end states
        s.tol = 1e-6
        [p,s] = stablab.profile_flux(p,s)
        # error_check_stable_model(p,s)
        plot_profile_stable_model(p,s)

    elif s.solve == 'ode':
        s.tol = 1e-8
        s.sol = get_prof(s.F,s.Flinear,np.array([[p.tau_plus,p.S_plus],
                         [p.tau_neg,p.S_neg]]),s.tol,time,p,s)
        s.I = 1
        s.L = s.sol.t[0]
        s.R = s.sol.t[-1]
        if s.L > s.R:
            temp  = s.R
            s.R = s.L
            s.L = temp

    # Evans matrix
    Amat = A_stable_model_balflux

    # structure variables
    s,e,m,c = stablab.emcset(s,'front',stablab.evans.LdimRdim(Amat,s,p),
                                'default',Amat)
    s.L_max = 10000
    s.R_max = 10000

    # refine the Evans function computation to achieve set relative error
    c.refine = 'on'

    # display a waitbar
    c.stats = 'on'

    #c.max_R = 1000
    #c.Rtol = 0.2
    #R = stablab.evans.radius(c,s,p,m,e)
    #print("\nRadius: ", R)
    R = 32
    p.R = R

    m.method = stablab.evans.drury_no_radial

    circpnts = 20;  imagpnts = 20;  spread = 4;  zerodist = 10**(-6);
    preimage = stablab.semicirc(circpnts,imagpnts,c.ksteps,R,spread,zerodist)

    # compute Evans function
    c.refine = 'off'
    halfw, domain = stablab.Evans_compute(preimage,c,s,p,m,e)
    w = halfw / halfw[0]
    w = np.concatenate((w,np.flipud(np.conj(w))))

    # plot the Evans function
    stablab.Evans_plot(w)
