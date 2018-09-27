import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import complex_ode
from stablab import (semicirc, winding_number, Evans_plot, emcset,
                       Evans_compute, soln, profile_flux, Struct)
from stablab.evans import drury_no_radial, LdimRdim, reflect_image


def RH_local_model(p):
    """
    Rankine Hugoniot conditions
    """
    st = Struct()
    st.H_fun = H_local_model

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

def H_local_model(p,tau,S):
    """
    Hugoniot curve equation
    """

    S0 = p.S0
    tau0 = p.tau0
    none = p.none
    mu = p.mu
    kappa = p.kappa

    out = (np.exp(S) / tau + S + tau ** 2 / 0.2e1 - np.exp(S0) / tau0 - S0
            - tau0 ** 2 / 0.2e1 + (np.exp(S) / tau ** 2 - tau + np.exp(S0)
            / tau0 ** 2 - tau0) * (tau - tau0) / 0.2e1)
    return out

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

def error_check_local_model(p,s):
    """
    This function error checks our model, looking for values which do not make
    physical sense.
    """
    x = np.linspace(s.L,s.R,200)
    y = np.zeros((len(x),2))
    for j in range(len(x)):
        temp = soln(x[j],s)
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

def F_local_model(x,y,s,p):
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

    v = v_neg - spd * (tau - tau_neg)

    # profile equations
    out = np.array([
        -tau / spd / mu * (-spd * (v - v_neg) + np.exp(S) / tau ** 2 - tau
            - np.exp(S_neg) / tau_neg ** 2 + tau_neg) ,
        tau * ((-0.1e1 / tau / spd / mu - 0.1e1 / np.exp(S) * tau * v / kappa)
            * (-spd * (v - v_neg) + np.exp(S) / tau ** 2 - tau - np.exp(S_neg)
            / tau_neg ** 2 + tau_neg) + 0.1e1 / np.exp(S) * tau / kappa * (-spd
            * (np.exp(S) / tau + S + tau ** 2 / 0.2e1 + v ** 2 / 0.2e1
            - np.exp(S_neg) / tau_neg - S_neg - tau_neg ** 2 / 0.2e1
            - v_neg ** 2 / 0.2e1) + v * (np.exp(S) / tau ** 2 - tau) - v_neg
            * (np.exp(S_neg) / tau_neg ** 2 - tau_neg)))
        ])

    return out

def Flinear_local_model(y,p):

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
        [ -0.1e1 / spd / mu * (spd ** 2 * (tau - tau_neg) + np.exp(S)
            / tau ** 2 - tau - np.exp(S_neg) / tau_neg ** 2 + tau_neg)
            - tau / spd / mu * (spd ** 2 - 0.2e1 * np.exp(S) / tau ** 3
            - 0.1e1), -0.1e1 / tau / spd / mu * np.exp(S) ],
        [ (-0.1e1 / tau / spd / mu - 0.1e1 / np.exp(S) * tau * (v_neg - spd
            * (tau - tau_neg)) / kappa) * (spd ** 2 * (tau - tau_neg)
            + np.exp(S) / tau ** 2 - tau - np.exp(S_neg) / tau_neg ** 2
            + tau_neg) + 0.1e1 / np.exp(S) * tau / kappa * (-spd * (np.exp(S)
            / tau + S + tau ** 2 / 0.2e1 + (v_neg - spd * (tau - tau_neg)) ** 2
            / 0.2e1 - np.exp(S_neg) / tau_neg - S_neg - tau_neg ** 2 / 0.2e1
            - v_neg ** 2 / 0.2e1) + (v_neg - spd * (tau - tau_neg))
            * (np.exp(S) / tau ** 2 - tau) - v_neg * (np.exp(S_neg)
            / tau_neg ** 2 - tau_neg)) + tau * ((0.1e1 / tau ** 2 / spd / mu
            - 0.1e1 / np.exp(S) * (v_neg - spd * (tau - tau_neg)) / kappa
            + 0.1e1 / np.exp(S) * tau * spd / kappa) * (spd ** 2 * (tau
            - tau_neg) + np.exp(S) / tau ** 2 - tau - np.exp(S_neg)
            / tau_neg ** 2 + tau_neg) + (-0.1e1 / tau / spd / mu - 0.1e1
            / np.exp(S) * tau * (v_neg - spd * (tau - tau_neg)) / kappa)
            * (spd ** 2 - 0.2e1 * np.exp(S) / tau ** 3 - 0.1e1) + 0.1e1
            / np.exp(S) / kappa * (-spd * (np.exp(S) / tau + S + tau ** 2
            / 0.2e1 + (v_neg - spd * (tau - tau_neg)) ** 2 / 0.2e1
            - np.exp(S_neg) / tau_neg - S_neg - tau_neg ** 2 / 0.2e1
            - v_neg ** 2 / 0.2e1) + (v_neg - spd * (tau - tau_neg))
            * (np.exp(S) / tau ** 2 - tau) - v_neg * (np.exp(S_neg)
            / tau_neg ** 2 - tau_neg)) + 0.1e1 / np.exp(S) * tau / kappa
            * (-spd * (-np.exp(S) / tau ** 2 + tau - (v_neg - spd * (tau
            - tau_neg)) * spd) - spd * (np.exp(S) / tau ** 2 - tau) + (v_neg
            - spd * (tau - tau_neg)) * (-0.2e1 * np.exp(S) / tau ** 3
            - 0.1e1))),
             tau * (0.1e1 / np.exp(S) * tau * (v_neg - spd * (tau - tau_neg))
             / kappa * (spd ** 2 * (tau - tau_neg) + np.exp(S) / tau ** 2 - tau
             - np.exp(S_neg) / tau_neg ** 2 + tau_neg) + (-0.1e1 / tau / spd
             / mu - 0.1e1 / np.exp(S) * tau * (v_neg - spd * (tau - tau_neg))
             / kappa) * np.exp(S) / tau ** 2 - 0.1e1 / np.exp(S) * tau / kappa
             * (-spd * (np.exp(S) / tau + S + tau ** 2 / 0.2e1 + (v_neg - spd
             * (tau - tau_neg)) ** 2 / 0.2e1 - np.exp(S_neg) / tau_neg - S_neg
             - tau_neg ** 2 / 0.2e1 - v_neg ** 2 / 0.2e1) + (v_neg - spd
             * (tau - tau_neg)) * (np.exp(S) / tau ** 2 - tau) - v_neg
             * (np.exp(S_neg) / tau_neg ** 2 - tau_neg)) + 0.1e1 / np.exp(S)
             * tau / kappa * (-spd * (np.exp(S) / tau + 0.1e1) + (v_neg - spd
             * (tau - tau_neg)) * np.exp(S) / tau ** 2)) ]
        ])
    return out

def plot_profile_local_model(p,s):

    x = np.linspace(s.L,s.R,200)
    y = np.zeros((len(x),2))
    for j,xVal in enumerate(x):
        temp = soln(xVal,s)
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
    r = np.zeros(len(x))
    w = np.zeros(len(x))

    for j in range(len(x)):
        tau = y[j,0]
        S = y[j,1]
        v = v_neg - spd * (tau - tau_neg)
        r[j] = v
        T = np.exp(S) / tau + 0.1e1
        w[j] = T

    plt.plot(x,r)
    plt.plot(x,y)
    plt.plot(x,w)
    plt.xlabel('x')
    plt.ylabel('profiles')
    plt.legend(['v','tau','S','T','Location','Best'])
    plt.show()

def A_local_model_balflux(x,lamda,s,p):
    """
    Balanced flux Evans matrix for the system named local_model.
    """
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

    # interpolate profile solution
    temp = soln(x,s)

    #profile values
    tau = temp[0]
    S = temp[1]

    ynot = np.array([tau,S])

    temp2 = F_local_model(x,ynot,s,p);

    tau_x = temp2[0]
    S_x = temp2[1]

    v = v_neg - spd * (tau - tau_neg)
    v_x = -spd * tau_x
    T_x = S_x * np.exp(S) / tau - np.exp(S) / tau ** 2 * tau_x

    # Evans matrix
    out = np.array([
        [ lamda / spd, 0, 0, -0.1e1 / spd, 0 ],
        [ 0, 0, 0, 1, 0 ],
        [ lamda * (-np.exp(S) / tau ** 2 + tau - (np.exp(S) / tau + 0.1e1)
            / tau) / spd, 0, 0, -(-np.exp(S) / tau ** 2 + tau - (np.exp(S)
            / tau + 0.1e1) / tau) / spd + v, (np.exp(S) / tau + 0.1e1)
            / np.exp(S) * tau ],
        [ lamda * tau / mu * (v_x * mu / tau ** 2 - 0.3e1 * np.exp(S)
            / tau ** 3 - 0.1e1) / spd, lamda * tau / mu, 0, -tau / mu * ((v_x
            * mu / tau ** 2 - 0.3e1 * np.exp(S) / tau ** 3 - 0.1e1) / spd
            + spd), 0.1e1 / mu],
        [ lamda * (-v * tau / kappa * (v_x * mu / tau ** 2 - 0.3e1 * np.exp(S)
            / tau ** 3 - 0.1e1) + tau / kappa * (-spd * (-np.exp(S) / tau ** 2
            + tau - (np.exp(S) / tau + 0.1e1) / tau) + v_x * mu * v / tau ** 2
            + T_x * kappa / tau ** 2 + v * (-0.3e1 * np.exp(S) / tau ** 3
            - 0.1e1))) / spd, -lamda * v * tau / kappa, lamda * tau / kappa,
            v * tau / kappa * ((v_x * mu / tau ** 2 - 0.3e1 * np.exp(S)
            / tau ** 3 - 0.1e1) / spd + spd) - tau / kappa * ((-spd
            * (-np.exp(S) / tau ** 2 + tau - (np.exp(S) / tau + 0.1e1) / tau)
            + v_x * mu * v / tau ** 2 + T_x * kappa / tau ** 2 + v * (-0.3e1
            * np.exp(S) / tau ** 3 - 0.1e1)) / spd + spd * v + v_x * mu / tau
            - np.exp(S) / tau ** 2 + tau), -v / kappa - tau / kappa * (spd
            * (np.exp(S) / tau + 0.1e1) / np.exp(S) * tau - v / tau) ]
        ])
    return out


if __name__ == "__main__":
    # driver for flux and balanced flux formulation
    # for full gas in Lagrangian coordinates

    # parameters
    S_neg_vals = [-1]  # -3.3 -3.31 #-(4.4:0.1:5);
    p = Struct()
    s = Struct()

    s_old = None
    for S in S_neg_vals:

        p.S_neg = S

        p.none = 1
        p.mu = 1
        p.kappa = 1
        p.S0 = 0
        p.tau0 = 1

        # plus and minus infinity end states
        p = RH_local_model(p)

        # phase condition
        s.phase = np.array([0.5*(p.tau_plus+p.tau_neg),0.5*(p.S_plus+p.S_neg)])

        #order in which to apply phase conditions
        s.order = np.array([0,1])

        #profile ode
        s.F = F_local_model

        #Jacobian file
        s.Flinear = Flinear_local_model

        #number of profile equations to integrate
        s.n = 2;

        #end states
        s.UL = np.array([p.tau_neg, p.S_neg])
        s.UR = np.array([p.tau_plus, p.S_plus])

        s.stats = 'off'
        #tolerance at end states
        s.tol = 1e-6

        p,s = profile_flux(p,s,s_old)

        s_old = s

    # error_check_local_model(p,s)

    plot_profile_local_model(p,s)

    # Evans matrix
    Amat = A_local_model_balflux

    # structure variables
    s,e,m,c = emcset(s,'front',LdimRdim(Amat,s,p),'default',Amat)

    # refine the Evans function computation to achieve set relative error
    c.refine = 'on'

    # display a waitbar
    c.stats = 'off'

    m.method = drury_no_radial

    R = 10
    p.R = R

    circpnts = 20; imagpnts = 20; spread = 4; zerodist = 10**(-6);
    preimage = semicirc(circpnts,imagpnts,c.ksteps,R,spread,zerodist)

    # circpnts = 20; imagpnts = 20; innerpnts = 7; spread = 4; inner_radius = 10^(-4); lamda_steps = 0;
    # preimage = semicirc2(circpnts,imagpnts,innerpnts,c.ksteps,R,spread,inner_radius,lamda_steps);

    # c.ksteps = 2^12;
    # pnts = 2;
    # preimage = linspace(0.1,1e-4,pnts+(pnts-1)*c.ksteps);

    # compute Evans function
    halfw, domain = Evans_compute(preimage,c,s,p,m,e)
    w = halfw / halfw[0]
    w = reflect_image(w)

    # plot the Evans function
    Evans_plot(w)
