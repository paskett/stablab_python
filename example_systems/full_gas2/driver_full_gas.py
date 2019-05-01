#Import the necessary files.

import numpy as np
import matplotlib.pyplot as plt
import stablab
from stablab.get_profile import get_profile
from parameters import getParams
from profile_jacobian import profile_jacobian
from profile_ode import profile_ode
from profile_bc import profile_bc


if __name__ == "__main__":

    # Get the parameters for the system.
    params = getParams()
    sol = {
        'dim': 2,
        'F': profile_ode,  # F is the profile ode
        'Flinear': profile_jacobian,  # J is the profile ode Jacobian
        'UL': np.array([params['v_minus'], params['e_minus']]),
        'UR': np.array([params['v_plus'], params['e_plus']]),
        'tol': 1e-7,
        'BC': profile_bc,
        'BClen': 2}

    # Get the profile for the system.
    [s,p] = get_profile(params, sol, tol=1e-8, num_inf=2, timeout=50)

    #Setup the domain and the mapped range.
    x = np.linspace(s['L'],s['R'],2000)
    y = s['sol']['y'] # y = s['sol']['u']
    y, yDeriv = s['sol']['deval'](x)

    # Plot the results.
    plt.plot(x, y[0], label='Full gas profile (y[0])')
    plt.plot(x,y[1], label='Full gas profile (y[1])')
    # plt.plot(x, yDeriv, label='Full gas derivative')
    plt.legend()
    plt.show()
