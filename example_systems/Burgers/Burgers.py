import numpy as np
import matplotlib.pyplot as plt
from stablab import semicirc2,winding_number,Evans_plot,Evans_compute,emcset,reflect_image

def A(x, lamda, s, p):
    # Evans matrix for Burgers system in unintegrated coordinates.
    a = .5*(p['ul']-p['ur'])
    cc = .5*(p['ul']+p['ur']) # Wave speed
    u = cc - a*np.tanh(a*x/2) # Profile
    uderiv = (-a**2/2)*(1/np.cosh(a*x/2)**2)

    if p["integrated"] == "on":
        out = np.array([[0,   1],
                        [lamda, u - cc]])
    else:
        out = np.array([[0,   1],
                        [lamda + uderiv, u - cc]])
    return out

if __name__ == "__main__":
    ul = 1
    ur = 0
    I = 12

    #  parameters
    p = {'ul':ul,'ur':ur,'integrated': 'off'}

    #  numerical infinity
    s = {'I':I, 'R':I, 'L':-I}

    # set STABLAB structures to local default values
    # default for Burgers is reg_reg_polar
    s['A'] = A
    s, e, m, c = emcset(s,'front',[1,1],'default')

    # Create the Preimage Contour
    circpnts, imagpnts, innerpnts = 20, 20, 5
    r = 10
    spread = 4
    zerodist = 10**(-2)
    # Can set custom ksteps and lambda_steps here, if desired
    # ksteps, lambda_steps = 32, 0
    preimage = semicirc2(circpnts, imagpnts, innerpnts, c['ksteps'],
                       r,spread,zerodist,c['lambda_steps'])
    # plot the preimage
    plt.title("Preimage")
    plt.plot(np.real(preimage), np.imag(preimage))
    plt.show()

    # Compute the Evans function
    halfw, domain = Evans_compute(preimage,c,s,p,m,e)
    # Normalize the solution
    halfw = halfw/halfw[0]
    w = reflect_image(halfw)

    windnum = winding_number(w)

    Evans_plot(w)
    plt.show()
