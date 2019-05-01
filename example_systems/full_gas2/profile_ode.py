import numpy as np

def profile_ode(x, y, sol, p):
    v = y[0]
    e = y[1]

    out = np.vstack([(1 / p['mu']) * (v * (v - 1) + p['Gamma'] * (e - v * p['e_minus'])),
                   (v / p['nu']) * (-((v - 1) ** 2) / 2 + e - p['e_minus'] + (v - 1) * p['Gamma'] * p['e_minus'])])

    return out