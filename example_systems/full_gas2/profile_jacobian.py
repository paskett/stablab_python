import numpy as np

def profile_jacobian(y, p):
    v = y[0]
    e = y[1]
    J = np.array([[(1 / p['mu']) * (2 * v - 1 - p['Gamma'] * p['e_minus']), p['Gamma'] / p['mu']],
                  [(1 / p['nu']) * (-((v - 1) ** 2) / 2 + e - p['e_minus'] + (v - 1) * p['Gamma'] * e) + (
                              v / p['nu']) * (-(v - 1) + p['Gamma'] * p['e_minus']), v / p['nu']]])
    return J

