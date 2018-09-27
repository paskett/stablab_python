import numpy as np
import time

from stablab import Struct
from stablab.evans import power_expansion2

def  ks_taylor(s,p,m,c,e,d):
        # [p,s,d,st] = ks_taylor(s,p,m,c,e)
        #
        # Determine the Taylor expansion coefficients of the spectal curves
        # near the origin for the KS system.

        e.evans = 'balanced_polar_periodic'

        m['k_int_options'] = {'AbsTol': 10**(-10), 'RelTol': 10**(-8)}
        m['lambda_int_options'] = {'AbsTol': 10**(-8), 'RelTol': 10**(-6)}

        #Initialize the dictionary, st
        st = Struct()
        st['k_int_options'] = m['k_int_options']
        st['lambda_int_options'] = m['lambda_int_options']

        R_lambda = 0.01
        k_radii = np.array([0.001, 0.001, 0.001, 0.001])
        k_powers = np.array([0, 1, 2, 3])
        lambda_powers = np.array([0, 1, 2, 3])

        st['R_lambda'] = R_lambda
        st['k_radii'] = k_radii

        s_time = time.time()
        out = power_expansion2(R_lambda,k_radii,lambda_powers,k_powers,s,p,m,c,e)
        st['time'] = time.time() - s_time

        coef = np.transpose(out)

        aa = coef[2,0]
        bb = coef[1,1]
        cc = coef[0,2]
        dd = coef[3,0]
        ee = coef[2,1]
        ff = coef[1,2]
        gg = coef[0,3]

        alpha1 = (-bb+np.sqrt(bb**2-4*aa*cc))/(2*aa)
        alpha2 = (-bb-np.sqrt(bb**2-4*aa*cc))/(2*aa)
        beta1 = -(dd*alpha1**3+alpha1**2*ee+alpha1*ff+gg)/(2*aa*alpha1+bb)
        beta2 = -(dd*alpha2**3+alpha2**2*ee+alpha2*ff+gg)/(2*aa*alpha2+bb)

        d['a'] = aa
        d['b'] = bb
        d['c'] = cc
        d['d'] = dd
        d['e'] = ee
        d['f'] = ff
        d['g'] = gg
        d['alpha1'] = alpha1
        d['alpha2'] = alpha2
        d['beta1'] = beta1
        d['beta2'] = beta2

        return p,s,d,st
