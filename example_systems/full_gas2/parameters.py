

def getParams():

    # Independent parameters.
    p = {'Gamma': 2 / 3,
         'v_plus': 0.7,
         'mu': 1,
         'cnu': 1,
         'kappa': 1}

    # Dependent parameters.
    p.update({
        'v_star': p['Gamma'] / (p['Gamma'] + 2)  # correct see 2.33
    })
    p.update({
        'e_plus': p['v_plus'] * (p['Gamma'] + 2 - p['Gamma'] * p['v_plus']) / (2 * p['Gamma'] * (p['Gamma'] + 1)),
    # 2.35
        'e_minus': (p['Gamma'] + 2) * (p['v_plus'] - p['v_star']) / (2 * p['Gamma'] * (p['Gamma'] + 1)),  # 2.34
        'v_minus': 1,
        'nu': p['kappa'] / p['cnu']  # see below 2.25
    })
    return p
