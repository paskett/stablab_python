"""Hill's method. This method is based on the method given by Deconinck,
Kiyak, Carter, and Kutz (University of Washington, Seattle University) in
their software package SpectrUW (pronounced spectrum).
This is a numerical method that determines the spectra of a linear operator.
For more information, see:
    https://www.sciencedirect.com/science/article/pii/S0378475406002709

Authors: Taylor Paskett, Blake Barker

NOTE: This
"""

import stablab
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import scipy.integrate
from itertools import cycle

def get_chebyshev_coefficients(f,a,b,kind=1):
    """Obtain the chebyshev coefficients for the chebyshev polynomial
    which interpolates the data f on the chebyshev nodes in the interval
    [a,b].
    Parameters:
        f (ndarray): a 1-dimensional ndarray with the y-values of each
                     chebyshev node on the interval [a,b]
        a (int): the left endpoint of the interval for the interpolant
        b (int): the right endpoint of the interval for the interpolant
    """
    if len(np.shape(f)) > 1:
       raise ValueError('input f must be a 1-dimensional ndarray')

    if kind == 1:
        N = np.shape(f)[0]
        Id2 = (2/N)*np.eye(N)
        Id2[0,0] = Id2[0,0]/2
        theta = (np.arange(N)+0.5)*np.pi/N
        Tcf = Id2 @ np.cos(np.outer(theta, np.arange(N))).T
        cf = Tcf @ f
        T_x = (2/(b-a)) * (np.tile(np.arange(N),(N,1))
            * np.sin(np.outer(theta, np.arange(N)))
            / np.sin(np.outer(theta, np.ones(N))))
        fx = T_x @ cf
        cf_x = Tcf @ fx
        fun = lambda x: eval_cf(x,cf,cf_x,a,b)

    else:
        raise NotImplementedError("get_chebyshev_coefficients currently only "
        "works with chebyshev polynomials of the 1st kind (kind == 1).")

    return cf, fun

def eval_cf(x,cf,cf_x,a_x,b_x):
    """Transformation to get Chebyshev coefficients"""
    N = np.shape(cf)[0]
    xtilde = (x-0.5*(a_x+b_x))/(0.5*(b_x-a_x))
    theta = np.arccos(xtilde)
    T = np.cos(np.outer(theta, np.arange(N)))
    out1 = T @ cf
    out2 = (T @ cf_x).T
    return out1, out2

def hill_method(N,kappa,p,sol):
    """Executes Hill's method. Five steps are followed to solve the eigenvalue
    problem:
    1 - Determine the Fourier coefficients
    2 - Represent the eigenfunctions using Floquet theory
    3 - Construct the bi-infinite Floquet-Fourier difference equation
    4 - Truncate the difference equation
    5 - Determine the eigenvalues
    """
    M = hill_coef(p,sol,N)
    X = sol['x'][0,0][0,0]
    sx,sy = np.shape(M)
    lamda = np.zeros((2*N+1,len(kappa)),dtype=np.complex)

    L = np.zeros((2*N+1,2*N+1),dtype=np.complex)

    for j in range(len(kappa)):
        for n in range(-N,N+1):
            for m in range(-N,N+1):
                if (n-m) % 2 == 0:
                    temp = 0
                    for k in range(sx):
                        temp += M[k,(n-m)//2+N]*(1j*(kappa[j]+np.pi*m/X))**k
                    L[n+N,m+N] = temp
        lamda[:,j] = np.linalg.eigvals(L)

    return lamda

def complex_quad(fun,a,b,**kwargs):
    """A wrapper to scipy.integrate.quad which separately integrates the
    real and imaginary parts of the function fun, then puts them together.
    """
    real_integral = scipy.integrate.quad(lambda x: scipy.real(fun(x)), a, b,
                                                                    **kwargs)
    imag_integral = scipy.integrate.quad(lambda x: scipy.imag(fun(x)), a, b,
                                                                    **kwargs)
    # Returns the complex value and the error bounds for both calculations
    return (real_integral[0] + 1j*imag_integral[0],
            real_integral[1:],
            imag_integral[1:] )

def quadv(fun,a,b,**kwargs):
    """Uses scipy.integrate.quad on a function fun which returns a vector.
    Normally, quad can only be used on a scalar function. Essentially,
    quadv vectorizes quad. It also assumes that fun returns complex values.
    """
    # Sorry, the following line is basically unreadable. I'm creating a list
    #  of n functions where each function calls fun, then returns the ith
    #  value in the vector returned by fun. This way, we can use a map to call
    #  scipy's quad on each of the values in the vector returned by fun.
    func_inpts = [(lambda y: (lambda x: fun(x)[y]))(i) for i in
                                                range(np.shape(fun([1]))[0])]
    integrals = map(complex_quad, func_inpts, cycle([a]), cycle([b]))
    out = np.array([x[0] for x in integrals])
    return out


def hill_coef(p,sol,N):
    """
    """
    # index
    out = np.zeros((5,2*N+1),dtype=np.complex)

    out[2,N] = -p['delta'][0,0][0,0]
    out[4,N] = -p['delta'][0,0][0,0]
    out[3,N] = -p['epsilon'][0,0][0,0]

    def fun(x):
        return (np.exp((-2*np.pi*1j*np.arange(-N,N+1)
                    / sol['x'][0,0][0,0]).T*x) * deval_local(sol,x))

    Q = quadv(fun,0,sol['x'][0,0][0,0])

    out[1,:] = -Q.T/sol['x'][0,0][0,0]
    out[0,:] = ( -(2*np.pi*1j*np.arange(-N,N+1)/sol['x'][0,0][0,0])*Q.T
                     / sol['x'][0,0][0,0] )
    return out


def deval_local(sol,x):
    a, b = sol[0,0]['a'][0,0], sol[0,0]['b'][0,0]
    coeff, fun = get_chebyshev_coefficients(sol[0,0]['f'][0], a,b)
    out = eval_cf(x, coeff, coeff, a, b)[0]
    return out

if __name__ == "__main__":
    # driver for Hill's method
    # parameters
    N = 20
    num_floquet = 31
    epsilon = 0.9
    num = 84 - 1

    file_name = 'dep'+str(int(10**6*epsilon))+'.mat'

    ld = scipy.io.loadmat(file_name)
    dd = ld['rundata']
    dd_num = dd[0,num]
    p = dd_num[0,0]
    s = dd_num[0,3]['sol'][0,0]

    ld = scipy.io.loadmat('d_profile')

    d = ld['d']
    profile_fun = []
    for j in range(3): # 1:3
        cf,fun = get_chebyshev_coefficients(d['Y'][0,0][j],
                                    d['a'][0,0][0,0], d['b'][0,0][0,0])
        profile_fun.append(fun)


    dom = np.linspace(s['x'][0,0][0,0],s['x'][0,0][0,-1],1000)
    plt.figure()
    for j in range(3): # 1:3
        plt.plot(dom,profile_fun[j](dom)[0],'-k',markersize=.25)
    plt.show()

    plt.figure()
    kappa = np.linspace(-np.pi/(2*p['X'][0,0][0,0]),
                        np.pi/(2*p['X'][0,0][0,0]), 201)
    tc = dd_num[0,1]['dtaylor'][0,0][0,0]
    alpha1 = tc['alpha1'][0,0]
    alpha2 = tc['alpha2'][0,0]
    beta1 = tc['beta1'][0,0]
    beta2 = tc['beta2'][0,0]

    lam1 = alpha1*kappa + beta1*kappa**2
    lam2 = alpha2*kappa + beta2*kappa**2

    plt.plot(lam1.real, lam1.imag, '-b')
    plt.plot(lam2.real, lam2.imag, '-b')
    plt.show()

    s = scipy.io.loadmat("specific_wave.mat")
    s = s['d']
    lamda = hill_method(N, kappa, p, s)

    plt.plot(lamda.real, lamda.imag, '.')
    plt.plot([0,0], [-5,5], '.k')
    plt.plot(lamda[:,(len(kappa)-1)//2].real, lamda[:,(len(kappa)-1)//2].imag,'.m')
    plt.axis([-0.1,0.1,-0.3,0.3])
