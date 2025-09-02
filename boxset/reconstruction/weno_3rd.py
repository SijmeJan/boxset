import numpy as np
from numba import jit_module

# Maximum order 2*r-1
weno_r = 2

def interpolate_to_edge(U, X):
    '''
    Last dimension of U should have 3 elements. X is the evaluation point: X=2 for x_{i+1/2}.
    Returns estimates for U using the left- and right-biased stencils.
    '''

    UL = U[...,1] + U[...,1]*(X - 1.5) - U[...,0]*(X - 1.5)
    UR = U[...,1] + U[...,2]*(X - 1.5) - U[...,1]*(X - 1.5)

    return UL, UR

def smoothness_coefficients(U):
    '''
    Last dimension of U should have 3 elements.
    Returns the smoothness coefficients for the left and right-biased stencils.
    '''

    betaL = (U[...,0] - U[...,1])**2
    betaR = (U[...,2] - U[...,1])**2

    return betaL, betaR

def nonlinear_weights(U, epsilon):
    '''
    Calculate nonlinear weights based on the smoothness of the solution.
    Last dimension of U should have 5 elements.
    epsilon is a small number (might need to be tuned).
    Returns: nonlinear weights.
    '''
    betaL, betaR = smoothness_coefficients(U)

    # JS-type weights
    wL = (1/(epsilon + betaL)**2)/3
    wR = (2/(epsilon + betaR)**2)/3

    # Z-type weights
    tau = np.abs(betaL - betaR)
    wL = wL*((epsilon + betaL)**2 + tau**2)
    wR = wR*((epsilon + betaR)**2 + tau**2)

    # Alternative (better near critical points)
    #wL = (1 + tau/(betaL + epsilon)**(0.75))/3
    #wR = 2*(1 + tau/(betaR + epsilon)**(0.75))/3

    norm_fac = 1.0/(wL + wR)


    return wL*norm_fac, wR*norm_fac

def calc_interface_flux(U, X, epsilon=1.0e-12):
    '''
    Calculate 3rd-order WENO approximation at position X.
    Last dimension of U should have 3 elements
    X=0 is the leftmost cell edge, X=3 the rightmost cell edge.
    Returns: approximation of U at X.
    '''

    # Calculate the ENO approximation for the 2 different 2-point stencils
    uL, uR = interpolate_to_edge(U, X)

    # Calculate the nonlinear weights based on the smoothness
    wL, wR = nonlinear_weights(U, epsilon)

    # Return final approximation
    return wL*uL + wR*uR

jit_module(nopython=True, error_model="numpy")