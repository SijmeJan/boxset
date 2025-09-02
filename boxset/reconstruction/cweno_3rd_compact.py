import numpy as np
from numba import jit_module

# Maximum order = stencil size = 2*r-1
weno_r = 2

# For CWENO, any symmetric combination will do.
ideal_weights = np.asarray([0.25, 0.5, 0.25])

def interpolate_to_edge(U, X):
    '''
    Last dimension of U should have 3 elements. X is the evaluation point: X=2 for x_{i+1/2}.
    Returns estimates for U using the left- and right-biased stencils, as well as central
    '''

    UL = U[...,1] + U[...,1]*(X - 1.5) - U[...,0]*(X - 1.5)
    UR = U[...,1] + U[...,2]*(X - 1.5) - U[...,1]*(X - 1.5)

    uxx = U[...,2] - 2*U[...,1] + U[...,0]
    ux = 0.5*(U[...,2] - U[...,0])
    u = U[...,1] - uxx/24

    UC = (u + ux*(X - 1.5) + 0.5*uxx*(X - 1.5)**2 - ideal_weights[0]*UL - ideal_weights[2]*UR)/ideal_weights[1]

    return UL, UR, UC

def smoothness_coefficients(U):
    '''
    Last dimension of U should have 3 elements.
    Returns the smoothness coefficients for the left and right-biased stencils,as well as central
    '''

    betaL = (U[...,0] - U[...,1])**2
    betaR = (U[...,2] - U[...,1])**2
    betaC = 13*(U[...,2] - 2*U[...,1] + U[...,0])**2/3 + 0.25*(U[...,2] - U[...,0])**2

    return betaL, betaR, betaC


def nonlinear_weights(U, epsilon):
    '''
    Calculate nonlinear weights based on the smoothness of the solution.
    Last dimension of U should have 5 elements.
    epsilon is a small number (might need to be tuned).
    Returns: nonlinear weights.
    '''
    betaL, betaR, betaC = smoothness_coefficients(U)

    # JS-type weights
    wL = ideal_weights[0]/(epsilon + betaL)**2
    wC = ideal_weights[1]/(epsilon + betaC)**2
    wR = ideal_weights[2]/(epsilon + betaR)**2

    norm_fac = 1.0/(wL + wC + wR)

    return wL*norm_fac, wC*norm_fac, wR*norm_fac

def calc_interface_flux(U, X, epsilon=1.0e-12):
    '''
    Calculate 3rd-order CWENO approximation at position X.
    Last dimension of U should have 3 elements
    X=0 is the leftmost cell edge, X=3 the rightmost cell edge.
    Returns: approximation of U at X.
    '''

    # Calculate the ENO approximation for the 3 different 3-point stencils
    uL, uR, uC = interpolate_to_edge(U, X)

    # Calculate the nonlinear weights based on the smoothness
    wL, wC, wR = nonlinear_weights(U, epsilon)

    # Return final approximation
    return wL*uL + wC*uC + wR*uR

jit_module(nopython=True, error_model="numpy")