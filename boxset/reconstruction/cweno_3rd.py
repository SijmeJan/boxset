import numpy as np
from numba import jit_module

# Maximum order 2*r-1
weno_r = 3

def interpolate_to_edge(U, X):
    '''
    Last dimension of U should have 5 elements. X is the evaluation point: X=3 for x_{i+1/2}.
    Returns estimates for U using the three-point left-, centre- and right-biased stencils.
    '''
    # | i-2 | i-1 | i | i+1 | i+2 |
    # This is where we calculate PL, PC, PR

    u   = U[..., 1] - (U[..., 0] - 2*U[..., 1] + U[..., 2])/24
    ux  = 0.5*(U[...,2] - U[...,0])
    uxx = U[...,2] - 2*U[...,1] + U[...,0]
    PL = u + ux*(X - 1.5) + 0.5*uxx*(X - 1.5)**2

    u   = U[..., 2] - (U[..., 1] - 2*U[..., 2] + U[..., 3])/24
    ux  = 0.5*(U[...,3] - U[...,1])
    uxx = U[...,3] - 2*U[...,2] + U[...,1]
    PC = u + ux*(X - 2.5) + 0.5*uxx*(X - 2.5)**2

    u   = U[..., 3] - (U[..., 2] - 2*U[..., 3] + U[..., 4])/24
    ux  = 0.5*(U[...,4] - U[...,2])
    uxx = U[...,4] - 2*U[...,3] + U[...,2]
    PR = u + ux*(X - 3.5) + 0.5*uxx*(X - 3.5)**2

    return PL, PC, PR

def smoothness_coefficients(U):
    '''
    Last dimension of U should have five elements.
    Returns the smoothness coefficients for the left, centre and right-biased stencils.
    '''
    betaL = 13*(U[...,0] - 2*U[...,1] + U[...,2])**2/12 + 0.25*(U[...,0] - 4*U[...,1] + 3*U[...,2])**2
    betaC = 13*(U[...,1] - 2*U[...,2] + U[...,3])**2/12 + 0.25*(U[...,1] - U[...,3])**2
    betaR = 13*(U[...,2] - 2*U[...,3] + U[...,4])**2/12 + 0.25*(3*U[...,2] - 4*U[...,3] + U[...,4])**2

    return betaL, betaC, betaR

def nonlinear_weights(U, epsilon):
    '''
    Calculate nonlinear weights based on the smoothness of the solution.
    Last dimension of U should have 5 elements.
    epsilon is a small number (might need to be tuned).
    Returns: nonlinear weights.
    '''
    betaL, betaC, betaR = smoothness_coefficients(U)

    # JS-type weights
    wL = 3/16/(epsilon + betaL)**2
    wC = 5/8/(epsilon + betaC)**2
    wR = 3/16/(epsilon + betaR)**2

    norm_fac = 1.0/(wL + wC + wR)

    return wL*norm_fac, wC*norm_fac, wR*norm_fac

def calc_interface_flux(U, X, epsilon=1.0e-12):
    '''
    Calculate 3rd-order CWENO approximation at position X.
    Last dimension of U should have 5 elements
    X=0 is the leftmost cell edge, X=5 the rightmost cell edge.
    Returns: approximation of U at X.
    '''

    # Calculate the ENO approximation for the 3 different 3-point stencils
    uL, uC, uR = interpolate_to_edge(U, X)

    # Calculate the nonlinear weights based on the smoothness
    wL, wC, wR = nonlinear_weights(U, epsilon)

    # Return final approximation
    return wL*uL + wC*uC + wR*uR

jit_module(nopython=True, error_model="numpy")