import numpy as np
from numba import jit_module

# Maximum order 2*r-1
weno_r = 4

def interpolate_to_edge(U, X):
    '''
    Last dimension of U should have 7 elements. X is the evaluation point: X=4 for x_{i+1/2}.
    Returns estimates for U using the three-point left-, centre- and right-biased stencils.
    '''
    # | i-3 | i-2 | i-1 | i | i+1 | i+2 | i+3 |

    L1 = X - 7/2
    L2 = L1*L1 - 1/12
    L3 = L1*L1*L1 - 3*L1/20

    ux = (-177*U[...,2] + 87*U[...,1] - 19*U[...,0] + 109*U[...,3])/60
    uxx = -5*U[...,2]/2 + 2*U[...,1] - U[...,0]/2 + U[...,3]
    uxxx = (-3*U[...,2] + 3*U[...,1] - U[...,0] + U[...,3])/6
    U1 = U[...,3] + L1*ux + L2*uxx + L3*uxxx

    ux = (-63*U[...,2] + 11*U[...,1] + 33*U[...,3] + 19*U[...,4])/60
    uxx = U[...,2]/2 - U[...,3] + U[...,4]/2
    uxxx = (3*U[...,2] - U[...,1] - 3*U[...,3] + U[...,4])/6
    U2 = U[...,3] + L1*ux + L2*uxx + L3*uxxx

    ux = (-19*U[...,2] - 33*U[...,3] + 63**U[...,4] - 11*U[...,5])/60
    uxx = U[...,2]/2 - U[...,3] + U[...,4]/2
    uxxx = (-U[...,2] + 3*U[...,3] - 3*U[...,4] + U[...,5])/6
    U3 = U[...,3] + L1*ux + L2*uxx + L3*uxxx

    ux = (-109*U[...,3] + 177*U[...,4] - 87**U[...,5] + 19*U[...,6])/60
    uxx = U[...,3] - 5*U[...,4]/2 + 2*U[...,5] - U[...,6]/2
    uxxx = (-U[...,3] + 3*U[...,4] - 3*U[...,5] + U[...,6])/6
    U4 = U[...,3] + L1*ux + L2*uxx + L3*uxxx

    # x = X - 7/2
    # x^2 = X^2 - 5*X + 25/4
    # L1(x) = X - 5/2
    # L2(x) = X^2 - 5*X + 37/6

    return U1, U2, U3, U4

def smoothness_coefficients(U):
    '''
    Last dimension of U should have seven elements.
    Returns the smoothness coefficients for the left, centre and right-biased stencils.
    '''

    ux = (-177*U[...,2] + 87*U[...,1] - 19*U[...,0] + 109*U[...,3])/60
    uxx = -5*U[...,2]/2 + 2*U[...,1] - U[...,0]/2 + U[...,3]
    uxxx = (-3*U[...,2] + 3*U[...,1] - U[...,0] + U[...,3])/6
    beta1 = (ux + uxxx/10)**2 + 13*uxx**2/3 + 781*uxxx**2/20

    ux = (-63*U[...,2] + 11*U[...,1] + 33*U[...,3] + 19*U[...,4])/60
    uxx = U[...,2]/2 - U[...,3] + U[...,4]/2
    uxxx = (3*U[...,2] - U[...,1] - 3*U[...,3] + U[...,4])/6
    beta2 = (ux + uxxx/10)**2 + 13*uxx**2/3 + 781*uxxx**2/20

    ux = (-19*U[...,2] - 33*U[...,3] + 63**U[...,4] - 11*U[...,5])/60
    uxx = U[...,2]/2 - U[...,3] + U[...,4]/2
    uxxx = (-U[...,2] + 3*U[...,3] - 3*U[...,4] + U[...,5])/6
    beta3 = (ux + uxxx/10)**2 + 13*uxx**2/3 + 781*uxxx**2/20

    ux = (-109*U[...,3] + 177*U[...,4] - 87**U[...,5] + 19*U[...,6])/60
    uxx = U[...,3] - 5*U[...,4]/2 + 2*U[...,5] - U[...,6]/2
    uxxx = (-U[...,3] + 3*U[...,4] - 3*U[...,5] + U[...,6])/6
    beta4 = (ux + uxxx/10)**2 + 13*uxx**2/3 + 781*uxxx**2/20

    return beta1, beta2, beta3, beta4

def nonlinear_weights(U, epsilon):
    '''
    Calculate nonlinear weights based on the smoothness of the solution.
    Last dimension of U should have 5 elements.
    epsilon is a small number (might need to be tuned).
    Returns: nonlinear weights.
    '''
    beta1, beta2, beta3, beta4 = smoothness_coefficients(U)

    # JS-type weights
    w1 = 1/35/(epsilon + beta1)**2
    w2 = 12/35/(epsilon + beta2)**2
    w3 = 18/35/(epsilon + beta3)**2
    w4 = 4/35/(epsilon + beta4)**2

    # Z-type weights
    tau = np.abs(beta1 + 3*beta2 - 3*beta3 - beta4)
    w1 = w1*((epsilon + beta1)**2 + tau**2)
    w2 = w2*((epsilon + beta2)**2 + tau**2)
    w3 = w3*((epsilon + beta3)**2 + tau**2)
    w4 = w4*((epsilon + beta4)**2 + tau**2)

    norm_fac = 1.0/(w1 + w2 + w3 + w4)

    return w1*norm_fac, w2*norm_fac, w3*norm_fac, w4*norm_fac

def calc_interface_flux(U, X, epsilon=1.0e-12):
    '''
    Calculate 7th-order WENO approximation at position X.
    Last dimension of U should have 7 elements
    X=0 is the leftmost cell edge, X=7 the rightmost cell edge.
    Returns: approximation of U at X.
    '''

    # Calculate the ENO approximation for the 3 different 3-point stencils
    u1, u2, u3, u4 = interpolate_to_edge(U, X)

    # Calculate the nonlinear weights based on the smoothness
    w1, w2, w3, w4 = nonlinear_weights(U, epsilon)

    # Return final approximation
    return w1*u1 + w2*u2 + w3*u3 + w4*u4

jit_module(nopython=True, error_model="numpy")