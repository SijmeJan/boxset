import numpy as np
from numba import jit_module

# Maximum order 2*r-1
weno_r = 3

gamma_lo = 0.85
gamma_hi = 0.95

gamma3_5 = gamma_hi
gamma1_3 = (1 - gamma_hi)*(1 - gamma_lo)/2
gamma2_3 = (1 - gamma_hi)*gamma_lo
gamma3_3 = gamma1_3

def interpolate_to_edge(U, X):
    '''
    Last dimension of U should have 5 elements. X is the evaluation point: X=3 for x_{i+1/2}.
    Returns estimates for U using the three-point left-, centre- and right-biased stencils,
    as well as an estimate based on the 5-point centered stencil.
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

    L1 = X - 5/2
    L2 = L1*L1 - 1/12
    L3 = L1*L1*L1 - 3*L1/20
    L4 = L1*L1*L1*L1 - 3*L1*L1/14 + 3/560

    ux = (-82*U[...,1] + 11*U[...,0] + 82*U[...,3] - 11*U[...,4])/120
    ux2 = (40*U[...,1] - 3*U[...,0] - 74*U[...,2] + 40*U[...,3] - 3*U[...,4])/56
    ux3 = (2*U[...,1] - U[...,0] - 2*U[...,3] + U[...,4])/12
    ux4 = (-4*U[...,1] + U[...,0] + 6*U[...,2] - 4*U[...,3] + U[...,4])/24

    P5 = U[...,2] + ux*L1 + ux2*L2 + ux3*L3 + ux4*L4

    return PL, PC, PR, P5

def smoothness_coefficients(U):
    '''
    Last dimension of U should have five elements.
    Returns the smoothness coefficients for the left, centre and right-biased stencils,
    as well as the smoothness coefficient for the 5-point centered stencil.
    '''
    betaL = 13*(U[...,0] - 2*U[...,1] + U[...,2])**2/12 + 0.25*(U[...,0] - 4*U[...,1] + 3*U[...,2])**2
    betaC = 13*(U[...,1] - 2*U[...,2] + U[...,3])**2/12 + 0.25*(U[...,1] - U[...,3])**2
    betaR = 13*(U[...,2] - 2*U[...,3] + U[...,4])**2/12 + 0.25*(3*U[...,2] - 4*U[...,3] + U[...,4])**2

    ux = (-82*U[...,1] + 11*U[...,0] + 82*U[...,3] - 11*U[...,4])/120
    ux2 = (40*U[...,1] - 3*U[...,0] - 74*U[...,2] + 40*U[...,3] - 3*U[...,4])/56
    ux3 = (2*U[...,1] - U[...,0] - 2*U[...,3] + U[...,4])/12
    ux4 = (-4*U[...,1] + U[...,0] + 6*U[...,2] - 4*U[...,3] + U[...,4])/24

    beta5 = (ux + ux3/10)**2 + 13*(ux2 + 123*ux4/455)**2/3 + 781*ux3**2/20 + 1421461*ux4**2/2275

    return betaL, betaC, betaR, beta5

def nonlinear_weights(U, epsilon):
    '''
    Calculate nonlinear weights based on the smoothness of the solution.
    Last dimension of U should have 5 elements.
    epsilon is a small number (might need to be tuned).
    Returns: nonlinear weights.
    '''
    betaL, betaC, betaR, beta5 = smoothness_coefficients(U)

    tau = (np.abs(beta5 - betaR) + np.abs(beta5 - betaC) + np.abs(beta5 - betaL))/3
    w5 = gamma3_5*(1 + (tau/(beta5 + epsilon))**2)
    wL = gamma1_3*(1 + (tau/(betaL + epsilon))**2)
    wC = gamma2_3*(1 + (tau/(betaC + epsilon))**2)
    wR = gamma3_3*(1 + (tau/(betaR + epsilon))**2)

    norm_fac = 1.0/(wL + wC + wR + w5)

    return wL*norm_fac, wC*norm_fac, wR*norm_fac, w5*norm_fac

def calc_interface_flux(U, X, epsilon=1.0e-12):
    '''
    Calculate 5th/3rd-order WENO-AO approximation at position X.
    Last dimension of U should have 5 elements
    X=0 is the leftmost cell edge, X=5 the rightmost cell edge.
    Returns: approximation of U at X.
    '''

    # Calculate the ENO approximation for the 3 different 3-point stencils
    uL, uC, uR, u5 = interpolate_to_edge(U, X)

    # Calculate the nonlinear weights based on the smoothness
    wL, wC, wR, w5 = nonlinear_weights(U, epsilon)

    # Return final approximation
    return wL*uL + wC*uC + wR*uR + w5*(u5 - gamma1_3*uL - gamma2_3*uC - gamma3_3*uR)/gamma3_5

jit_module(nopython=True, error_model="numpy")