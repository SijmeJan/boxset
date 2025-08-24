import numpy as np
from numba import jit_module

# Maximum order 2*r-1
weno_r = 4

gamma_lo = 0.85
gamma_hi = 0.95

gamma3_7 = gamma_hi
gamma1_3 = (1 - gamma_hi)*(1 - gamma_lo)/2
gamma2_3 = (1 - gamma_hi)*gamma_lo
gamma3_3 = gamma1_3

def interpolate_to_edge(U, X):
    '''
    Last dimension of U should have 7 elements. X is the evaluation point: X=4 for x_{i+1/2}.
    Returns estimates for U using the three-point left-, centre- and right-biased stencils,
    as well as an estimate based on the 7-point centered stencil.
    '''
    # | i-3 | i-2 | i-1 | i | i+1 | i+2 | i+3 |
    # This is where we calculate PL, PC, PR

    u   = U[..., 2] - (U[..., 1] - 2*U[..., 2] + U[..., 3])/24
    ux  = 0.5*(U[...,3] - U[...,1])
    uxx = U[...,3] - 2*U[...,2] + U[...,1]
    PL = u + ux*(X - 2.5) + 0.5*uxx*(X - 2.5)**2

    u   = U[..., 3] - (U[..., 2] - 2*U[..., 3] + U[..., 4])/24
    ux  = 0.5*(U[...,4] - U[...,2])
    uxx = U[...,4] - 2*U[...,3] + U[...,2]
    PC = u + ux*(X - 3.5) + 0.5*uxx*(X - 3.5)**2

    u   = U[..., 4] - (U[..., 3] - 2*U[..., 4] + U[..., 5])/24
    ux  = 0.5*(U[...,5] - U[...,3])
    uxx = U[...,5] - 2*U[...,4] + U[...,3]
    PR = u + ux*(X - 4.5) + 0.5*uxx*(X - 4.5)**2

    L1 = X - 7/2
    L2 = L1*L1 - 1/12
    L3 = L1*L1*L1 - 3*L1/20
    L4 = L1*L1*L1*L1 - 3*L1*L1/14 + 3/560
    L5 = L1**5 - 5*L1**3/18 + 5*L1/336
    L6 = L1**6 - 15*L1**4/44 + 5*L1**2/176 - 5/14784

    ux = (-7843*U[..., 2] + 1688*U[..., 1] - 191*U[..., 0] + 7843*U[..., 4] - 1688*U[..., 5] + 191*U[..., 6])/10080
    ux2 = (8385*U[..., 2] - 1014*U[..., 1] + 79*U[..., 0] - 14900*U[..., 3] + 8385*U[..., 4] - 1014*U[..., 5] + 79*U[..., 6])/10080
    ux3 = (61*U[..., 2] - 38*U[..., 1] + 5*U[..., 0] - 61*U[..., 4] + 38*U[..., 5] - 5*U[..., 6])/216
    ux4 = (-459*U[..., 2] + 144*U[..., 1] - 13*U[..., 0] + 656*U[..., 3] - 459*U[..., 4] + 144*U[..., 5] - 13*U[..., 6])/1584
    ux5 = (-5*U[..., 2] + 4*U[..., 1] - U[..., 0] + 5*U[..., 4] - 4*U[..., 5] + U[..., 6])/240
    ux6 = (15*U[..., 2] - 6*U[..., 1] + U[..., 0] - 20*U[..., 3] + 15*U[..., 4] - 6*U[..., 5] + U[..., 6])/720

    P7 = U[...,3] + ux*L1 + ux2*L2 + ux3*L3 + ux4*L4 + ux5*L5 + ux6*L6

    ux = (-82*U[...,2] + 11*U[...,1] + 82*U[...,4] - 11*U[...,5])/120
    ux2 = (40*U[...,2] -  3*U[...,1] - 74*U[...,3] + 40*U[...,4] - 3*U[...,5])/56
    ux3 = ( 2*U[...,2] -    U[...,1] -  2*U[...,4] +    U[...,5])/12
    ux4 = (-4*U[...,2] +    U[...,1] +  6*U[...,3] -  4*U[...,4] +   U[...,5])/24

    P5 = U[...,3] + ux*L1 + ux2*L2 + ux3*L3 + ux4*L4

    return PL, PC, PR, P5, P7

def smoothness_coefficients(U):
    '''
    Last dimension of U should have five elements.
    Returns the smoothness coefficients for the left, centre and right-biased stencils,
    as well as the smoothness coefficient for the 5-point centered stencil.
    '''
    betaL = 13*(U[...,1] - 2*U[...,2] + U[...,3])**2/12 + 0.25*(  U[...,1] - 4*U[...,2] + 3*U[...,3])**2
    betaC = 13*(U[...,2] - 2*U[...,3] + U[...,4])**2/12 + 0.25*(  U[...,2] -   U[...,4])**2
    betaR = 13*(U[...,3] - 2*U[...,4] + U[...,5])**2/12 + 0.25*(3*U[...,3] - 4*U[...,4] +   U[...,5])**2

    ux = (-7843*U[..., 2] + 1688*U[..., 1] - 191*U[..., 0] + 7843*U[..., 4] - 1688*U[..., 5] + 191*U[..., 6])/10080
    ux2 = (8385*U[..., 2] - 1014*U[..., 1] + 79*U[..., 0] - 14900*U[..., 3] + 8385*U[..., 4] - 1014*U[..., 5] + 79*U[..., 6])/10080
    ux3 = (61*U[..., 2] - 38*U[..., 1] + 5*U[..., 0] - 61*U[..., 4] + 38*U[..., 5] - 5*U[..., 6])/216
    ux4 = (-459*U[..., 2] + 144*U[..., 1] - 13*U[..., 0] + 656*U[..., 3] - 459*U[..., 4] + 144*U[..., 5] - 13*U[..., 6])/1584
    ux5 = (-5*U[..., 2] + 4*U[..., 1] - U[..., 0] + 5*U[..., 4] - 4*U[..., 5] + U[..., 6])/240
    ux6 = (15*U[..., 2] - 6*U[..., 1] + U[..., 0] - 20*U[..., 3] + 15*U[..., 4] - 6*U[..., 5] + U[..., 6])/720

    beta7 = (ux + ux3/10 + ux5/126)**2 + 13*(ux2 + 123*ux4/455 + 85*ux6/2002)**2/3 + 781*(ux3 + 26045*ux5/49203)**2/20 + 1421461*(ux4 + 81596225*ux6/93816426)**2/2275 + 21520059541*ux5**2/1377684 + 15510384942580921*ux6**2/27582029244

    ux = (-82*U[...,2] + 11*U[...,1] + 82*U[...,4] - 11*U[...,5])/120
    ux2 = (40*U[...,2] - 3*U[...,1] - 74*U[...,3] + 40*U[...,4] - 3*U[...,5])/56
    ux3 = (2*U[...,2] - U[...,1] - 2*U[...,4] + U[...,5])/12
    ux4 = (-4*U[...,2] + U[...,1] + 6*U[...,3] - 4*U[...,4] + U[...,5])/24

    beta5 = (ux + ux3/10)**2 + 13*(ux2 + 123*ux4/455)**2/3 + 781*ux3**2/20 + 1421461*ux4**2/2275

    return betaL, betaC, betaR, beta5, beta7

def nonlinear_weights(U, epsilon):
    '''
    Calculate nonlinear weights based on the smoothness of the solution.
    Last dimension of U should have 7 elements.
    epsilon is a small number (might need to be tuned).
    Returns: nonlinear weights.
    '''
    betaL, betaC, betaR, beta5, beta7 = smoothness_coefficients(U)

    tau7 = (np.abs(beta7 - betaR) + np.abs(beta7 - betaC) + np.abs(beta7 - betaL))/3
    w7 = gamma3_7*(1 + (tau7/(beta7 + epsilon))**2)
    wL7 = gamma1_3*(1 + (tau7/(betaL + epsilon))**2)
    wC7 = gamma2_3*(1 + (tau7/(betaC + epsilon))**2)
    wR7 = gamma3_3*(1 + (tau7/(betaR + epsilon))**2)

    tau5 = (np.abs(beta5 - betaR) + np.abs(beta5 - betaC) + np.abs(beta5 - betaL))/3
    w5 = gamma3_7*(1 + (tau5/(beta5 + epsilon))**2)
    wL5 = gamma1_3*(1 + (tau5/(betaL + epsilon))**2)
    wC5 = gamma2_3*(1 + (tau5/(betaC + epsilon))**2)
    wR5 = gamma3_3*(1 + (tau5/(betaR + epsilon))**2)

    sigma = np.abs(beta5 - beta7)
    nu73 = gamma_hi*(1 + sigma/(beta7 + epsilon))
    nu53 = (1- gamma_hi)*(1 + sigma/(beta5 + epsilon))

    norm_fac7 = 1.0/(wL7 + wC7 + wR7 + w7)
    norm_fac5 = 1.0/(wL5 + wC5 + wR5 + w5)
    norm_fac_nu = 1.0/(nu53 + nu73)

    return wL5*norm_fac5, wC5*norm_fac5, wR5*norm_fac5, w5*norm_fac5, wL7*norm_fac7, wC7*norm_fac7, wR7*norm_fac7, w7*norm_fac7, nu53*norm_fac_nu, nu73*norm_fac_nu

def calc_interface_flux(U, X, epsilon=1.0e-12):
    '''
    Calculate 5th/3rd-order WENO-AO approximation at position X.
    Last dimension of U should have 7 elements
    X=0 is the leftmost cell edge, X=7 the rightmost cell edge.
    Returns: approximation of U at X.
    '''

    # Calculate the ENO approximation for the 3 different 3-point stencils
    #uL, uC, uR, u7 = interpolate_to_edge(U, X)
    uL, uC, uR, u5, u7 = interpolate_to_edge(U, X)

    # Calculate the nonlinear weights based on the smoothness
    #wL, wC, wR, w7 = nonlinear_weights(U, epsilon)
    wL5, wC5, wR5, w5, wL7, wC7, wR7, w7, nu53, nu73 = nonlinear_weights(U, epsilon)

    # Return final approximation
    P73 = wL7*uL + wC7*uC + wR7*uR + w7*(u7 - gamma1_3*uL - gamma2_3*uC - gamma3_3*uR)/gamma3_7
    P53 = wL5*uL + wC5*uC + wR5*uR + w5*(u5 - gamma1_3*uL - gamma2_3*uC - gamma3_3*uR)/gamma3_7
    return nu73*(P73 - (1 - gamma_hi)*P53)/gamma_hi + nu53*P53

    #return wL5*uL + wC5*uC + wR5*uR + w5*(u5 - gamma1_3*uL - gamma2_3*uC - gamma3_3*uR)/gamma3_7

jit_module(nopython=True, error_model="numpy")