import numpy as np
from numba import jit_module

gamma = 1.4

def _primitive_variables(conserved_variables):
    prim = np.zeros_like(conserved_variables)
    prim[0] = conserved_variables[0]
    prim[1] = conserved_variables[1]/prim[0]
    prim[2] = conserved_variables[2]/prim[0]
    prim[3] = (gamma - 1.0)*(conserved_variables[3] - 0.5*(conserved_variables[1]**2 + conserved_variables[2]**2)/prim[0])

    return prim

def _flux_from_state_x(state_vector):
    prim = _primitive_variables(state_vector)

    flx = np.zeros_like(state_vector)
    flx[0] = state_vector[1]
    flx[1] = state_vector[1]*prim[1] + prim[3]
    flx[2] = state_vector[1]*prim[2]
    flx[3] = (state_vector[3] + prim[3])*prim[1]

    return flx

def _flux_from_state_y(state_vector):
    prim = _primitive_variables(state_vector)

    flx = np.zeros_like(state_vector)
    flx[0] = state_vector[2]
    flx[1] = state_vector[2]*prim[1]
    flx[2] = state_vector[2]*prim[2] + prim[3]
    flx[3] = (state_vector[3] + prim[3])*prim[2]

    return flx

def _multiply_with_left_eigenvectors_x(primitive_variables, state_vector):
    '''Multiply state_vector with left eigenvectors based on primitive_variables'''
    prim = _primitive_variables(primitive_variables)

    # Sound speed
    c = np.sqrt(gamma*prim[3]/prim[0])

    b1 = (gamma-1)/(c*c)
    b2 = 0.5*b1*(prim[1]**2 + prim[2]**2)

    ret = np.zeros_like(state_vector)

    ret[0] = \
        0.5*(b2 + prim[1]/c)*state_vector[0] \
        - 0.5*(b1*prim[1] + 1/c)*state_vector[1] \
        - 0.5*b1*prim[2]*state_vector[2] \
        + 0.5*b1*state_vector[3]
    ret[1] = \
        (1 - b2)*state_vector[0] \
        + b1*prim[1]*state_vector[1] \
        + b1*prim[2]*state_vector[2] \
        - b1*state_vector[3]
    ret[2] = state_vector[2] - prim[2]*state_vector[0]
    ret[3] = \
        0.5*(b2 - prim[1]/c)*state_vector[0] \
        - 0.5*(b1*prim[1] - 1/c)*state_vector[1] \
        - 0.5*b1*prim[2]*state_vector[2] \
        + 0.5*b1*state_vector[3]

    return ret

def _multiply_with_left_eigenvectors_y(primitive_variables, state_vector):
    '''Multiply state_vector with left eigenvectors based on primitive_variables'''
    prim = _primitive_variables(primitive_variables)

    # Sound speed
    c = np.sqrt(gamma*prim[3]/prim[0])

    b1 = (gamma-1)/(c*c)
    b2 = 0.5*b1*(prim[1]**2 + prim[2]**2)

    ret = np.zeros_like(state_vector)

    ret[0] = \
        0.5*(b2 + prim[2]/c)*state_vector[0] \
        - 0.5*b1*prim[1]*state_vector[1] \
        - 0.5*(b1*prim[2] + 1/c)*state_vector[2] \
        + 0.5*b1*state_vector[3]
    ret[1] = state_vector[1] - prim[1]*state_vector[0]
    ret[2] = \
        (1 - b2)*state_vector[0] \
        + b1*prim[1]*state_vector[1] \
        + b1*prim[2]*state_vector[2] \
        - b1*state_vector[3]
    ret[3] = \
        0.5*(b2 - prim[2]/c)*state_vector[0] \
        - 0.5*b1*prim[1]*state_vector[1] \
        - 0.5*(b1*prim[2] - 1/c)*state_vector[2] \
        + 0.5*b1*state_vector[3]

    return ret

def _multiply_with_right_eigenvectors_x(primitive_variables, state_vector):
    '''Multiply state_vector with right eigenvectors based on primitive_variables'''
    prim = _primitive_variables(primitive_variables)

    ekin = 0.5*(prim[1]**2 + prim[2]**2)

    # Enthalpy and sound speed
    h = ekin + gamma*prim[3]/(gamma-1)/prim[0]
    c = np.sqrt(gamma*prim[3]/prim[0])

    ret = np.zeros_like(state_vector)

    ret[0] = state_vector[0] + state_vector[1] + state_vector[3]
    ret[1] = (prim[1] - c)*state_vector[0] + prim[1]*(state_vector[1] + 0*state_vector[2]) + (prim[1] + c)*state_vector[3]
    ret[2] = prim[2]*ret[0] + state_vector[2]
    ret[3] = (h - c*prim[1])*state_vector[0] + ekin*state_vector[1] + prim[2]*state_vector[2] + (h + c*prim[1])*state_vector[3]

    return ret

def _multiply_with_right_eigenvectors_y(primitive_variables, state_vector):
    '''Multiply state_vector with right eigenvectors based on primitive_variables'''
    prim = _primitive_variables(primitive_variables)

    ekin = 0.5*(prim[1]**2 + prim[2]**2)

    # Enthalpy and sound speed
    h = ekin + gamma*prim[3]/(gamma-1)/prim[0]
    c = np.sqrt(gamma*prim[3]/prim[0])

    ret = np.zeros_like(state_vector)

    ret[0] = state_vector[0] + state_vector[2] + state_vector[3]
    ret[1] = prim[1]*ret[0] + state_vector[1]
    ret[2] = (prim[2] - c)*state_vector[0] + prim[2]*(0*state_vector[1] + state_vector[2]) + (prim[2] + c)*state_vector[3]
    ret[3] = (h - c*prim[2])*state_vector[0] + ekin*state_vector[2] + prim[1]*state_vector[1] + (h + c*prim[2])*state_vector[3]

    return ret

def _max_wave_speed_x(state_vector):
    prim = _primitive_variables(state_vector)
    return np.abs(prim[1]) + np.sqrt(prim[3]/prim[0])

def _max_wave_speed_y(state_vector):
    prim = _primitive_variables(state_vector)
    return np.abs(prim[2]) + np.sqrt(prim[3]/prim[0])

def flux_from_state(state, coords, dim):
    if dim == 0:
        return _flux_from_state_x(state)
    return _flux_from_state_y(state)

def multiply_with_left_eigenvectors(prim, state, dim):
    if dim == 0:
        return _multiply_with_left_eigenvectors_x(prim, state)
    return _multiply_with_left_eigenvectors_y(prim, state)

def multiply_with_right_eigenvectors(prim, state, dim):
    if dim == 0:
        return _multiply_with_right_eigenvectors_x(prim, state)
    return _multiply_with_right_eigenvectors_y(prim, state)

def max_wave_speed(U , coords, dim):
    if dim == 0:
        return _max_wave_speed_x(U)
    return _max_wave_speed_y(U)

def source_func(U, coords):
    return 0.0*U

jit_module(nopython=True, error_model="numpy")