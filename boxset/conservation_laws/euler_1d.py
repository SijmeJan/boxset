import numpy as np
from numba import jit_module

gamma = 1.4

def _primitive_variables(conserved_variables):
    prim = np.zeros_like(conserved_variables)
    prim[0] = conserved_variables[0]
    prim[1] = conserved_variables[1]/prim[0]
    prim[2] = (gamma - 1.0)*(conserved_variables[2] - 0.5*conserved_variables[1]**2/prim[0])

    return prim

def _flux_from_state_x(state_vector):
    prim = _primitive_variables(state_vector)

    flx = np.zeros_like(state_vector)
    flx[0] = state_vector[1]
    flx[1] = state_vector[1]*prim[1] + prim[2]
    flx[2] = (state_vector[2] + prim[2])*prim[1]

    return flx

def _multiply_with_left_eigenvectors_x(primitive_variables, state_vector):
    #prim = primitive_variables
    prim = _primitive_variables(_primitive_variables)

    # Sound speed
    c = np.sqrt(gamma*prim[2]/prim[0])

    b1 = (gamma-1)/(c*c)
    b2 = 0.5*b1*prim[1]**2

    ret = np.zeros_like(state_vector)

    ret[0] = \
        0.5*(b2 + prim[1]/c)*state_vector[0] \
        - 0.5*(b1*prim[1] + 1/c)*state_vector[1] \
        + 0.5*b1*state_vector[2]
    ret[1] = \
        (1 - b2)*state_vector[0] \
        + b1*prim[1]*state_vector[1] \
        - b1*state_vector[2]
    ret[2] = \
        0.5*(b2 - prim[1]/c)*state_vector[0] \
        - 0.5*(b1*prim[1] - 1/c)*state_vector[1] \
        + 0.5*b1*state_vector[2]

    return ret

def _multiply_with_right_eigenvectors_x(primitive_variables, state_vector):
    #prim = primitive_variables
    prim = _primitive_variables(_primitive_variables)

    ekin = 0.5*prim[1]**2

    # Enthalpy and sound speed
    h = ekin + gamma*prim[2]/(gamma-1)/prim[0]
    c = np.sqrt(gamma*prim[2]/prim[0])

    ret = np.zeros_like(state_vector)

    ret[0] = state_vector[0] + state_vector[1] + state_vector[2]
    ret[1] = (prim[1] - c)*state_vector[0] + prim[1]*state_vector[1] + (prim[1] + c)*state_vector[2]
    ret[2] = (h - c*prim[1])*state_vector[0] + ekin*state_vector[1] + (h + c*prim[1])*state_vector[2]
    return ret

def _max_wave_speed_x(state_vector):
    prim = _primitive_variables(state_vector)

    return np.abs(prim[1]) + np.sqrt(prim[2]/prim[0])

flux_from_state = {
    'dim1' : lambda state, coords: _flux_from_state_x(state),
}

multiply_with_left_eigenvectors = {
    'dim1' : lambda prim, state: _multiply_with_left_eigenvectors_x(prim, state),
}

multiply_with_right_eigenvectors = {
    'dim1' : lambda prim, state: _multiply_with_right_eigenvectors_x(prim, state),
}

max_wave_speed = {
    'dim1' : lambda U, coords: _max_wave_speed_x(U),
}

def source_func(U, coords):
    return 0.0*U

claw_funcs = {
    'max_wave_speed' : max_wave_speed,
    'multiply_with_left_eigenvectors' : multiply_with_left_eigenvectors,
    'multiply_with_right_eigenvectors' : multiply_with_right_eigenvectors,
    'flux_from_state' : flux_from_state,
    'source_func' : source_func
}

jit_module(nopython=True, error_model="numpy")