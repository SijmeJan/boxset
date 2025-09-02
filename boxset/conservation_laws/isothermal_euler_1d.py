import numpy as np
from numba import jit_module

sound_speed = 1.0

def _primitive_variables(conserved_variables):
    prim = np.zeros_like(conserved_variables)
    prim[0] = conserved_variables[0]
    prim[1] = conserved_variables[1]/prim[0]
    return prim

def _flux_from_state(state_vector):
    flx = np.zeros_like(state_vector)
    flx[0] = state_vector[1]
    flx[1] = state_vector[1]**2/state_vector[0] + sound_speed**2*state_vector[0]
    return flx

def _multiply_with_left_eigenvectors(primitive_variables, state_vector):
    primitive_variables = _primitive_variables(primitive_variables)

    v = primitive_variables[1]
    c = sound_speed

    ret = np.zeros_like(state_vector)
    ret[0] = 0.5*(v*v - c*c)*state_vector[0]/c + 0.5*(c - v)*state_vector[1]/c
    ret[1] = 0.5*(c*c - v*v)*state_vector[0]/c + 0.5*(c + v)*state_vector[1]/c
    return ret

def _multiply_with_right_eigenvectors(primitive_variables, state_vector):
    primitive_variables = _primitive_variables(primitive_variables)

    v = primitive_variables[1]
    c = sound_speed

    ret = np.zeros_like(state_vector)
    ret[0] = state_vector[0]/(v - c) + state_vector[1]/(v + c)
    ret[1] = state_vector[0] + state_vector[1]
    return ret

def _max_wave_speed(primitive_variables):
    return np.abs(primitive_variables[1]) + sound_speed

flux_from_state = {
    'dim1' : lambda state, coords: _flux_from_state(state)
}

multiply_with_left_eigenvectors = {
    'dim1' : lambda prim, state: _multiply_with_left_eigenvectors(prim, state)
}

multiply_with_right_eigenvectors = {
    'dim1' : lambda prim, state: _multiply_with_right_eigenvectors(prim, state)
}

max_wave_speed = {
    'dim1' : lambda U, coords: _max_wave_speed(U)
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

def allowed_state(state):
    return (state[0] > 0.0)

jit_module(nopython=True, error_model="numpy")