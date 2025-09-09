import numpy as np
from numba import jit_module

sound_speed = 1.0

def _primitive_variables(conserved_variables):
    prim = np.zeros_like(conserved_variables)
    prim[0] = conserved_variables[0]
    prim[1] = conserved_variables[1]/prim[0]
    prim[2] = conserved_variables[2]/prim[0]
    prim[3] = conserved_variables[3]/prim[0]

    return prim

def _flux_from_state_x(state_vector):
    prim = _primitive_variables(state_vector)

    flx = np.zeros_like(state_vector)
    flx[0] = state_vector[1]
    flx[1] = state_vector[1]*prim[1] + sound_speed**2*prim[0]
    flx[2] = state_vector[1]*prim[2]
    flx[3] = state_vector[1]*prim[3]

    return flx

def _flux_from_state_y(state_vector):
    prim = _primitive_variables(state_vector)

    flx = np.zeros_like(state_vector)
    flx[0] = state_vector[2]
    flx[1] = state_vector[2]*prim[1]
    flx[2] = state_vector[2]*prim[2] + sound_speed**2*prim[0]
    flx[3] = state_vector[2]*prim[3]

    return flx

def _flux_from_state_z(state_vector):
    prim = _primitive_variables(state_vector)

    flx = np.zeros_like(state_vector)
    flx[0] = state_vector[3]
    flx[1] = state_vector[3]*prim[1]
    flx[2] = state_vector[3]*prim[2]
    flx[3] = state_vector[3]*prim[3] + sound_speed**2*prim[0]

    return flx

def _multiply_with_left_eigenvectors_x(primitive_variables, state_vector):
    '''Multiply state_vector with left eigenvectors based on primitive_variables'''

    prim = _primitive_variables(primitive_variables)

    ret = np.zeros_like(state_vector)

    ret[0] = 0.5*(sound_speed - prim[1])*state_vector[0]/sound_speed \
        + 0.5*state_vector[1]/sound_speed
    ret[1] = 0.5*(sound_speed + prim[1])*state_vector[0]/sound_speed \
        - 0.5*state_vector[1]/sound_speed
    ret[2] = state_vector[2] - prim[2]*state_vector[0]
    ret[3] = state_vector[3] - prim[3]*state_vector[0]

    return ret

def _multiply_with_left_eigenvectors_y(primitive_variables, state_vector):
    '''Multiply state_vector with left eigenvectors based on primitive_variables'''

    prim = _primitive_variables(primitive_variables)

    ret = np.zeros_like(state_vector)

    ret[0] = 0.5*(sound_speed - prim[2])*state_vector[0]/sound_speed \
        + 0.5*state_vector[2]/sound_speed
    ret[2] = 0.5*(sound_speed + prim[2])*state_vector[0]/sound_speed \
        - 0.5*state_vector[2]/sound_speed
    ret[1] = state_vector[1] - prim[1]*state_vector[0]
    ret[3] = state_vector[3] - prim[3]*state_vector[0]

    return ret

def _multiply_with_left_eigenvectors_z(primitive_variables, state_vector):
    '''Multiply state_vector with left eigenvectors based on primitive_variables'''

    prim = _primitive_variables(primitive_variables)

    ret = np.zeros_like(state_vector)

    ret[0] = 0.5*(sound_speed - prim[3])*state_vector[0]/sound_speed \
        + 0.5*state_vector[3]/sound_speed
    ret[3] = 0.5*(sound_speed + prim[3])*state_vector[0]/sound_speed \
        - 0.5*state_vector[3]/sound_speed
    ret[2] = state_vector[2] - prim[2]*state_vector[0]
    ret[1] = state_vector[1] - prim[1]*state_vector[0]

    return ret

def _multiply_with_right_eigenvectors_x(primitive_variables, state_vector):
    '''Multiply state_vector with right eigenvectors based on primitive_variables'''

    prim = _primitive_variables(primitive_variables)

    ret = np.zeros_like(state_vector)

    ret[0] = state_vector[0] + state_vector[1]
    ret[1] = (prim[1] + sound_speed)*state_vector[0] + (prim[1] - sound_speed)*state_vector[1]
    ret[2] = prim[2]*ret[0] + state_vector[2]
    ret[3] = prim[3]*ret[0] + state_vector[3]

    return ret

def _multiply_with_right_eigenvectors_y(primitive_variables, state_vector):
    '''Multiply state_vector with right eigenvectors based on primitive_variables'''

    prim = _primitive_variables(primitive_variables)

    ret = np.zeros_like(state_vector)

    ret[0] = state_vector[0] + state_vector[2]
    ret[2] = (prim[2] + sound_speed)*state_vector[0] + (prim[2] - sound_speed)*state_vector[2]
    ret[1] = prim[1]*ret[0] + state_vector[1]
    ret[3] = prim[3]*ret[0] + state_vector[3]

    return ret

def _multiply_with_right_eigenvectors_z(primitive_variables, state_vector):
    '''Multiply state_vector with right eigenvectors based on primitive_variables'''

    prim = _primitive_variables(primitive_variables)

    ret = np.zeros_like(state_vector)

    ret[0] = state_vector[0] + state_vector[3]
    ret[3] = (prim[3] + sound_speed)*state_vector[0] + (prim[3] - sound_speed)*state_vector[3]
    ret[2] = prim[2]*ret[0] + state_vector[2]
    ret[1] = prim[1]*ret[0] + state_vector[1]

    return ret

def _max_wave_speed_x(state_vector):
    prim = _primitive_variables(state_vector)

    return np.abs(prim[1]) + sound_speed

def _max_wave_speed_y(state_vector):
    prim = _primitive_variables(state_vector)
    return np.abs(prim[2]) + sound_speed

def _max_wave_speed_z(state_vector):
    prim = _primitive_variables(state_vector)
    return np.abs(prim[3]) + sound_speed

def flux_from_state(state, coords, dim):
    if dim == 0:
        return _flux_from_state_x(state)
    if dim == 1:
        return _flux_from_state_y(state)
    return _flux_from_state_z(state)

def multiply_with_left_eigenvectors(prim, state, dim):
    if dim == 0:
        return _multiply_with_left_eigenvectors_x(prim, state)
    if dim == 1:
        return _multiply_with_left_eigenvectors_y(prim, state)
    return _multiply_with_left_eigenvectors_z(prim, state)

def multiply_with_right_eigenvectors(prim, state, dim):
    if dim == 0:
        return _multiply_with_right_eigenvectors_x(prim, state)
    if dim == 1:
        return _multiply_with_right_eigenvectors_y(prim, state)
    return _multiply_with_right_eigenvectors_z(prim, state)

def max_wave_speed(U , coords, dim):
    if dim == 0:
        return _max_wave_speed_x(U)
    if dim == 1:
        return _max_wave_speed_y(U)
    return _max_wave_speed_z(U)

def source_func(U, coords):
    return 0*U

def allowed_state(state):
    return (state[0] > 0.0)

jit_module(nopython=True, error_model="numpy")