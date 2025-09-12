import numpy as np
from numba import jit_module

sound_speed = 1.0


def _primitive_variables(conserved_variables):
    prim = np.zeros_like(conserved_variables)
    prim[0] = conserved_variables[0]
    prim[1] = conserved_variables[1]/prim[0]

    return prim


def _flux_from_state_x(state_vector):
    prim = _primitive_variables(state_vector)

    flx = np.zeros_like(state_vector)
    flx[0] = state_vector[1]
    flx[1] = state_vector[1]*prim[1] + sound_speed**2*prim[0]

    return flx


def _multiply_with_left_eigenvectors_x(primitive_variables, state_vector):
    '''Multiply state_vector with left eigenvectors
    based on primitive_variables'''

    prim = _primitive_variables(primitive_variables)

    ret = np.zeros_like(state_vector)

    ret[0] = 0.5*(sound_speed - prim[1])*state_vector[0]/sound_speed \
        + 0.5*state_vector[1]/sound_speed
    ret[1] = 0.5*(sound_speed + prim[1])*state_vector[0]/sound_speed \
        - 0.5*state_vector[1]/sound_speed

    return ret


def _multiply_with_right_eigenvectors_x(primitive_variables, state_vector):
    '''Multiply state_vector with right eigenvectors
    based on primitive_variables'''

    prim = _primitive_variables(primitive_variables)

    ret = np.zeros_like(state_vector)

    ret[0] = state_vector[0] + state_vector[1]
    ret[1] = (prim[1] + sound_speed)*state_vector[0] + \
        (prim[1] - sound_speed)*state_vector[1]

    return ret


def _max_wave_speed_x(state_vector):
    prim = _primitive_variables(state_vector)

    return np.abs(prim[1]) + sound_speed


def flux_from_state(state, coords, time, dim):
    return _flux_from_state_x(state)


def multiply_with_left_eigenvectors(prim, state, time, dim):
    return _multiply_with_left_eigenvectors_x(prim, state)


def multiply_with_right_eigenvectors(prim, state, time, dim):
    return _multiply_with_right_eigenvectors_x(prim, state)


def max_wave_speed(U, coords, time, dim):
    return _max_wave_speed_x(U)


def source_func(U, coords, time):
    return 0*U


def allowed_state(state):
    return (state[0] > 0.0)


jit_module(nopython=True, error_model="numpy")
