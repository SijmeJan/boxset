import numpy as np
from numba import jit_module

def _flux_from_state_x(state_vector, coords):
    # y*u
    # state: (1, n_y, n_x)

    flx = np.zeros_like(state_vector)
    for i in range(0, len(coords[0])):
        flx[...,i] = state_vector[...,i]*coords[1]

    return flx

def _flux_from_state_y(state_vector, coords):
    # -y*u
    # state: (1, n_x, n_y)

    flx = -state_vector*coords[1]

    return flx

def _multiply_with_left_eigenvectors_x(primitive_variables, state_vector):
    '''Multiply state_vector with left eigenvectors based on primitive_variables'''
    return state_vector

def _multiply_with_left_eigenvectors_y(primitive_variables, state_vector):
    '''Multiply state_vector with left eigenvectors based on primitive_variables'''
    return state_vector

def _multiply_with_right_eigenvectors_x(primitive_variables, state_vector):
    '''Multiply state_vector with right eigenvectors based on primitive_variables'''
    return state_vector

def _multiply_with_right_eigenvectors_y(primitive_variables, state_vector):
    '''Multiply state_vector with right eigenvectors based on primitive_variables'''
    return state_vector

def _max_wave_speed_x(state_vector, coords):
    ret = np.ones_like(state_vector)

    for i in range(0, len(coords[0])):
        ret[...,i] = ret[...,i]*np.abs(coords[1])

    return ret

def _max_wave_speed_y(state_vector, coords):
    ret = np.ones_like(state_vector)

    ret = ret*np.abs(coords[1])

    return ret

def flux_from_state(state, coords, dim):
    if dim == 0:
        return _flux_from_state_x(state, coords)
    return _flux_from_state_y(state, coords)

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
        return _max_wave_speed_x(U, coords)
    return _max_wave_speed_y(U, coords)

def source_func(U, coords):
    '''
    Shape of U should be (n_eq, len(x), len(y),...)
    '''

    ret = np.zeros_like(U)
    y = coords[1]
    dy = y[1] - y[0]
    for i in range(3, len(y)-3):
        ret[...,i] = U[...,i+1] - 2*U[...,i] + U[...,i-1]

    return 0.0002*ret/dy**2

jit_module(nopython=True, error_model="numpy")