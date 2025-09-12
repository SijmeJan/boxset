import numpy as np
from numba import jit_module


def _flux_from_state_x(state):
    return state


def _multiply_with_left_eigenvectors_x(state):
    return state


def _multiply_with_right_eigenvectors_x(state):
    return state


def _max_wave_speed_x(coords):
    return np.ones(np.shape(coords[0]))


def flux_from_state(state, coords, time, dim):
    return _flux_from_state_x(state)


def multiply_with_left_eigenvectors(prim, state, time, dim):
    return _multiply_with_left_eigenvectors_x(state)


def multiply_with_right_eigenvectors(prim, state, time, dim):
    return _multiply_with_right_eigenvectors_x(state)


def max_wave_speed(state, coords, time, dim):
    return _max_wave_speed_x(coords)


def source_func(U, coords, time):
    return 0.0*U


def allowed_state(state):
    return np.logical_or(state > 0.0, state <= 0.0)


jit_module(nopython=True, error_model="numpy")
