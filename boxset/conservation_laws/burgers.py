import numpy as np
from numba import jit_module

flux_from_state = {
    'dim1' : lambda state, coords: 0.5*state*state
}

multiply_with_left_eigenvectors = {
    'dim1' : lambda prim, state: state
}

multiply_with_right_eigenvectors = {
    'dim1' : lambda prim, state: state
}

max_wave_speed = {
    'dim1' : lambda U, coords: np.abs(U[0,:])
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