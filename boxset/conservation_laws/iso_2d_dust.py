import numpy as np
from numba import jit_module

sound_speed = 1.0
sound_speed_dust = 1.0e-10
stokes = 1.0
pressure_parameter = 0.05

def _primitive_variables(conserved_variables):
    prim = np.zeros_like(conserved_variables)
    prim[0] = conserved_variables[0]
    prim[1] = conserved_variables[1]/prim[0]
    prim[2] = conserved_variables[2]/prim[0]
    prim[3] = conserved_variables[3]/prim[0]
    prim[4] = conserved_variables[4]
    prim[5] = conserved_variables[5]/prim[4]
    prim[6] = conserved_variables[6]/prim[4]
    prim[7] = conserved_variables[7]/prim[4]

    return prim

def _flux_from_state_x(state_vector):
    prim = _primitive_variables(state_vector)

    flx = np.zeros_like(state_vector)
    flx[0] = state_vector[1]
    flx[1] = state_vector[1]*prim[1] + sound_speed**2*prim[0]
    flx[2] = state_vector[1]*prim[2]
    flx[3] = state_vector[1]*prim[3]
    flx[4] = state_vector[5]
    flx[5] = state_vector[5]*prim[5] + sound_speed_dust**2*prim[4]
    flx[6] = state_vector[5]*prim[6]
    flx[7] = state_vector[5]*prim[7]

    return flx

def _flux_from_state_y(state_vector):
    prim = _primitive_variables(state_vector)

    flx = np.zeros_like(state_vector)
    flx[0] = state_vector[2]
    flx[1] = state_vector[2]*prim[1]
    flx[2] = state_vector[2]*prim[2] + sound_speed**2*prim[0]
    flx[3] = state_vector[2]*prim[3]
    flx[4] = state_vector[6]
    flx[5] = state_vector[6]*prim[5]
    flx[6] = state_vector[6]*prim[6] + sound_speed_dust**2*prim[4]
    flx[7] = state_vector[6]*prim[7]

    return flx

def _flux_from_state_z(state_vector):
    prim = _primitive_variables(state_vector)

    flx = np.zeros_like(state_vector)
    flx[0] = state_vector[3]
    flx[1] = state_vector[3]*prim[1]
    flx[2] = state_vector[3]*prim[2]
    flx[3] = state_vector[3]*prim[3] + sound_speed**2*prim[0]
    flx[4] = state_vector[7]
    flx[5] = state_vector[7]*prim[5]
    flx[6] = state_vector[7]*prim[6]
    flx[7] = state_vector[7]*prim[7]+ sound_speed_dust**2*prim[4]

    return flx

def _multiply_with_left_eigenvectors_x(primitive_variables, state_vector):
    '''Multiply state_vector with left eigenvectors based on primitive_variables'''
    return state_vector

    #prim = _primitive_variables(primitive_variables)

    #ret = np.zeros_like(state_vector)

    #ret[0] = 0.5*(sound_speed - prim[1])*state_vector[0]/sound_speed \
    #    + 0.5*state_vector[1]/sound_speed
    #ret[1] = 0.5*(sound_speed + prim[1])*state_vector[0]/sound_speed \
    #    - 0.5*state_vector[1]/sound_speed
    #ret[2] = state_vector[2] - prim[2]*state_vector[0]
    #ret[3] = state_vector[3] - prim[3]*state_vector[0]

    #ret[4] = 0.5*(sound_speed_dust - prim[5])*state_vector[4]/sound_speed_dust \
    #    + 0.5*state_vector[5]/sound_speed_dust
    #ret[5] = 0.5*(sound_speed_dust + prim[5])*state_vector[4]/sound_speed_dust \
    #    - 0.5*state_vector[5]/sound_speed_dust
    #ret[6] = state_vector[6] - prim[6]*state_vector[4]
    #ret[7] = state_vector[7] - prim[7]*state_vector[4]

    #ret[4] = state_vector[4]
    #ret[5] = state_vector[5]
    #ret[6] = state_vector[6]
    #ret[7] = state_vector[7]

    #return ret

def _multiply_with_left_eigenvectors_y(primitive_variables, state_vector):
    '''Multiply state_vector with left eigenvectors based on primitive_variables'''
    return state_vector

    #prim = _primitive_variables(primitive_variables)

    #ret = np.zeros_like(state_vector)

    #ret[0] = 0.5*(sound_speed - prim[2])*state_vector[0]/sound_speed \
    #    + 0.5*state_vector[2]/sound_speed
    #ret[2] = 0.5*(sound_speed + prim[2])*state_vector[0]/sound_speed \
    #    - 0.5*state_vector[2]/sound_speed
    #ret[1] = state_vector[1] - prim[1]*state_vector[0]
    #ret[3] = state_vector[3] - prim[3]*state_vector[0]

    #ret[4] = 0.5*(sound_speed_dust - prim[6])*state_vector[4]/sound_speed_dust \
    #    + 0.5*state_vector[6]/sound_speed_dust
    #ret[6] = 0.5*(sound_speed_dust + prim[6])*state_vector[4]/sound_speed_dust \
    #    - 0.5*state_vector[6]/sound_speed_dust
    #ret[5] = state_vector[5] - prim[5]*state_vector[4]
    #ret[7] = state_vector[7] - prim[7]*state_vector[4]

    #ret[4] = state_vector[4]
    #ret[5] = state_vector[5]
    #ret[6] = state_vector[6]
    #ret[7] = state_vector[7]

    #return ret

def _multiply_with_left_eigenvectors_z(primitive_variables, state_vector):
    '''Multiply state_vector with left eigenvectors based on primitive_variables'''
    return state_vector

    #prim = _primitive_variables(primitive_variables)

    #ret = np.zeros_like(state_vector)

    #ret[0] = 0.5*(sound_speed - prim[3])*state_vector[0]/sound_speed \
    #    + 0.5*state_vector[3]/sound_speed
    #ret[3] = 0.5*(sound_speed + prim[3])*state_vector[0]/sound_speed \
    #    - 0.5*state_vector[3]/sound_speed
    #ret[2] = state_vector[2] - prim[2]*state_vector[0]
    #ret[1] = state_vector[1] - prim[1]*state_vector[0]

    #ret[4] = 0.5*(sound_speed_dust - prim[7])*state_vector[4]/sound_speed_dust \
    #    + 0.5*state_vector[7]/sound_speed_dust
    #ret[7] = 0.5*(sound_speed_dust + prim[7])*state_vector[4]/sound_speed_dust \
    #    - 0.5*state_vector[7]/sound_speed_dust
    #ret[6] = state_vector[6] - prim[6]*state_vector[4]
    #ret[5] = state_vector[5] - prim[5]*state_vector[4]

    #ret[4] = state_vector[4]
    #ret[5] = state_vector[5]
    #ret[6] = state_vector[6]
    #ret[7] = state_vector[7]

    #return ret

def _multiply_with_right_eigenvectors_x(primitive_variables, state_vector):
    '''Multiply state_vector with right eigenvectors based on primitive_variables'''
    return state_vector

    #prim = _primitive_variables(primitive_variables)

    #ret = np.zeros_like(state_vector)

    #ret[0] = state_vector[0] + state_vector[1]
    #ret[1] = (prim[1] + sound_speed)*state_vector[0] + (prim[1] - sound_speed)*state_vector[1]
    #ret[2] = prim[2]*ret[0] + state_vector[2]
    #ret[3] = prim[3]*ret[0] + state_vector[3]

    #ret[4] = state_vector[4] + state_vector[5]
    #ret[5] = (prim[5] + sound_speed_dust)*state_vector[4] + (prim[5] - sound_speed_dust)*state_vector[5]
    #ret[6] = prim[6]*ret[4] + state_vector[6]
    #ret[7] = prim[7]*ret[4] + state_vector[7]

    #ret[4] = state_vector[4]
    #ret[5] = state_vector[5]
    #ret[6] = state_vector[6]
    #ret[7] = state_vector[7]

    #return ret

def _multiply_with_right_eigenvectors_y(primitive_variables, state_vector):
    '''Multiply state_vector with right eigenvectors based on primitive_variables'''
    return state_vector

    #prim = _primitive_variables(primitive_variables)

    #ret = np.zeros_like(state_vector)

    #ret[0] = state_vector[0] + state_vector[2]
    #ret[2] = (prim[2] + sound_speed)*state_vector[0] + (prim[2] - sound_speed)*state_vector[2]
    #ret[1] = prim[1]*ret[0] + state_vector[1]
    #ret[3] = prim[3]*ret[0] + state_vector[3]

    #ret[4] = state_vector[4] + state_vector[6]
    #ret[6] = (prim[6] + sound_speed_dust)*state_vector[4] + (prim[6] - sound_speed_dust)*state_vector[6]
    #ret[5] = prim[5]*ret[4] + state_vector[5]
    #ret[7] = prim[7]*ret[4] + state_vector[7]

    #ret[4] = state_vector[4]
    #ret[5] = state_vector[5]
    #ret[6] = state_vector[6]
    #ret[7] = state_vector[7]

    #return ret

def _multiply_with_right_eigenvectors_z(primitive_variables, state_vector):
    '''Multiply state_vector with right eigenvectors based on primitive_variables'''
    # When using CWENO, no need for characteristic decomposition
    return state_vector

    #prim = _primitive_variables(primitive_variables)

    #ret = np.zeros_like(state_vector)

    #ret[0] = state_vector[0] + state_vector[3]
    #ret[3] = (prim[3] + sound_speed)*state_vector[0] + (prim[3] - sound_speed)*state_vector[3]
    #ret[2] = prim[2]*ret[0] + state_vector[2]
    #ret[1] = prim[1]*ret[0] + state_vector[1]

    #ret[4] = state_vector[4] + state_vector[7]
    #ret[7] = (prim[7] + sound_speed_dust)*state_vector[4] + (prim[7] - sound_speed_dust)*state_vector[7]
    #ret[6] = prim[6]*ret[4] + state_vector[6]
    #ret[5] = prim[5]*ret[4] + state_vector[5]

    #ret[4] = state_vector[4]
    #ret[5] = state_vector[5]
    #ret[6] = state_vector[6]
    #ret[7] = state_vector[7]

    #return ret

def _max_wave_speed_x(state_vector):
    prim = _primitive_variables(state_vector)

    max_gas = np.abs(prim[1]) + sound_speed
    max_dust = np.abs(prim[5]) + sound_speed_dust

    return np.maximum(max_gas, max_dust)

def _max_wave_speed_y(state_vector):
    prim = _primitive_variables(state_vector)

    max_gas = np.abs(prim[2]) + sound_speed
    max_dust = np.abs(prim[6]) + sound_speed_dust

    return np.maximum(max_gas, max_dust)


def _max_wave_speed_z(state_vector):
    prim = _primitive_variables(state_vector)

    max_gas = np.abs(prim[3]) + sound_speed
    max_dust = np.abs(prim[7]) + sound_speed_dust

    return np.maximum(max_gas, max_dust)

def flux_from_state(state, coords, dim):
    if dim == 0:
        return _flux_from_state_x(state)
    #if dim == 1:
    #    return _flux_from_state_y(state)
    return _flux_from_state_z(state)

def multiply_with_left_eigenvectors(prim, state, dim):
    if dim == 0:
        return _multiply_with_left_eigenvectors_x(prim, state)
    #if dim == 1:
    #    return _multiply_with_left_eigenvectors_y(prim, state)
    return _multiply_with_left_eigenvectors_z(prim, state)

def multiply_with_right_eigenvectors(prim, state, dim):
    if dim == 0:
        return _multiply_with_right_eigenvectors_x(prim, state)
    #if dim == 1:
    #    return _multiply_with_right_eigenvectors_y(prim, state)
    return _multiply_with_right_eigenvectors_z(prim, state)

def max_wave_speed(U , coords, dim):
    if dim == 0:
        return _max_wave_speed_x(U)
    #if dim == 1:
    #    return _max_wave_speed_y(U)
    return _max_wave_speed_z(U)

def source_func(U, coords):
    ret = np.zeros_like(U)

    mu = U[4]/U[0]

    ret[1] = 2*pressure_parameter*U[0] + 2*U[2] + (U[5] - mu*U[1])/stokes
    ret[2] = -0.5*U[1] + (U[6] - mu*U[2])/stokes
    ret[3] = (U[7] - mu*U[3])/stokes

    ret[5] = 2*U[6] - (U[5] - mu*U[1])/stokes
    ret[6] = -0.5*U[5] - (U[6] - mu*U[2])/stokes
    ret[7] = -(U[7] - mu*U[3])/stokes

    return ret

jit_module(nopython=True, error_model="numpy")