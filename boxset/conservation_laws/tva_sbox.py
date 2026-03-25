import numpy as np
from numba import jit_module

eta = 0.05
shear_param = 1.5
dust_diff = 1.0e-6
stokes = 1.0e-2

def _primitive_variables(conserved_variables):
    prim = np.zeros_like(conserved_variables)
    prim[0] = conserved_variables[0]
    prim[1] = conserved_variables[1]/prim[0]
    prim[2] = conserved_variables[2]/prim[0]
    prim[3] = conserved_variables[3]/prim[0]
    prim[4] = conserved_variables[4]

    return prim


def _flux_from_state_x(state_vector):
    prim = _primitive_variables(state_vector)

    flx = np.zeros_like(state_vector)
    flx[0] = state_vector[1]
    flx[1] = state_vector[1]*prim[1] + prim[4]
    flx[2] = state_vector[1]*prim[2]
    flx[3] = state_vector[1]*prim[3]
    flx[4] = state_vector[4]*prim[1]

    return flx


def _flux_from_state_y(state_vector, t):
    prim = _primitive_variables(state_vector)

    flx = np.zeros_like(state_vector)
    flx[0] = state_vector[2]
    flx[1] = state_vector[2]*prim[1]
    flx[2] = state_vector[2]*prim[2] + prim[4]
    flx[3] = state_vector[2]*prim[3]
    flx[4] = state_vector[4]*prim[2]

    return flx + shear_param*(t % (2/shear_param))*_flux_from_state_x(state_vector)


def _flux_from_state_z(state_vector):
    prim = _primitive_variables(state_vector)

    flx = np.zeros_like(state_vector)
    flx[0] = state_vector[3]
    flx[1] = state_vector[3]*prim[1]
    flx[2] = state_vector[3]*prim[2]
    flx[3] = state_vector[3]*prim[3] + prim[4]
    flx[4] = state_vector[4]*prim[3]

    return flx


def _multiply_with_left_eigenvectors_x(primitive_variables, state_vector):
    '''Multiply state_vector with left eigenvectors
    based on primitive_variables'''
    return state_vector


def _multiply_with_left_eigenvectors_y(primitive_variables, state_vector, time):
    '''Multiply state_vector with left eigenvectors
    based on primitive_variables'''
    return state_vector

def _multiply_with_left_eigenvectors_z(primitive_variables, state_vector):
    '''Multiply state_vector with left eigenvectors
    based on primitive_variables'''
    return state_vector

def _multiply_with_right_eigenvectors_x(primitive_variables, state_vector):
    '''Multiply state_vector with right eigenvectors
    based on primitive_variables'''
    return state_vector

def _multiply_with_right_eigenvectors_y(primitive_variables, state_vector, time):
    '''Multiply state_vector with right eigenvectors
    based on primitive_variables'''
    return state_vector

def _multiply_with_right_eigenvectors_z(primitive_variables, state_vector):
    '''Multiply state_vector with right eigenvectors
    based on primitive_variables'''
    return state_vector

def _max_wave_speed_x(state_vector):
    prim = _primitive_variables(state_vector)
    return np.abs(prim[1]) + np.sqrt(prim[4]/prim[0])


def _max_wave_speed_y(state_vector, t):
    prim = _primitive_variables(state_vector)

    st = shear_param*(t % (2/shear_param))
    eta = np.sqrt(1 + st*st)

    return np.abs(prim[2] + st*prim[1]) + np.sqrt(prim[4]/prim[0])*eta


def _max_wave_speed_z(state_vector):
    prim = _primitive_variables(state_vector)
    return np.abs(prim[3]) + np.sqrt(prim[4]/prim[0])


def flux_from_state(state, coords, time, dim):
    if dim == 0:
        return _flux_from_state_x(state)
    #if dim == 1:
    #    return _flux_from_state_y(state, time)
    return _flux_from_state_z(state)


def multiply_with_left_eigenvectors(prim, state, time, dim):
    if dim == 0:
        return _multiply_with_left_eigenvectors_x(prim, state)
    #if dim == 1:
    #    return _multiply_with_left_eigenvectors_y(prim, state, time)
    return _multiply_with_left_eigenvectors_z(prim, state)


def multiply_with_right_eigenvectors(prim, state, time, dim):
    if dim == 0:
        return _multiply_with_right_eigenvectors_x(prim, state)
    #if dim == 1:
    #    return _multiply_with_right_eigenvectors_y(prim, state, time)
    return _multiply_with_right_eigenvectors_z(prim, state)


def max_wave_speed(U, coords, time, dim):
    if dim == 0:
        return _max_wave_speed_x(U)
    #if dim == 1:
    #    return _max_wave_speed_y(U, time)
    return _max_wave_speed_z(U)


def source_func(U, coords, time):
    ret = np.zeros_like(U)
    x = coords[0]
    dx = x[1] - x[0]
    z = coords[-1]
    dz = z[1] - z[0]
    diff = U[4,...]*np.abs(1 - U[4,...]/U[0,...])/U[0,...]

    for i in range(1, len(z)-1):
        ret[0, ..., i] += dust_diff*(0.5*(U[4, ..., i] + U[4,...,i+1])*(U[0,...,i+1]/U[4,...,i+1] - U[0,...,i]/U[4,...,i]) - \
                                     0.5*(U[4, ..., i] + U[4,...,i-1])*(U[0,...,i]/U[4,...,i] - U[0,...,i-1]/U[4,...,i-1]))/dz**2
        ret[1, ..., i] += 2*eta*U[4, ..., i] + 2*U[2, ..., i]
        ret[2, ..., i] += (shear_param - 2)*U[1, ..., i]
        ret[3, ..., i] -= U[0, ..., i]*z[i]
        ret[4, ..., i] += stokes*(0.5*(diff[..., i] + diff[...,i+1])*(U[4,...,i+1] - U[4,...,i]) - \
                                  0.5*(diff[..., i] + diff[...,i-1])*(U[4,...,i] - U[4,...,i-1]))/dz**2

    for i in range(1, len(x)-1):
        ret[0,i,...] += dust_diff*(0.5*(U[4,i,...] + U[4,i+1,...])*(U[0,i+1,...]/U[4,i+1,...] - U[0,i,...]/U[4,i,...]) - \
                                     0.5*(U[4,i,...] + U[4,i-1,...])*(U[0,i,...]/U[4,i,...] - U[0,i-1,...]/U[4,i-1,...]))/dx**2
        ret[4,i,...] += (stokes*(0.5*(diff[i,...] + diff[i+1,...])*(U[4,i+1,...] - U[4,i,...]) - \
                                  0.5*(diff[i,...] + diff[i-1,...])*(U[4,i,...] - U[4,i-1,...]))/dx**2 -\
                          stokes*eta*(diff[i+1,...]*U[4,i+1,...] - diff[i-1,...]*U[4,i-1,...])/dx)

    return ret


def allowed_state(state):
    prim = _primitive_variables(state)

    # Want density and pressure positive
    return np.logical_and(prim[0] > 0.0, prim[4] > 0.0)


jit_module(nopython=True, error_model="numpy")
