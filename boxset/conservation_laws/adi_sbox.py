import numpy as np
from numba import jit_module

gamma = 1.4
shear_param = 1.5
g0 = 1.0

def _primitive_variables(conserved_variables):
    prim = np.zeros_like(conserved_variables)
    prim[0] = conserved_variables[0]
    prim[1] = conserved_variables[1]/prim[0]
    prim[2] = conserved_variables[2]/prim[0]
    prim[3] = conserved_variables[3]/prim[0]
    prim[4] = (gamma - 1.0)*(conserved_variables[4] -
                             0.5*(conserved_variables[1]**2 +
                                  conserved_variables[2]**2 +
                                  conserved_variables[3]**2)/prim[0])

    return prim


def _flux_from_state_x(state_vector):
    prim = _primitive_variables(state_vector)

    flx = np.zeros_like(state_vector)
    flx[0] = state_vector[1]
    flx[1] = state_vector[1]*prim[1] + prim[4]
    flx[2] = state_vector[1]*prim[2]
    flx[3] = state_vector[1]*prim[3]
    flx[4] = (state_vector[4] + prim[4])*prim[1]

    return flx


def _flux_from_state_y(state_vector, t):
    prim = _primitive_variables(state_vector)

    flx = np.zeros_like(state_vector)
    flx[0] = state_vector[2]
    flx[1] = state_vector[2]*prim[1]
    flx[2] = state_vector[2]*prim[2] + prim[4]
    flx[3] = state_vector[2]*prim[3]
    flx[4] = (state_vector[4] + prim[4])*prim[2]

    return flx + shear_param*(t % (2/shear_param))*_flux_from_state_x(state_vector)


def _flux_from_state_z(state_vector):
    prim = _primitive_variables(state_vector)

    flx = np.zeros_like(state_vector)
    flx[0] = state_vector[3]
    flx[1] = state_vector[3]*prim[1]
    flx[2] = state_vector[3]*prim[2]
    flx[3] = state_vector[3]*prim[3] + prim[4]
    flx[4] = (state_vector[4] + prim[4])*prim[3]

    return flx


def _multiply_with_left_eigenvectors_x(primitive_variables, state_vector):
    '''Multiply state_vector with left eigenvectors
    based on primitive_variables'''
    return state_vector

    prim = _primitive_variables(primitive_variables)

    # Sound speed
    c = np.sqrt(gamma*prim[4]/prim[0])

    b1 = (gamma-1)/(c*c)
    b2 = 0.5*b1*(prim[1]**2 + prim[2]**2 + prim[3]**2)

    ret = np.zeros_like(state_vector)

    ret[0] = \
        0.5*(b2 + prim[1]/c)*state_vector[0] \
        - 0.5*(b1*prim[1] + 1/c)*state_vector[1] \
        - 0.5*b1*prim[2]*state_vector[2] \
        - 0.5*b1*prim[3]*state_vector[3] \
        + 0.5*b1*state_vector[4]
    ret[1] = \
        (1 - b2)*state_vector[0] \
        + b1*prim[1]*state_vector[1] \
        + b1*prim[2]*state_vector[2] \
        + b1*prim[3]*state_vector[3] \
        - b1*state_vector[4]
    ret[2] = state_vector[2] - prim[2]*state_vector[0]
    ret[3] = state_vector[3] - prim[3]*state_vector[0]
    ret[4] = \
        0.5*(b2 - prim[1]/c)*state_vector[0] \
        - 0.5*(b1*prim[1] - 1/c)*state_vector[1] \
        - 0.5*b1*prim[2]*state_vector[2] \
        - 0.5*b1*prim[3]*state_vector[3] \
        + 0.5*b1*state_vector[4]

    return ret


def _multiply_with_left_eigenvectors_y(primitive_variables, state_vector):
    '''Multiply state_vector with left eigenvectors
    based on primitive_variables'''
    return state_vector

    prim = _primitive_variables(primitive_variables)

    # Sound speed
    c = np.sqrt(gamma*prim[4]/prim[0])

    b1 = (gamma-1)/(c*c)
    b2 = 0.5*b1*(prim[1]**2 + prim[2]**2 + prim[3]**2)

    ret = np.zeros_like(state_vector)

    ret[0] = \
        0.5*(b2 + prim[2]/c)*state_vector[0] \
        - 0.5*b1*prim[1]*state_vector[1] \
        - 0.5*(b1*prim[2] + 1/c)*state_vector[2] \
        - 0.5*b1*prim[3]*state_vector[3] \
        + 0.5*b1*state_vector[4]
    ret[1] = state_vector[1] - prim[1]*state_vector[0]
    ret[2] = \
        (1 - b2)*state_vector[0] \
        + b1*prim[1]*state_vector[1] \
        + b1*prim[2]*state_vector[2] \
        + b1*prim[3]*state_vector[3] \
        - b1*state_vector[4]
    ret[3] = state_vector[3] - prim[3]*state_vector[0]
    ret[4] = \
        0.5*(b2 - prim[2]/c)*state_vector[0] \
        - 0.5*b1*prim[1]*state_vector[1] \
        - 0.5*(b1*prim[2] - 1/c)*state_vector[2] \
        - 0.5*b1*prim[3]*state_vector[3] \
        + 0.5*b1*state_vector[4]

    return ret


def _multiply_with_left_eigenvectors_z(primitive_variables, state_vector):
    '''Multiply state_vector with left eigenvectors
    based on primitive_variables'''
    return state_vector

    prim = _primitive_variables(primitive_variables)

    # Sound speed
    c = np.sqrt(gamma*prim[4]/prim[0])

    b1 = (gamma-1)/(c*c)
    b2 = 0.5*b1*(prim[1]**2 + prim[2]**2 + prim[3]**2)

    ret = np.zeros_like(state_vector)

    ret[0] = \
        0.5*(b2 + prim[3]/c)*state_vector[0] \
        - 0.5*b1*prim[1]*state_vector[1] \
        - 0.5*b1*prim[2]*state_vector[2] \
        - 0.5*(b1*prim[3] + 1/c)*state_vector[3] \
        + 0.5*b1*state_vector[4]
    ret[1] = state_vector[1] - prim[1]*state_vector[0]
    ret[2] = state_vector[2] - prim[2]*state_vector[0]
    ret[3] = \
        (1 - b2)*state_vector[0] \
        + b1*prim[1]*state_vector[1] \
        + b1*prim[2]*state_vector[2] \
        + b1*prim[3]*state_vector[3] \
        - b1*state_vector[4]
    ret[4] = \
        0.5*(b2 - prim[3]/c)*state_vector[0] \
        - 0.5*b1*prim[1]*state_vector[1] \
        - 0.5*b1*prim[2]*state_vector[2] \
        - 0.5*(b1*prim[3] - 1/c)*state_vector[3] \
        + 0.5*b1*state_vector[4]

    return ret


def _multiply_with_right_eigenvectors_x(primitive_variables, state_vector):
    '''Multiply state_vector with right eigenvectors
    based on primitive_variables'''
    return state_vector

    prim = _primitive_variables(primitive_variables)

    ekin = 0.5*(prim[1]**2 + prim[2]**2 + prim[3]**2)

    # Enthalpy and sound speed
    h = ekin + gamma*prim[4]/(gamma-1)/prim[0]
    c = np.sqrt(gamma*prim[4]/prim[0])

    ret = np.zeros_like(state_vector)

    ret[0] = state_vector[0] + state_vector[1] + state_vector[4]
    ret[1] = (prim[1] - c)*state_vector[0] + \
        prim[1]*state_vector[1] + \
        (prim[1] + c)*state_vector[4]
    ret[2] = prim[2]*ret[0] + state_vector[2]
    ret[3] = prim[3]*ret[0] + state_vector[3]
    ret[4] = (h - c*prim[1])*state_vector[0] + ekin*state_vector[1] + \
        prim[2]*state_vector[2] + prim[3]*state_vector[3] + \
        (h + c*prim[1])*state_vector[4]

    return ret


def _multiply_with_right_eigenvectors_y(primitive_variables, state_vector):
    '''Multiply state_vector with right eigenvectors
    based on primitive_variables'''
    return state_vector

    prim = _primitive_variables(primitive_variables)

    ekin = 0.5*(prim[1]**2 + prim[2]**2 + prim[3]**2)

    # Enthalpy and sound speed
    h = ekin + gamma*prim[4]/(gamma-1)/prim[0]
    c = np.sqrt(gamma*prim[4]/prim[0])

    ret = np.zeros_like(state_vector)

    ret[0] = state_vector[0] + state_vector[2] + state_vector[4]
    ret[1] = prim[1]*ret[0] + state_vector[1]
    ret[2] = (prim[2] - c)*state_vector[0] + prim[2]*state_vector[2] + \
        (prim[2] + c)*state_vector[4]
    ret[3] = prim[3]*ret[0] + state_vector[3]
    ret[4] = (h - c*prim[2])*state_vector[0] + ekin*state_vector[2] + \
        prim[1]*state_vector[1] + prim[3]*state_vector[3] + \
        (h + c*prim[2])*state_vector[4]

    return ret


def _multiply_with_right_eigenvectors_z(primitive_variables, state_vector):
    '''Multiply state_vector with right eigenvectors
    based on primitive_variables'''
    return state_vector

    prim = _primitive_variables(primitive_variables)

    ekin = 0.5*(prim[1]**2 + prim[2]**2 + prim[3]**2)

    # Enthalpy and sound speed
    h = ekin + gamma*prim[4]/(gamma-1)/prim[0]
    c = np.sqrt(gamma*prim[4]/prim[0])

    ret = np.zeros_like(state_vector)

    ret[0] = state_vector[0] + state_vector[3] + state_vector[4]
    ret[1] = prim[1]*ret[0] + state_vector[1]
    ret[2] = prim[2]*ret[0] + state_vector[2]
    ret[3] = (prim[3] - c)*state_vector[0] + prim[3]*state_vector[3] + \
        (prim[3] + c)*state_vector[4]
    ret[4] = (h - c*prim[3])*state_vector[0] + ekin*state_vector[3] + \
        prim[1]*state_vector[1] + prim[2]*state_vector[2] + \
        (h + c*prim[3])*state_vector[4]

    return ret


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
    if dim == 1:
        return _flux_from_state_y(state, time)
    return _flux_from_state_z(state)


def multiply_with_left_eigenvectors(prim, state, time, dim):
    if dim == 0:
        return _multiply_with_left_eigenvectors_x(prim, state)
    if dim == 1:
        return _multiply_with_left_eigenvectors_y(prim, state)
    return _multiply_with_left_eigenvectors_z(prim, state)


def multiply_with_right_eigenvectors(prim, state, time, dim):
    if dim == 0:
        return _multiply_with_right_eigenvectors_x(prim, state)
    if dim == 1:
        return _multiply_with_right_eigenvectors_y(prim, state)
    return _multiply_with_right_eigenvectors_z(prim, state)


def max_wave_speed(U, coords, time, dim):
    if dim == 0:
        return _max_wave_speed_x(U)
    if dim == 1:
        return _max_wave_speed_y(U, time)
    return _max_wave_speed_z(U)


def source_func(U, coords, time):
    ret = np.zeros_like(U)
    z = coords[-1]

    for i in range(0, len(z)):
        ret[1, ..., i] = 2*U[2, ..., i]
        ret[2, ..., i] = (shear_param - 2)*U[1, ..., i]
        ret[3, ..., i] = -g0*U[0, ..., i]     # Constant vertical acceleration a la Marcus+15
        ret[4, ..., i] = shear_param*U[1, ..., i]*U[2, ..., i]/U[0, ..., i] -\
            U[3, ..., i]*g0
    return ret


def allowed_state(state):
    prim = _primitive_variables(state)

    # Want density and pressure positive
    return np.logical_and(prim[0] > 0.0, prim[4] > 0.0)


jit_module(nopython=True, error_model="numpy")
