import numpy as np
from numba import jit

from .reconstruction.cweno_3rd import calc_interface_flux, weno_r
from .conservation_laws.iso_2d_dust import *
#from boundary_conditions.streaming_instability import set_boundary

@jit
def my_atleast_1d(x):
    if isinstance(x, float):
        return np.array([x])
    return np.atleast_1d(x)

@jit
def min(arr):
    minval = arr.flatten()[0]
    for e in arr.flatten():
        if e < minval: minval = e
    return minval

@jit
def calc_time_step(state, coords, n_ghost, boundary_conditions):
    '''Calculate allowed time step based on maximum wave speeds'''
    dt = 1.0e10

    for i in range(0, len(coords)):
        #set_boundary(state, coords, i, n_ghost)
        state = boundary_conditions(state, coords, i, n_ghost)

    # Loop over all space dimensions
    for i in range(0, len(coords)):
        x = coords[i]
        dt = min(np.asarray([dt, np.min((x[1] - x[0])/max_wave_speed(state, coords, i))]))

    return dt

@jit
def split_flux(state, a, flux):
    '''
    Lax-Friedrichs flux splitting based on state, flux, and wave speed a.
    '''
    # If we are integrating in the x direction,
    # state and flux will have shape (n_eq, ny, nz, 5) if we are using 5th order weno.
    # On the other hand, a will have shape (ny, nz).
    ret = 0*flux
    for i in range(0, 2*weno_r-1):
        ret[...,i] = flux[...,i] + a*state[...,i]
    return ret

@jit
def total_interface_flux(Fplus, Fmin):
    '''Add positive and negative contributions of the flux.'''
    return 0.5*(calc_interface_flux(Fplus, weno_r, epsilon=1.0e-12) +
                calc_interface_flux(Fmin, weno_r-1, epsilon=1.0e-12))

@jit
def calculate_interface_flux(U, centre_flux, a, dim, x):
    '''Calculate interface fluxes in one particular direction.
    This is where the bulk of the computational time is spent, usually.
    '''
    interface_flux = np.empty_like(U)

    for i in range(weno_r-1, len(x)-weno_r):
        # Calculate flux splitting: U and centre_flux provide full stencil
        Fplus = split_flux(U[...,i-weno_r+1:i+weno_r], a[...,i], centre_flux[...,i-weno_r+1:i+weno_r])
        Fmin  = split_flux(U[...,i-weno_r+2:i+weno_r+1], -a[...,i], centre_flux[...,i-weno_r+2:i+weno_r+1])

        # Multiply with left eigenvectors of cell i (only for systems)
        for j in range(0, 2*weno_r-1):
            Fplus[...,j] = multiply_with_left_eigenvectors(U[...,i], Fplus[...,j], dim)
            Fmin[...,j]  = multiply_with_left_eigenvectors(U[...,i], Fmin[...,j], dim)

        # Add positive and negative portions, so that interface_flux[i] = flux_i+1/2
        interface_flux[...,i] = total_interface_flux(Fplus, Fmin)

        # Multiply with right eigenvectors of cell i+2 (only for systems)
        interface_flux[...,i] = multiply_with_right_eigenvectors(U[...,i], interface_flux[...,i], dim)

    return interface_flux

@jit
def add_to_rhs(rhs, U, coords, dim):
    x = coords[dim]

    # Calculate fluxes at cell centres. Force a copy of U in case the flux_from_state function returns a view.
    centre_flux = flux_from_state(np.copy(U), coords, dim)

    # a[..,i] = maximum wave speed associated with interface i+1/2
    a = max_wave_speed(U, coords, dim)
    for i in range(1, len(x)-1):
        # Make sure these are arrays even for scalar states
        a0 = my_atleast_1d(a[...,i]).flatten()
        a1 = my_atleast_1d(a[...,i-1]).flatten()

        for j in range(0, len(a0)):
            if a0[j] > a1[j]: a1[j] = a0[j]

        a[...,i] = np.reshape(a1, np.shape(a[...,i]))

    interface_flux = calculate_interface_flux(U, centre_flux, a, dim, x)

    # Add flux difference to rhs
    for i in range(1, len(x)):
        rhs[...,i] = rhs[...,i] - (interface_flux[...,i] - interface_flux[...,i-1])/(x[1]-x[0])

    return rhs

def calculate_rhs(state, coords, n_ghost, boundary_conditions):
    '''
    Calculate the right-hand side for the method of lines, based on state and coordinates.
    The state should have shape (n_eq, len(dim1), len(dim2), ...)
    The coordinates should be a list with n_dim entries 'dim1', 'dim2', ...
    Conservation law functions are imported from the relevant file.
    '''
    # Return array: dU/dt = rhs
    rhs = np.zeros_like(state)

    # Loop over all space dimensions
    for dim in range(0, len(coords)):
        # Set boundary conditions
        #state = set_boundary(state, coords, dim, n_ghost)
        state = boundary_conditions(state, coords, dim, n_ghost)

        # Swap dimension so that state shape is (n_state, dim1, dim2, ..., dim)
        state = np.swapaxes(state, dim+1, len(coords))
        rhs = np.swapaxes(rhs, dim+1, len(coords))

        # Add contribution from this dimesnsion to rhs
        rhs = add_to_rhs(rhs, state, coords, dim)

        # Swap back to original shape
        state = np.swapaxes(state, dim+1, len(coords))
        rhs = np.swapaxes(rhs, dim+1, len(coords))

    # Add source term
    return rhs + source_func(state, coords)


