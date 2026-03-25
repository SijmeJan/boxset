import numpy as np
from numba import jit_module
from mpi4py import MPI

from boxset.conservation_laws.iso_2d_dust import stokes, pressure_parameter, metallicity

def user_source_func(U, coords, time):
    return np.zeros_like(U)

def _set_boundary_x(state, x, n_ghost):
    # Periodic boundary conditions
    state[:,0:n_ghost,:] = state[:,len(x)-2*n_ghost:len(x)-n_ghost,:]
    state[:,len(x)-n_ghost:len(x),:] = state[:,n_ghost:2*n_ghost,:]

    return state

def _set_boundary_z(state, x, n_ghost):
    # Periodic boundary conditions
    #state[:,:,0:n_ghost] = state[:,:,len(x)-2*n_ghost:len(x)-n_ghost]
    #state[:,:,len(x)-n_ghost:len(x)] = state[:,:,n_ghost:2*n_ghost]

    # Nonreflecting
    eta = pressure_parameter

    state[0,:,0:n_ghost] = np.exp(-0.5*x[0:n_ghost]**2)
    state[1,:,0:n_ghost] = 0.0
    state[2,:,0:n_ghost] = -eta*state[0,:,0:n_ghost]
    state[3,:,0:n_ghost] = 0.0
    state[4,:,0:n_ghost] = \
      2*metallicity*np.exp(-2.0*x[0:n_ghost]**2/eta**2)/eta
    state[5,:,0:n_ghost] = state[4,:,0:n_ghost]*(-2*stokes*eta/(1 + stokes**2))
    state[6,:,0:n_ghost] = -eta*state[4,:,0:n_ghost]
    state[7,:,0:n_ghost] = 0.0

    state[0,:,len(x)-n_ghost:len(x)] = np.exp(-0.5*x[len(x)-n_ghost:len(x)]**2)
    state[1,:,len(x)-n_ghost:len(x)] = 0.0
    state[2,:,len(x)-n_ghost:len(x)] = -eta*state[0,:,len(x)-n_ghost:len(x)]
    state[3,:,len(x)-n_ghost:len(x)] = 0.0
    state[4,:,len(x)-n_ghost:len(x)] = \
      2*metallicity*np.exp(-2.0*x[len(x)-n_ghost:len(x)]**2/eta**2)/eta
    state[5,:,len(x)-n_ghost:len(x)] = \
      state[4,:,len(x)-n_ghost:len(x)]*(-2*stokes*eta/(1 + stokes**2))
    state[6,:,len(x)-n_ghost:len(x)] = -eta*state[4,:,len(x)-n_ghost:len(x)]
    state[7,:,len(x)-n_ghost:len(x)] = 0.0


    return state

def set_boundary(state, coords, dim, n_ghost):
    if dim == 0:
        return _set_boundary_x(state, coords[0], n_ghost)
    return _set_boundary_z(state, coords[1], n_ghost)

jit_module(nopython=True, error_model="numpy")

def initial_conditions(coords, config):
    n_eq = 8
    #pressure_parameter = 0.05
    #metallicity = 0.01
    #stokes = 1.0
    noise_level = 0.01

    h = 0.5*noise_level
    l = -0.5*noise_level

    rng = np.random.default_rng(seed=1+MPI.COMM_WORLD.Get_rank())

    z = coords[-1]
    U = np.zeros((n_eq, len(coords[0]), len(coords[1])))

    for i in range(0, len(z)):
        U[0, ..., i] = np.exp(-0.5*z[i]*z[i])
    U[1,:,:] = U[0,:,:]*rng.uniform(low=l, high=h, size=np.shape(U[1,:,:]))
    U[2,:,:] = U[0,:,:]*(-pressure_parameter + \
      rng.uniform(low=l, high=h, size=np.shape(U[1,:,:])))
    U[3,:,:] = U[0,:,:]*rng.uniform(low=l, high=h, size=np.shape(U[1,:,:]))

    for i in range(0, len(z)):
        U[4, ..., i] = \
          2*metallicity*np.exp(-2.0*z[i]*z[i]/pressure_parameter**2)/pressure_parameter
    U[5,:,:] = U[4,:,:]*(-2*stokes*pressure_parameter/(1 + stokes**2) + \
      rng.uniform(low=l, high=h, size=np.shape(U[1,:,:])))
    U[6,:,:] = U[4,:,:]*(-pressure_parameter/(1 + stokes**2) + \
      rng.uniform(low=l, high=h, size=np.shape(U[1,:,:])))
    U[7,:,:] = U[4,:,:]*rng.uniform(low=l, high=h, size=np.shape(U[1,:,:]))

    return U

from boxset.output.parallel import *
from boxset.coords import create_coordinates

def visualise(ini_file, save_index):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import configparser


    config = configparser.ConfigParser()
    config.read(ini_file)

    n_ghost = np.int32(config['Grid']['n_ghost'])

    coords, pos, global_dims, periodic_flags = create_coordinates(config)
    x = coords[0]
    y = coords[1]

    state = initial_conditions(coords, config)

    t, U = restore_from_dump(state, save_index, config['Output']['direc'], pos, global_dims, n_ghost)

    f = U[4,n_ghost:-n_ghost,n_ghost:-n_ghost]
    #levels = np.linspace(np.log10(0.02), np.log10(2), 100)

    #cf = plt.contourf(x[n_ghost:-n_ghost], y[n_ghost:-n_ghost],
    #                  np.transpose(np.log10(f)),
    #                  levels=100, cmap='plasma')
    cf = plt.contourf(x[n_ghost:-n_ghost], y[n_ghost:-n_ghost],
                      np.transpose(f),
                      levels=100, cmap='plasma')

    plt.xlabel('x')
    plt.ylabel('z')
    plt.colorbar()

    plt.show()


from boxset.simulation import simulation

#simulation("/Users/sjp/Desktop/boxset/stratified_si.ini", initial_conditions, set_boundary, user_source_func, restore_index=-1)
#simulation("/home/sijmejanpaarde/streaming_instability.ini", initial_conditions, set_boundary, restore_index=-1)

visualise("/Users/sjp/Desktop/boxset/stratified_si.ini", 62)
