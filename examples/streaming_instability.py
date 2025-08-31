import numpy as np
from numba import jit_module
from mpi4py import MPI

def _set_boundary_x(state, x, n_ghost):
    # Periodic boundary conditions
    state[:,0:n_ghost,:] = state[:,len(x)-2*n_ghost:len(x)-n_ghost,:]
    state[:,len(x)-n_ghost:len(x),:] = state[:,n_ghost:2*n_ghost,:]

    return state

def _set_boundary_z(state, x, n_ghost):
    # Periodic boundary conditions
    state[:,:,0:n_ghost] = state[:,:,len(x)-2*n_ghost:len(x)-n_ghost]
    state[:,:,len(x)-n_ghost:len(x)] = state[:,:,n_ghost:2*n_ghost]

    return state

def set_boundary(state, coords, dim, n_ghost):
    if dim == 0:
        return _set_boundary_x(state, coords[0], n_ghost)
    return _set_boundary_z(state, coords[1], n_ghost)

jit_module(nopython=True, error_model="numpy")

def initial_conditions(coords):
    n_eq = 8
    pressure_parameter = 0.05
    mu = 0.2
    stokes = 1.0
    noise_level = 0.01

    h = 0.5*noise_level
    l = -0.5*noise_level

    rng = np.random.default_rng(seed=1+MPI.COMM_WORLD.Get_rank())

    J0 = mu/(1 + stokes**2)
    J1 = stokes*J0
    denom = 1/((1 + J0)**2 + J1*J1)

    U = np.zeros((n_eq, len(coords[0]), len(coords[1])))
    U[0,:,:] = 1.0
    U[1,:,:] = 2*pressure_parameter*J1*denom + \
      rng.uniform(low=l, high=h, size=np.shape(U[1,:,:]))
    U[2,:,:] = -pressure_parameter*(1 + J0)*denom + \
      rng.uniform(low=l, high=h, size=np.shape(U[1,:,:]))
    U[3,:,:] = rng.uniform(low=l, high=h, size=np.shape(U[1,:,:]))

    U[4,:,:] = mu
    U[5,:,:] = 2*pressure_parameter*(J1 - stokes*(1 + J0))*denom/(1 + stokes**2) + rng.uniform(low=l, high=h, size=np.shape(U[1,:,:]))
    U[6,:,:] = -pressure_parameter*(1 + J0 + stokes*J1)*denom/(1 + stokes**2)+ rng.uniform(low=l, high=h, size=np.shape(U[1,:,:]))
    U[7,:,:] = rng.uniform(low=l, high=h, size=np.shape(U[1,:,:]))

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

    coords, pos, global_dims = create_coordinates(config)
    x = coords[0]
    y = coords[1]

    state = initial_conditions(coords)

    t, U = restore_from_dump(state, save_index, config['Output']['direc'], pos, global_dims, n_ghost)

    f = U[4,n_ghost:-n_ghost,n_ghost:-n_ghost]
    #levels = np.linspace(np.log10(0.02), np.log10(2), 100)

    cf = plt.contourf(x[n_ghost:-n_ghost], y[n_ghost:-n_ghost],
                      np.transpose(np.log10(f)),
                      levels=100, cmap='plasma')

    plt.xlabel('x')
    plt.ylabel('z')
    plt.colorbar()

    plt.show()


from boxset.simulation import simulation

#simulation("/Users/sjp/Desktop/boxset/streaming_instability.ini", initial_conditions, set_boundary, restore_index=-1)
#simulation("/home/sijmejanpaarde/streaming_instability.ini", initial_conditions, set_boundary, restore_index=-1)

visualise("/Users/sjp/Desktop/boxset/streaming_instability.ini", 100)
