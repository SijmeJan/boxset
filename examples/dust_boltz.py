import numpy as np
from numba import jit_module

def _set_boundary_x(state, x, n_ghost):
    # Periodic boundary conditions
    state[:,0:n_ghost,:] = state[:,len(x)-2*n_ghost:len(x)-n_ghost,:]
    state[:,len(x)-n_ghost:len(x),:] = state[:,n_ghost:2*n_ghost,:]

    return state

def _set_boundary_y(state, x, n_ghost):
    state[:,:,0:n_ghost] = 0.0
    state[:,:,len(x)-n_ghost:len(x)] = 0.0

    return state

def set_boundary(state, coords, dim, n_ghost):
    if dim == 0:
        return _set_boundary_x(state, coords[0], n_ghost)
    return _set_boundary_y(state, coords[1], n_ghost)

jit_module(nopython=True, error_model="numpy")

def initial_conditions(coords):
    n_eq = 1

    U = np.zeros((n_eq, len(coords[0]), len(coords[1])))

    U[0,0:int(len(coords[0])/2),len(coords[1])-10:] = 1.0
    U[0,int(len(coords[0])/2):,0:10] = 1.0

    return U

from boxset.output.parallel import *
from boxset.coords import create_coordinates

def visualise(direc, save_index):
    import matplotlib.pyplot as plt
    import configparser

    config = configparser.ConfigParser()
    config.read(direc + 'dust_boltz.ini')

    n_ghost = np.int32(config['Grid']['n_ghost'])

    coords, pos, global_dims = create_coordinates(config)
    x = coords[0]
    y = coords[1]

    state = initial_conditions(coords)

    t, U = restore_from_dump(state, save_index, config['Output']['direc'], pos, global_dims, n_ghost)


    f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)

    cf = ax1.contourf(x[n_ghost:-n_ghost], y[n_ghost:-n_ghost], np.transpose(U[0,n_ghost:-n_ghost,n_ghost:-n_ghost]),
                     levels=100, cmap='terrain')

    #ax1.set_xlabel('x')
    #ax1.set_ylabel('v')
    #plt.colorbar(cf, ax=ax1)

    dens = np.sum(U[0,n_ghost:-n_ghost,n_ghost:-n_ghost], axis=1)
    momx = np.sum(U[0,n_ghost:-n_ghost,n_ghost:-n_ghost]*y[n_ghost:-n_ghost], axis=1)

    ax2.plot(x[n_ghost:-n_ghost], dens)
    ax3.plot(x[n_ghost:-n_ghost], momx)


    plt.show()

from boxset.simulation import simulation

#simulation("/Users/sjp/Desktop/boxset/dust_boltz.ini", initial_conditions, set_boundary, restore_index=-1)

visualise('/Users/sjp/Desktop/boxset/', 8)
