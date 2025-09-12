import numpy as np
from numba import jit_module
from mpi4py import MPI

def _set_boundary_x(state, x, n_ghost):
    state[0,0:n_ghost] = 1.0
    state[1,0:n_ghost] = 0.0
    state[2,0:n_ghost] = 1.0/(1.4 - 1.0)

    state[0,len(x)-n_ghost:len(x)] = 0.125
    state[1,len(x)-n_ghost:len(x)] = 0.0
    state[2,len(x)-n_ghost:len(x)] = 0.1/(1.4 - 1.0)

    return state

def set_boundary(state, coords, dim, n_ghost):
    return _set_boundary_x(state, coords[0], n_ghost)

jit_module(nopython=True, error_model="numpy")

def initial_conditions(coords):
    n_eq = 3
    x = coords[0]

    U = np.zeros((n_eq, len(x)))

    U[0,:] = 1.0
    U[1,:] = 0.0
    U[2,:] = 1.0/(1.4 - 1.0)

    for i in range(0, len(x)):
        if x[i] > 0.5:
            U[0,i] = 0.125
            U[2,i] = 0.1/(1.4 - 1.0)

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

    state = initial_conditions(coords)

    t, U = restore_from_dump(state, save_index, config['Output']['direc'], pos, global_dims, n_ghost)

    plt.plot(x, U[0,:])
    plt.xlabel('x')

    plt.show()


from boxset.simulation import simulation

simulation("/Users/sjp/Desktop/boxset/sod.ini", initial_conditions, set_boundary, restore_index=-1)

visualise("/Users/sjp/Desktop/boxset/sod.ini", 1)
