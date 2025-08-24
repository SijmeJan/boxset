import numpy as np
from numba import jit_module

def _set_boundary_x(state, x, n_ghost):
    # Periodic boundary conditions
    state[0,0:n_ghost] = state[0,len(x)-2*n_ghost:len(x)-n_ghost]
    state[0,len(x)-n_ghost:len(x)] = state[0,n_ghost:2*n_ghost]
    return state

def set_boundary(state, coords, dim, n_ghost):
    return _set_boundary_x(state, coords[0], n_ghost)

jit_module(nopython=True, error_model="numpy")

def initial_conditions(coords):
    n_eq = 1
    x = coords[0]

    U = np.zeros((n_eq, len(coords[0])))
    #U[0,:] = np.exp(-np.log(2)*(x+0.7)**2/0.0009)*(x > -0.8)*(x < -0.6) +\
    #    1.0*(x > -0.4)*(x < -0.2) +\
    #    (1 - np.abs(10*(x - 0.1)))*(x > 0.0)*(x < 0.2) +\
    #    np.sqrt(np.abs(1 - 100*(x - 0.5)**2))*(x > 0.4)*(x < 0.6)
    U[0,:] = np.sin(np.pi*coords[0])

    return U

from boxset.output.basic import *
from boxset.coords import create_coordinates

def visualise(direc, save_index):
    import matplotlib.pyplot as plt
    import configparser

    config = configparser.ConfigParser()
    config.read(direc + 'scalar_advection.ini')

    n_ghost=4
    t, U = restore_from_dump(save_index, config['Output']['direc'])

    coords = create_coordinates(config)
    x = coords[0]

    #plt.plot(x[n_ghost:-n_ghost], U[0,n_ghost:-n_ghost])

    plt.xlabel('x')

    t, U0 = restore_from_dump(0, config['Output']['direc'])
    plt.plot(x[n_ghost:-n_ghost], U[0,n_ghost:-n_ghost] - U0[0,n_ghost:-n_ghost])
    #plt.plot(x[n_ghost:-n_ghost], U0[0,n_ghost:-n_ghost])

    print('L1 norm: ', np.mean(np.abs(U[0,n_ghost:-n_ghost] + U0[0,n_ghost:-n_ghost])))
    print('L2 norm: ', np.sqrt(np.mean((U[0,n_ghost:-n_ghost] + U0[0,n_ghost:-n_ghost])**2)))
    plt.show()

from boxset.simulation import simulation

simulation("/Users/sjp/Desktop/boxset/scalar_advection.ini", initial_conditions, set_boundary, restore_index=-1)

visualise('/Users/sjp/Desktop/boxset/', 1)
