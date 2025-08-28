import numpy as np
from numba import jit_module

def _set_boundary_x(state, x, y, n_ghost):
    # x and y edge of step
    xpos = np.argmax(x > 0.6)
    ypos = np.argmax(y > 0.2)

    # Reset everything under the step
    state[0,xpos:,0:ypos] = 1.4
    state[1,xpos:,0:ypos] = 0.0
    state[2,xpos:,0:ypos] = 0.0
    state[3,xpos:,0:ypos] = 1/(1.4-1)

    # Left and right boundary outflow/inflow
    state[:,0:n_ghost,:] = state[:,2*n_ghost:n_ghost:-1,:]
    state[:,len(x)-n_ghost:len(x),:] = state[:,len(x)-n_ghost-1:len(x)-2*n_ghost-1:-1,:]

    # Left step boundary: reflect
    state[:,xpos:xpos+n_ghost,0:ypos] = state[:,xpos-1:xpos-n_ghost-1:-1,0:ypos]
    state[1,xpos:xpos+n_ghost,0:ypos] =-state[1,xpos-1:xpos-n_ghost-1:-1,0:ypos]

    return state

def _set_boundary_y(state, x, y, n_ghost):
    xpos = np.argmax(x > 0.6)
    ypos = np.argmax(y > 0.2)

    # Reset everything under the step
    state[0,xpos:,0:ypos] = 1.4
    state[1,xpos:,0:ypos] = 0.0
    state[2,xpos:,0:ypos] = 0.0
    state[3,xpos:,0:ypos] = 1/(1.4-1)

    # Top boundary: reflect
    state[:,:,len(y)-n_ghost:len(y)] =  state[:,:,len(y)-n_ghost-1:len(y)-2*n_ghost-1:-1]
    state[2,:,len(y)-n_ghost:len(y)] = -state[2,:,len(y)-n_ghost-1:len(y)-2*n_ghost-1:-1]

    # Bottom boundary left from step
    state[:,0:xpos,0:n_ghost] = state[:,0:xpos,2*n_ghost-1:n_ghost-1:-1]
    state[2,0:xpos,0:n_ghost] = -state[2,0:xpos,2*n_ghost-1:n_ghost-1:-1]

    # Bottom boundary right from step
    state[:,xpos:,ypos-n_ghost:ypos] = state[:,xpos:,ypos+n_ghost-1:ypos-1:-1]
    state[2,xpos:,ypos-n_ghost:ypos] = -state[2,xpos:,ypos+n_ghost-1:ypos-1:-1]

    return state

def set_boundary(state, coords, dim, n_ghost):
    if dim == 0:
        return _set_boundary_x(state, coords[0], coords[1], n_ghost)
    return _set_boundary_y(state, coords[0], coords[1], n_ghost)

jit_module(nopython=True, error_model="numpy")

def initial_conditions(coords):
    n_eq = 4

    U = np.zeros((n_eq, len(coords[0]), len(coords[1])))
    U[0,:,:] = 1.4
    U[1,:,:] = 1.4*3.0
    U[2,:,:] = 0.0
    U[3,:,:] = 0.5*1.4*9.0 + 1/(1.4-1)

    return U

from boxset.output.parallel import *
from boxset.coords import create_coordinates

def visualise(direc, save_index):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import configparser


    config = configparser.ConfigParser()
    config.read(direc + 'windtunnel.ini')

    n_ghost = np.int32(config['Grid']['n_ghost'])

    coords, pos, global_dims = create_coordinates(config)
    x = coords[0]
    y = coords[1]

    state = initial_conditions(coords)

    t, U = restore_from_dump(state, save_index, config['Output']['direc'], pos, global_dims, n_ghost)


    cf = plt.contourf(x[n_ghost:-n_ghost], y[n_ghost:-n_ghost], np.log(np.transpose(np.abs(U[0,n_ghost:-n_ghost,n_ghost:-n_ghost]))),
                     levels=100, cmap='terrain')
    # Create a Rectangle patch
    rect = patches.Rectangle((0.6, 0), 2.4, 0.2, linewidth=1, edgecolor='white', facecolor='white')
    plt.gca().add_patch(rect)

    plt.xlabel('x')
    plt.ylabel('y')
    plt.colorbar()

    plt.show()

from boxset.simulation import simulation

simulation("/Users/sjp/Desktop/boxset/windtunnel.ini", initial_conditions, set_boundary, restore_index=-1)

visualise('/Users/sjp/Desktop/boxset/', 4)
