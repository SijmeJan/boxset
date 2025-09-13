import numpy as np
from mpi4py import MPI

from ..domain_decomposition import send_boundaries

def remap(state, coords, time_shift, cpu_grid, n_ghost):
    x = coords[0]
    y = coords[1]
    velocity = -1.5*x


    # Number of cells to shift in y direction
    Nshift = np.int32(time_shift*velocity/(x[1] - x[0]))
    #Nshift = 4*np.ones_like(x, dtype=int)

    #print('Nshift: ', time_shift*velocity/(x[1] - x[0]))
    #print(Nshift)

    # Maximum number of grid cells to shift
    Nmax_local = np.asarray([np.max(Nshift)])
    Nmax = Nmax_local.copy()
    MPI.COMM_WORLD.Allreduce([Nmax_local, MPI.INT],
                             [Nmax, MPI.INT], op=MPI.MAX)
    Nmax = Nmax[0]
    #print('Nmax: ', Nmax)

    finished = False
    while finished is False:
        finished = True

        # Boundaries have to be periodic!
        state[:,:,0:n_ghost,...] = state[:,:,len(y)-2*n_ghost:len(y)-n_ghost,...]
        state[:,:,len(y)-n_ghost:len(y),...] = state[:,:,n_ghost:2*n_ghost,...]

        send_boundaries(state, cpu_grid, 1, n_ghost)

        for i in range(0, len(x)):
            current_shift = np.sign(Nshift[i])*np.min([n_ghost, np.abs(Nshift[i])])
            if current_shift != 0:
                finished = False
                state[:,i,...] = np.roll(state[:,i,...], current_shift, axis=1)
                #print('Shifting i = {} by {} cells'.format(i, current_shift))
            Nshift[i] = Nshift[i] - current_shift

    return state
