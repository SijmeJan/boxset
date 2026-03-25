import numpy as np
from mpi4py import MPI
import numba_mpi
import numba
from numba import jit
from numba.np.unsafe.ndarray import to_fixed_tuple, tuple_setitem

from .utils.utils import my_swapaxes

def get_grid_dimensions(config):
    '''Get global grid dimensions from config file.

    Parameters:

    config: configparser dictionary

    Returns:

    list of grid dimensions'''

    ret = []

    finished = False
    dim = 0
    while finished is False:
        if 'min{}'.format(dim) in config['Grid']:
            ret.append(np.int32(config['Grid']['N{}'.format(dim)]))
        else:
            finished = True
        dim = dim + 1

    return ret


def get_cpu_grid(dims):
    '''Create cpu grid for parallel computation.

    Parameters:

    dims: list of global grid dimensions

    Returns:

    grid of cpus, position of current cpu in grid'''

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    # 'balanced' dist of cpus over dimensions, sorted from large to small
    proc_dims_temp = MPI.Compute_dims(size, len(dims))

    # Give smallest grid dimension smallest numer of CPUs
    idx = np.argsort(dims)
    proc_dims = np.zeros_like(idx)
    for i in range(0, len(dims)):
        proc_dims[i] = proc_dims_temp[idx[-i-1]]

    # Grid of cpus and position of this rank in array
    cpu_grid = np.asarray(range(size)).reshape(proc_dims)
    my_position_in_grid = np.asarray(cpu_grid == rank).nonzero()

    return cpu_grid, my_position_in_grid


def get_pos_in_global_array(config):
    '''Get position of current patch in global array.

    Parameters:

    config: configparser dictionary

    Returns:

    position in global array, size of local patch'''

    dims = get_grid_dimensions(config)

    cpu_grid, my_position_in_grid = get_cpu_grid(dims)

    ret_pos = []
    ret_size = []

    for dim in range(0, len(dims)):
        # Size of global grid in this direction
        Nx = np.int32(config['Grid']['N{}'.format(dim)])

        # Local grid sizes in this direction
        Nx_local = int(Nx/cpu_grid.shape[dim])*np.ones(cpu_grid.shape[dim],
                                                       dtype=int)
        Nx_local[-1] = Nx - np.sum(Nx_local[0:-1])

        my_position_dim = my_position_in_grid[dim][0]

        ret_pos.append(np.sum(Nx_local[0:my_position_dim]))
        ret_size.append(Nx_local[my_position_dim])

    return ret_pos, ret_size

@jit
def sub_array_size(dim, state, n_ghost):
    # Shape of subarray of ghost cells: i.e. (n_eq, n_ghost, ny, nz)
    #sub_size = (np.shape(state)[0],)
    sub_size = (np.shape(state)[0],0,0)

    #pos = to_fixed_tuple(np.arange(a.ndim), a.ndim)
    #tmp = pos[axis1]
    #pos = tuple_setitem(pos, axis1, pos[axis2])

    for i in range(1, len(np.shape(state))):
        if i == dim + 1:
            #print('HALLO ', i, sub_size, sub_size + (int(n_ghost),))
            #sub_size = sub_size + (int(n_ghost),)
            sub_size = tuple_setitem(sub_size, i, int(n_ghost))
        else:
            #sub_size = sub_size + (np.shape(state)[i],)
            sub_size = tuple_setitem(sub_size, i, np.shape(state)[i])

    return sub_size

def send_boundaries_mpi4py(state, cpu_grid, dim, n_ghost, periodic_flag):
    '''Send ghost cell variables to neighbouring MPI processes.

    Parameters:

    state: state vector
    cpu_grid: grid of cpus (output of get_cpu_grid)
    dim: current dimension
    n_ghost: number of ghost cells
    periodic_flag: bool whether this dimension is periodic
    Returns:

    updated state'''

    comm = MPI.COMM_WORLD
    my_position_in_grid = np.asarray(cpu_grid == comm.Get_rank()).nonzero()

    # Shape of subarray of ghost cells: i.e. (n_eq, n_ghost, ny, nz)
    sub_size = sub_array_size(dim, state, n_ghost)

    # Position of subarray in state array, default to zeros
    pos_in_global_state = np.zeros(len(np.shape(state)), dtype=int)

    for direction in [1, -1]:
        # direction == 1: send to right, receive from left
        # direction == -1: send to left, receive from right

        # Create MPI datatypes for our subarrays
        pos_in_global_state[dim+1] = \
            (direction == -1)*(np.shape(state)[dim+1] - n_ghost)
        subarr_recv = MPI.DOUBLE.Create_subarray(np.shape(state), sub_size,
                                                 pos_in_global_state)
        subarr_recv.Commit()

        pos_in_global_state[dim+1] = (direction == -1)*n_ghost + \
            (direction == 1)*(np.shape(state)[dim+1] - 2*n_ghost)
        subarr_send = MPI.DOUBLE.Create_subarray(np.shape(state), sub_size,
                                                 pos_in_global_state)
        subarr_send.Commit()

        dest_pos = list(my_position_in_grid)
        dest = -1
        target_idx = my_position_in_grid[dim][0] + direction
        if periodic_flag is True:
            target_idx = (target_idx + np.shape(cpu_grid)[dim]) \
                % np.shape(cpu_grid)[dim]

        if (target_idx < np.shape(cpu_grid)[dim] and target_idx >= 0):
            dest_pos[dim] = np.asarray([target_idx])     # direc
            dest = cpu_grid[tuple(dest_pos)][0]

        src_pos = list(my_position_in_grid)
        src = -1
        target_idx = my_position_in_grid[dim][0] - direction
        if periodic_flag is True:
            target_idx = (target_idx + np.shape(cpu_grid)[dim]) \
                % np.shape(cpu_grid)[dim]

        if (target_idx < np.shape(cpu_grid)[dim] and target_idx >= 0):
            src_pos[dim] = np.asarray([target_idx])
            src = cpu_grid[tuple(src_pos)][0]

        if my_position_in_grid[dim][0] % 2 == 0:
            if src != -1 and src != comm.Get_rank():
                comm.Recv([state, subarr_recv], source=src)
            if dest != -1 and dest != comm.Get_rank():
                comm.Send([state, subarr_send], dest=dest)
        else:
            if dest != -1 and dest != comm.Get_rank():
                comm.Send([state, subarr_send], dest=dest)
            if src != -1 and src != comm.Get_rank():
                comm.Recv([state, subarr_recv], source=src)
        subarr_send.Free()
        subarr_recv.Free()

    return state

@jit
def temp_send(state, dim, my_position_in_grid, src, dest, direction, n_ghost, state_send, state_recv):
    #if my_position_in_grid[dim][0] % 2 == 0:
    if my_position_in_grid[dim] % 2 == 0:
        if src != -1 and src != numba_mpi.rank():
            numba_mpi.recv(state_recv, source=src)
            if direction == -1:
                state[...,np.shape(state)[-1]-n_ghost:np.shape(state)[-1]] = state_recv
            else:
                state[...,0:n_ghost] = state_recv

        if dest != -1 and dest != numba_mpi.rank():
            numba_mpi.send(state_send, dest=dest)
    else:
        if dest != -1 and dest != numba_mpi.rank():
            numba_mpi.send(state_send, dest=dest)
        if src != -1 and src != numba_mpi.rank():
            numba_mpi.recv(state_recv, source=src)
            if direction == -1:
                state[...,np.shape(state)[-1]-n_ghost:np.shape(state)[-1]] = state_recv
            else:
                state[...,0:n_ghost] = state_recv

    return state

@jit
def find_destination(my_position_in_grid, cpu_grid, dim, direction, periodic_flag):
    dest = -1
    target_idx = my_position_in_grid[dim] + direction
    if periodic_flag is True:
        target_idx = (target_idx + np.shape(cpu_grid)[dim]) \
            % np.shape(cpu_grid)[dim]

    dest_pos = np.copy(my_position_in_grid)

    if (target_idx < np.shape(cpu_grid)[dim] and target_idx >= 0):
        dest_pos[dim] = target_idx     # direc
        #dest = cpu_grid[tuple(dest_pos)]
        dest = cpu_grid[to_fixed_tuple(dest_pos, cpu_grid.ndim)]

    return dest

@jit
def find_source(my_position_in_grid, cpu_grid, dim, direction, periodic_flag):
    src = -1
    target_idx = my_position_in_grid[dim] - direction
    if periodic_flag is True:
        target_idx = (target_idx + np.shape(cpu_grid)[dim]) \
            % np.shape(cpu_grid)[dim]

    src_pos = np.copy(my_position_in_grid)

    if (target_idx < np.shape(cpu_grid)[dim] and target_idx >= 0):
        src_pos[dim] = target_idx
        #src = cpu_grid[tuple(src_pos)]
        src = cpu_grid[to_fixed_tuple(src_pos, cpu_grid.ndim)]

    return src

@jit
def send_boundaries(state, cpu_grid, dim, n_ghost, periodic_flag):
    '''Send ghost cell variables to neighbouring MPI processes.

    Parameters:

    state: state vector
    cpu_grid: grid of cpus (output of get_cpu_grid)
    dim: current dimension
    n_ghost: number of ghost cells
    periodic_flag: bool whether this dimension is periodic
    Returns:

    updated state'''

    # Last dimension is the one to set boundaries for
    state = my_swapaxes(state, dim+1, len(np.shape(state))-1)

    # Some voodoo needed here: my_pos is a list of arrays, each with a single entry.
    # We want an ndarray, same shape as list, but with single entries.
    my_pos = list(np.asarray(cpu_grid == numba_mpi.rank()).nonzero())
    my_position_in_grid = np.zeros(len(my_pos), dtype=numba.int64)
    for i in range(0, len(my_position_in_grid)):
        my_position_in_grid[i] = my_pos[i][0]

    for direction in [1, -1]:
        # direction == 1: send to right, receive from left
        # direction == -1: send to left, receive from right

        if direction == -1:
            state_send = state[...,n_ghost:2*n_ghost]
        else:
            state_send = state[...,-2*n_ghost:-n_ghost]
        state_recv = 0*state_send

        dest = find_destination(my_position_in_grid, cpu_grid, dim, direction, periodic_flag)
        src = find_source(my_position_in_grid, cpu_grid, dim, direction, periodic_flag)

        state = temp_send(state, dim, my_position_in_grid, src, dest, direction, n_ghost, state_send, state_recv)

    state = my_swapaxes(state, dim+1, len(np.shape(state))-1)

    return state
