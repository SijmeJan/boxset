import numpy as np
from mpi4py import MPI

def get_grid_dimensions(config):
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
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    # 'balanced' distribution of cpus over dimensions, sorted from large to small
    proc_dims_temp = MPI.Compute_dims(size, len(dims))

    # Give smallest grid dimension smallest numer of CPUs
    idx = np.argsort(dims)
    proc_dims = np.zeros_like(idx)
    for i in range(0, len(dims)):
        proc_dims[i] = proc_dims_temp[idx[-i-1]]

    # Grid of cpus and position of this rank in array
    cpu_grid = np.asarray(range(size)).reshape(proc_dims)
    my_position_in_grid = np.asarray(cpu_grid==rank).nonzero()

    return cpu_grid, my_position_in_grid

def get_position_in_global_array(config):
    dims = get_grid_dimensions(config)

    cpu_grid, my_position_in_grid = get_cpu_grid(dims)

    ret_pos = []
    ret_size = []

    for dim in range(0, len(dims)):
        # Size of global grid in this direction
        Nx = np.int32(config['Grid']['N{}'.format(dim)])

        # Local grid sizes in this direction
        Nx_local = int(Nx/cpu_grid.shape[dim])*np.ones(cpu_grid.shape[dim], dtype=int)
        Nx_local[-1] = Nx - np.sum(Nx_local[0:-1])

        my_position_dim = my_position_in_grid[dim][0]

        ret_pos.append(np.sum(Nx_local[0:my_position_dim]))
        ret_size.append(Nx_local[my_position_dim])

    return ret_pos, ret_size

def send_boundaries(state, cpu_grid, dim, n_ghost):
    '''Send ghost cell variables to neighbouring MPI processes'''
    periodic_flag = True

    comm = MPI.COMM_WORLD
    my_position_in_grid = np.asarray(cpu_grid==comm.Get_rank()).nonzero()
    #if comm.Get_rank() == 0:
    #    print('---------------------------------------------')
    #comm.Barrier()

    # Shape of subarray of ghost cells: i.e. (n_eq, n_ghost, ny, nz)
    sub_size = (np.shape(state)[0],)
    for i in range(1, len(np.shape(state))):
        if i == dim + 1:
            sub_size = sub_size + (n_ghost,)
        else:
            sub_size = sub_size + (np.shape(state)[i],)

    # Position of subarray in state array, default to zeros
    pos_in_global_state = np.zeros(len(np.shape(state)), dtype=int)

    for direction in [1, -1]:
        # direction == 1: send to right, receive from left
        # direction == -1: send to left, receive from right

        # Create MPI datatypes for our subarrays
        pos_in_global_state[dim+1] = (direction == -1)*(np.shape(state)[dim+1] - n_ghost)
        subarr_recv = MPI.FLOAT64_T.Create_subarray(np.shape(state), sub_size, pos_in_global_state)
        subarr_recv.Commit()

        pos_in_global_state[dim+1] = (direction == -1)*n_ghost + \
            (direction == 1)*(np.shape(state)[dim+1] - 2*n_ghost)
        subarr_send = MPI.FLOAT64_T.Create_subarray(np.shape(state), sub_size, pos_in_global_state)
        subarr_send.Commit()

        dest_pos = list(my_position_in_grid)
        dest = -1
        target_idx = my_position_in_grid[dim][0] + direction
        if periodic_flag is True:
            target_idx = (target_idx + np.shape(cpu_grid)[dim]) % np.shape(cpu_grid)[dim]

        if (target_idx < np.shape(cpu_grid)[dim] and target_idx >= 0):
            dest_pos[dim] = np.asarray([target_idx])     # direc
            dest = cpu_grid[tuple(dest_pos)][0]

        #print(comm.Get_rank(), ' Dest: ', dest)

        src_pos = list(my_position_in_grid)
        src = -1
        target_idx = my_position_in_grid[dim][0] - direction
        if periodic_flag is True:
            target_idx = (target_idx + np.shape(cpu_grid)[dim]) % np.shape(cpu_grid)[dim]

        if (target_idx < np.shape(cpu_grid)[dim] and target_idx >= 0):
            src_pos[dim] = np.asarray([target_idx])
            src = cpu_grid[tuple(src_pos)][0]
        #print(comm.Get_rank(), ' Src: ', src)

        #if src != dest:
        #    comm.Sendrecv([state, subarr_send], dest, sendtag=0,
        #                  recvbuf=[state, subarr_recv], source=src, recvtag=0, status=None)

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

    #comm.Barrier()
    #if comm.Get_rank() == 0:
    #    print('---------------------------------------------')

    return state
