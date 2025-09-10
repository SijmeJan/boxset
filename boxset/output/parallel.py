import numpy as np
from mpi4py import MPI


def save_dump(time, state, save_index, save_path,
              pos, global_dims, n_ghost):
    '''Save current time and state in npz file'''

    # Strip off ghost cells
    state_shape = np.shape(state)
    sel = (slice(0, state_shape[0]),)
    for dim in range(1, len(state_shape)):
        sel = sel + (slice(n_ghost, state_shape[dim] - n_ghost),)
    stripped_state = state[sel].copy()

    # Add state dimension, so that global_dims is the shape
    # that state would have on a single CPU
    global_state_dims = global_dims.copy()
    global_state_dims.insert(0, np.shape(state)[0])
    # Position of the subarray in the global array (all zeros for single CPU)
    pos_in_global_state = pos.copy()
    pos_in_global_state.insert(0, 0)

    # Create MPI datatype for our subaray
    subarr = MPI.DOUBLE.Create_subarray(global_state_dims,
                                        np.shape(stripped_state),
                                        pos_in_global_state)
    subarr.Commit()

    # Open file and write everything
    comm = MPI.COMM_WORLD
    amode = MPI.MODE_WRONLY | MPI.MODE_CREATE
    fh = MPI.File.Open(comm,
                       save_path + 'dump{}.dat'.format(save_index),
                       amode)
    if comm.Get_rank() == 0:
        fh.Write(np.asarray([time]))
    displacement = MPI.DOUBLE.Get_size()
    fh.Set_view(displacement, filetype=subarr)
    fh.Write_all(stripped_state)
    subarr.Free()
    fh.Close()

    return


def restore_from_dump(state, restore_index, restore_path,
                      pos, global_dims, n_ghost):
    '''Restore simulation time and state from npz file'''

    # Strip off ghost cells
    state_shape = np.shape(state)
    sel = (slice(0, state_shape[0]),)
    for dim in range(1, len(state_shape)):
        sel = sel + (slice(n_ghost, state_shape[dim] - n_ghost),)
    stripped_state = state[sel].copy()

    # Add state dimension, so that global_dims is the shape
    # that state would have on a single CPU
    global_state_dims = global_dims.copy()
    global_state_dims.insert(0, np.shape(state)[0])
    # Position of the subarray in the global array (all zeros for single CPU)
    pos_in_global_state = pos.copy()
    pos_in_global_state.insert(0, 0)

    # Create MPI datatype for our subaray
    subarr = MPI.DOUBLE.Create_subarray(global_state_dims,
                                        np.shape(stripped_state),
                                        pos_in_global_state)
    subarr.Commit()

    # Open file and read everything
    comm = MPI.COMM_WORLD
    amode = MPI.MODE_RDONLY
    fh = MPI.File.Open(comm,
                       restore_path + 'dump{}.dat'.format(restore_index),
                       amode)
    t = np.asarray([0.0])
    if comm.Get_rank() == 0:
        fh.Read(t)
    displacement = MPI.DOUBLE.Get_size()
    fh.Set_view(displacement, filetype=subarr)
    fh.Read_all(stripped_state)
    subarr.Free()
    fh.Close()

    state[sel] = stripped_state

    comm.Bcast(t, root=0)

    if comm.Get_rank() == 0:
        print('Restoring from ' + restore_path +
              'dump{}.dat'.format(restore_index))

    return t[0], state
