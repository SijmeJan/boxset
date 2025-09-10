import numpy as np


def save_dump(time, state, save_index, save_path,
              pos, global_dims, n_ghost):
    '''Save current time and state in npz file'''
    t = np.asarray([time])

    filename = save_path + 'dump{}.npz'.format(save_index)
    np.savez(filename, time=t, state=state)

    return


def restore_from_dump(state, restore_index, restore_path,
                      pos, global_dims, n_ghost):
    '''Restore simulation time and state from npz file'''

    filename = restore_path + 'dump{}.npz'.format(restore_index)
    dump = np.load(filename)
    print('Restoring from ' + filename)

    t = dump['time'][0]
    state = dump['state']

    return t, state
