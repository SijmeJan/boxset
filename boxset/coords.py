import numpy as np

from .domain_decomposition import get_pos_in_global_array, get_grid_dimensions


def create_coordinates(config):
    '''Create list of coordinate arrays based on configuration file.

    Parameters:

    config: configparser dictionary object.

    Returns:

    list of coordinate arrays, position of current patch within global array,
    global dimensions of grid.'''

    # Get position of this patch in global array
    pos, local_N = get_pos_in_global_array(config)
    global_dims = get_grid_dimensions(config)

    n_ghost = np.int32(config['Grid']['n_ghost'])

    # List of coordinate arrays
    ret = []
    periodic_flags = []

    # Loop through all spatial dimensions
    for dim in range(0, len(global_dims)):
        xmin = np.float64(config['Grid']['min{}'.format(dim)])
        xmax = np.float64(config['Grid']['max{}'.format(dim)])

        Nx = np.int32(config['Grid']['N{}'.format(dim)])
        dx = (xmax - xmin)/Nx

        xmin = xmin + pos[dim]*dx
        xmax = xmin + local_N[dim]*dx

        ret.append(np.linspace(xmin - (n_ghost-0.5)*dx,
                               xmax + (n_ghost-0.5)*dx,
                               local_N[dim] + 2*n_ghost))

        periodic_flags.append(eval(config['Grid']['Periodic{}'.format(dim)]))

    return ret, pos, global_dims, periodic_flags
