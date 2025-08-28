import numpy as np

from .domain_decomposition import *

def create_coordinates(config):
    pos, local_N = get_position_in_global_array(config)
    global_dims = get_grid_dimensions(config)

    n_ghost = np.int32(config['Grid']['n_ghost'])

    ret = []

    for dim in range(0, len(global_dims)):
        xmin = np.float64(config['Grid']['min{}'.format(dim)])
        xmax = np.float64(config['Grid']['max{}'.format(dim)])

        Nx = np.int32(config['Grid']['N{}'.format(dim)])
        dx = (xmax - xmin)/Nx

        xmin = xmin + pos[dim]*dx
        xmax = xmin + local_N[dim]*dx

        ret.append(np.linspace(xmin - (n_ghost-0.5)*dx, xmax + (n_ghost-0.5)*dx, local_N[dim] + 2*n_ghost))

    return ret, pos, global_dims
