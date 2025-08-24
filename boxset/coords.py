import numpy as np

def create_coordinates(config):
    n_ghost = np.int32(config['Grid']['n_ghost'])

    ret = []

    finished = False
    dim = 0
    while finished is False:
        if 'min{}'.format(dim) in config['Grid']:
            xmin = np.float64(config['Grid']['min{}'.format(dim)])
            xmax = np.float64(config['Grid']['max{}'.format(dim)])

            Nx = np.int32(config['Grid']['N{}'.format(dim)])
            dx = (xmax - xmin)/Nx

            ret.append(np.linspace(xmin - (n_ghost-0.5)*dx, xmax + (n_ghost-0.5)*dx, Nx + 2*n_ghost))
        else:
            finished = True
        dim = dim + 1

    return ret
