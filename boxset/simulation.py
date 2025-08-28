import configparser
import numpy as np
#import cProfile
from mpi4py import MPI

from .timeloop import timeloop
from .coords import create_coordinates
from .output.parallel import *
from .domain_decomposition import get_cpu_grid

def simulation(configuration_file, initial_conditions, boundary_conditions, restore_index=-1):
    '''Run simulation based on configuration file, and functions for setting initial and boundary conditions.

    Parameters:

    configuration_file: name of INI file with grid size etc.
    initial_conditions: function to set initial contitions. Must accept a singe argument, which is a list of coordinates. Should return a valid state.
    boundary conditions: function to set boundary conditions.
    restore_index: dump number to restore from. Defaults to -1, in which case a simulation will start from initial conditions.
    '''

    # Read from ini file. Rank 0 reads, then broadcasts to everyone.
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    config = []
    if rank == 0:
        config = configparser.ConfigParser()
        config.read(configuration_file)
    config = comm.bcast(config, root=0)

    n_ghost = np.int32(config['Grid']['n_ghost'])

    # Spatial coordinate list
    coords, pos, global_dims = create_coordinates(config)

    # CPU grid
    cpu_grid, my_pos = get_cpu_grid(global_dims)

    # Set initial conditions
    state = initial_conditions(coords)

    start_time = np.float64(config['Time']['start_time'])
    end_time = np.float64(config['Time']['end_time'])
    dump_dt = np.float64(config['Output']['dump_dt'])
    cfl = np.float64(config['Time']['courant_number'])

    t = start_time

    save_index = restore_index + 1

    # Restore from dump if necessary
    if restore_index > -1:
        t, state = restore_from_dump(state, restore_index, config['Output']['direc'], pos, global_dims, n_ghost)
    else:
        # First dump
        save_dump(t, state, save_index, config['Output']['direc'], pos, global_dims, n_ghost)
        save_index = save_index + 1

    while t < end_time:
        t_stop = t + dump_dt
        if t_stop > end_time:
            t_stop = end_time

        # Evolve until next dump
        state = timeloop(state, coords, t, t_stop, cfl, n_ghost, boundary_conditions, cpu_grid)
        t = t_stop

        # Data dump
        save_dump(t, state, save_index, config['Output']['direc'], pos, global_dims, n_ghost)
        save_index = save_index + 1

        if comm.Get_rank() == 0:
            print('Dum at t = ', t)
