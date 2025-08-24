import configparser
import numpy as np
#import cProfile

from .timeloop import timeloop
from .coords import create_coordinates
from .output.basic import *

def simulation(configuration_file, initial_conditions, boundary_conditions, restore_index=-1):
    # Read from ini file
    config = configparser.ConfigParser()
    config.read(configuration_file)

    n_ghost = np.int32(config['Grid']['n_ghost'])

    # MPI domain composition


    # Spatial coordinate list
    coords = create_coordinates(config)

    # Set initial conditions
    state = initial_conditions(coords)

    start_time = np.float64(config['Time']['start_time'])
    end_time = np.float64(config['Time']['end_time'])
    dump_dt = np.float64(config['Output']['dump_dt'])

    t = start_time

    save_index = restore_index + 1

    # Restore from dump if necessary
    if restore_index > -1:
        t, state = restore_from_dump(restore_index, config['Output']['direc'])
    else:
        # First dump
        save_dump(t, state, save_index, config['Output']['direc'])
        save_index = save_index + 1

    while t < end_time:
        t_stop = t + dump_dt
        if t_stop > end_time:
            t_stop = end_time

        # Evolve until next dump
        state = timeloop(state, coords, t, t_stop, 0.8, n_ghost, boundary_conditions)
        t = t_stop

        # Data dump
        save_dump(t, state, save_index, config['Output']['direc'])
        save_index = save_index + 1

        print(t)

        # Monitoring functions
