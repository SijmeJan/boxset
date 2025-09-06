import numpy as np
from mpi4py import MPI

from .timesteppers.rk3 import time_stepper, rk_cfl
from .rhs import calculate_rhs, calc_time_step

def timeloop(state, coords, start_time, end_time, courant_number, n_ghost, boundary_conditions, cpu_grid, safety_factor):
    '''
    Evolve the state from start_time to end_time. Returns final state.
    '''

    # Loop until end_time reached
    t = start_time
    while t < end_time:
        # Calculate maximum allowed time step
        dt_local = courant_number*rk_cfl*calc_time_step(state, coords, n_ghost, boundary_conditions)/len(coords)

        # MPI: calculate minimum time step
        dt_local = np.asarray([dt_local])
        dt = dt_local.copy()
        MPI.COMM_WORLD.Allreduce([dt_local, MPI.DOUBLE], [dt, MPI.DOUBLE], op=MPI.MIN)
        dt = dt[0]

        # End exactly on end_time
        if t + dt > end_time:
            dt = end_time - t

        # Do one time step
        state = time_stepper(state, coords, dt, calculate_rhs, n_ghost, boundary_conditions, cpu_grid, safety_factor)

        if MPI.COMM_WORLD.Get_rank() == 0:
            print('t = ', t, 'dt = ', dt)
        t = t + dt

    return state
