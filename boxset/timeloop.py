from .timesteppers.rk3 import time_stepper, rk_cfl
from .rhs import calculate_rhs, calc_time_step

def timeloop(state, coords, start_time, end_time, courant_number, n_ghost, boundary_conditions):
    '''
    Evolve the state from start_time to end_time. Returns final state.
    '''

    # Loop until end_time reached
    t = start_time
    while t < end_time:
        # Calculate maximum allowed time step
        dt = courant_number*rk_cfl*calc_time_step(state, coords, n_ghost, boundary_conditions)/len(coords)
        # End exactly on end_time
        if t + dt > end_time:
            dt = end_time - t

        # Do one time step
        state = time_stepper(state, coords, dt, calculate_rhs, n_ghost, boundary_conditions)

        print(t, dt)
        t = t + dt

    return state
