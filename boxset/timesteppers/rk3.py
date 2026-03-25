from numba import jit

rk_cfl = 1.0

@jit
def time_stepper(U, coords, time, dt, rhs, n_ghost, boundary_conditions,
                 cpu_grid, safety_factor, periodic_flags, user_source_func):
    '''
    Third order SSP RK3 integrator
    '''

    U1 = U + dt*rhs(U, coords, time, n_ghost, boundary_conditions,
                    cpu_grid, safety_factor*dt, periodic_flags, user_source_func)
    U2 = 0.75*U + 0.25*U1 + \
        0.25*dt*rhs(U1, coords, time + dt, n_ghost, boundary_conditions,
                    cpu_grid, safety_factor*dt, periodic_flags, user_source_func)

    return U/3 + 2*U2/3 + \
        2*dt*rhs(U2, coords, time + 0.5*dt, n_ghost, boundary_conditions,
                 cpu_grid, safety_factor*dt, periodic_flags, user_source_func)/3

