rk_cfl = 1.0


def time_stepper(U, coords, time, dt, rhs, n_ghost, boundary_conditions,
                 cpu_grid, safety_factor):
    '''
    Euler time step
    '''
    return U + dt*rhs(U, coords, time, n_ghost, boundary_conditions,
                      cpu_grid, safety_factor*dt)
