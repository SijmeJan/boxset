rk_cfl = 1.0


def time_stepper(U, coords, time, dt, rhs, n_ghost, boundary_conditions,
                 cpu_grid, safety_factor, periodic_flags):
    '''
    Third order SSP RK3 integrator
    '''

    U1 = U + dt*rhs(U, coords, time, n_ghost, boundary_conditions,
                    cpu_grid, safety_factor*dt, periodic_flags)
    U2 = 0.75*U + 0.25*U1 + \
        0.25*dt*rhs(U1, coords, time + dt, n_ghost, boundary_conditions,
                    cpu_grid, safety_factor*dt, periodic_flags)

    return U/3 + 2*U2/3 + \
        2*dt*rhs(U2, coords, time + 0.5*dt, n_ghost, boundary_conditions,
                 cpu_grid, safety_factor*dt, periodic_flags)/3

