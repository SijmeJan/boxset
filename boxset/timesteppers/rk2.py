rk_cfl = 1.0

def time_stepper(U, coords, dt, rhs, n_ghost, boundary_conditions, cpu_grid):
    '''
    SSP RK2 integrator
    '''
    U1 = U + dt*rhs(U, coords, n_ghost, boundary_conditions, cpu_grid)
    return 0.5*U + 0.5*U1 + 0.5*dt*rhs(U1, coords, n_ghost, boundary_conditions, cpu_grid)
