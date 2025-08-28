# Theoretically, this could be 6
rk_cfl = 3.0

def time_stepper(U, coords, dt, rhs, n_ghost, boundary_conditions, cpu_grid):
    '''
    4th order, 10 stage SSP RK3 integrator
    '''
    U1 = U + dt*rhs(U, coords, n_ghost, boundary_conditions, cpu_grid)/6
    U2 = U1 + dt*rhs(U1, coords, n_ghost, boundary_conditions, cpu_grid)/6
    U3 = U2 + dt*rhs(U2, coords, n_ghost, boundary_conditions, cpu_grid)/6
    U4 = U3 + dt*rhs(U3, coords, n_ghost, boundary_conditions, cpu_grid)/6
    U5 = 3*U/5 + 2*(U4 + dt*rhs(U4, coords, n_ghost, boundary_conditions, cpu_grid)/6)/5
    U6 = U5 + dt*rhs(U5, coords, n_ghost, boundary_conditions, cpu_grid)/6
    U7 = U6 + dt*rhs(U6, coords, n_ghost, boundary_conditions, cpu_grid)/6
    U8 = U7 + dt*rhs(U7, coords, n_ghost, boundary_conditions, cpu_grid)/6
    U9 = U8 + dt*rhs(U8, coords, n_ghost, boundary_conditions, cpu_grid)/6

    return U/25 + 9*(U4 + dt*rhs(U4, coords, n_ghost, boundary_conditions, cpu_grid)/6)/25 + \
        3*(U9 + dt*rhs(U9, coords, n_ghost, boundary_conditions, cpu_grid)/6)/5