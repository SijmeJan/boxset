rk_cfl = 1.0

def time_stepper(U, coords, dt, rhs, n_ghost, boundary_conditions):
    '''
    Euler time step
    '''
    return U + dt*rhs(U, coords, n_ghost, boundary_conditions)
