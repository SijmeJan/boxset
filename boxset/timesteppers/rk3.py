rk_cfl = 1.0

def time_stepper(U, coords, dt, rhs, n_ghost, boundary_conditions):
    '''
    Third order SSP RK3 integrator
    '''
    U1 = U + dt*rhs(U, coords, n_ghost, boundary_conditions)
    U2 = 0.75*U + 0.25*U1 + 0.25*dt*rhs(U1, coords, n_ghost, boundary_conditions)

    return U/3 + 2*U2/3 + 2*dt*rhs(U2, coords, n_ghost, boundary_conditions)/3
