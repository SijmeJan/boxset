import numpy as np
from numba import jit_module
from mpi4py import MPI

from boxset.conservation_laws.iso_sbox import shear_param
from boxset.output.parallel import *
from boxset.coords import create_coordinates
from boxset.simulation import simulation, read_config_file

from psi_stratified.equilibrium import Equilibrium
from psi_stratified.stokesdensity import StokesDensity

configuration_file = "/Users/sjp/Desktop/boxset/vsi.ini"
config = read_config_file(configuration_file)

# Goes wrong very fast
metallicity = 0.31
stokes = 0.001
alpha = 1.0e-4
eta = 1.0

n_coll = 2000

sigma = StokesDensity([stokes])   # Monodisperse

equilibrium = Equilibrium(metallicity, sigma, 1, viscous_alpha=alpha)
equilibrium.solve_horizontal_velocities(n_coll, neglect_gas_viscosity=True)

def gas_source(config):
    # Spatial coordinate list
    coords, pos, global_dims, periodic_flags = create_coordinates(config)
    z = coords[1]

    rhog, sigma, dg_ratio, dust_rho, gas_vx, gas_vy, gas_vz, dust_ux, dust_uy, dust_uz = equilibrium.get_state(z, eta=eta)
    gas_dvx, gas_dvy, gas_dvz, dust_dux, dust_duy, dust_duz = equilibrium.evaluate_velocities(z, k=1, eta=eta)

    Sx = -2*(gas_vy + eta)
    Sy = (2 - shear_param)*gas_vx
    Sz = 0.0*z

    return Sx, Sy, Sz

Sx, Sy, Sz = gas_source(config)

def user_source_func(U, coords, time):
    ret = np.zeros_like(U)
    z = coords[-1]

    for i in range(0, len(z)):
        ret[1, ..., i] = 2*U[0, ..., i]*eta + U[0, ..., i]*Sx[i]
        ret[2, ..., i] = U[0, ..., i]*Sy[i]
        ret[3, ..., i] = U[0, ..., i]*Sz[i]

    return ret

def _set_boundary_x(state, x, n_ghost):
    # Periodic boundary conditions
    state[:,0:n_ghost,:] = state[:,len(x)-2*n_ghost:len(x)-n_ghost,:]
    state[:,len(x)-n_ghost:len(x),:] = state[:,n_ghost:2*n_ghost,:]

    return state

def _set_boundary_z(state, x, n_ghost):
    # Reflecting boundary conditions
    #state[:,:,0:n_ghost] = state[:,:,n_ghost:2*n_ghost]
    #state[3,:,0:n_ghost] = -state[3,:,n_ghost:2*n_ghost]
    #state[:,:,len(x)-n_ghost:len(x)] = state[:,:,len(x)-2*n_ghost:len(x)-n_ghost]
    #state[3,:,len(x)-n_ghost:len(x)] = -state[3,:,len(x)-2*n_ghost:len(x)-n_ghost]

    # Nonreflecting
    state[0,:,0:n_ghost] = np.exp(-0.5*x[0:n_ghost]**2)
    state[1,:,0:n_ghost] = 0.0
    state[2,:,0:n_ghost] = -eta*state[0,:,0:n_ghost]
    state[3,:,0:n_ghost] = 0.0

    state[0,:,len(x)-n_ghost:len(x)] = np.exp(-0.5*x[len(x)-n_ghost:len(x)]**2)
    state[1,:,len(x)-n_ghost:len(x)] = 0.0
    state[2,:,len(x)-n_ghost:len(x)] = -eta*state[0,:,len(x)-n_ghost:len(x)]
    state[3,:,len(x)-n_ghost:len(x)] = 0.0

    return state

def set_boundary(state, coords, dim, n_ghost):
    if dim == 0:
        return _set_boundary_x(state, coords[0], n_ghost)
    return _set_boundary_z(state, coords[1], n_ghost)

jit_module(nopython=True, error_model="numpy")

def initial_conditions(coords, config):
    n_eq = 4
    #qz = 0.1
    noise_level = 0.01

    h = 0.5*noise_level
    l = -0.5*noise_level

    rng = np.random.default_rng(seed=1+MPI.COMM_WORLD.Get_rank())

    U = np.zeros((n_eq, len(coords[0]), len(coords[1])))
    z = coords[1]
    for i in range(0, len(z)):
        U[0,:,i] = np.exp(-0.5*z[i]**2)
        U[1,:,i] = U[0,:,i]*Sy[i]/(2-shear_param)
        U[2,:,i] = U[0,:,i]*(-eta - 0.5*Sx[i])

    U[1,...] = U[1,...] + U[0,...]*rng.uniform(low=l, high=h, size=np.shape(U[1,...]))
    U[2,...] = U[2,...] + U[0,...]*rng.uniform(low=l, high=h, size=np.shape(U[2,...]))
    U[3,...] = U[3,...] + U[0,...]*rng.uniform(low=l, high=h, size=np.shape(U[3,...]))

    return U

def visualise(ini_file, save_index):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import configparser


    config = configparser.ConfigParser()
    config.read(ini_file)

    n_ghost = np.int32(config['Grid']['n_ghost'])

    coords, pos, global_dims, periodic_flags = create_coordinates(config)
    x = coords[0]
    y = coords[1]

    state = initial_conditions(coords, config)

    t, U = restore_from_dump(state, save_index, config['Output']['direc'], pos, global_dims, n_ghost)

    #f = U[0,n_ghost:-n_ghost,n_ghost:-n_ghost]
    #cf = plt.contourf(x[n_ghost:-n_ghost], y[n_ghost:-n_ghost],
    #                  np.transpose(np.log10(f)),
    #                  levels=100, cmap='plasma')

    z = coords[1]
    for i in range(0, len(z)):
        #U[0,:,i] -= np.exp(-0.5*z[i]**2)
        U[1,:,i] -= U[0,:,i]*Sy[i]/(2-shear_param)
        U[2,:,i] += U[0,:,i]*(eta + 0.5*Sx[i])

    dens = U[0,n_ghost:-n_ghost,n_ghost:-n_ghost]
    vx = U[1,n_ghost:-n_ghost,n_ghost:-n_ghost]/dens
    vy = U[2,n_ghost:-n_ghost,n_ghost:-n_ghost]/dens
    vz = U[3,n_ghost:-n_ghost,n_ghost:-n_ghost]/dens


    minvel = np.min([vx, vy, vz])
    maxvel = np.max([vx, vy, vz])
    levels = np.linspace(minvel, maxvel, 100)

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(10,9))

    cf1 = ax1.contourf(x[n_ghost:-n_ghost], y[n_ghost:-n_ghost],
                      np.transpose(np.log10(dens)),
                      levels=100, cmap='plasma')

    cf2 = ax2.contourf(x[n_ghost:-n_ghost], y[n_ghost:-n_ghost],
                   np.transpose(vx),
                   levels=levels, cmap='coolwarm')
    cf3 = ax3.contourf(x[n_ghost:-n_ghost], y[n_ghost:-n_ghost],
                   np.transpose(vy),
                   levels=levels, cmap='seismic')
    cf4 = ax4.contourf(x[n_ghost:-n_ghost], y[n_ghost:-n_ghost],
                   np.transpose(vz),
                   levels=levels, cmap='bwr')

    #plt.plot(y, np.mean(U[3,:,:]/U[0,:,:], axis=0))

    #plt.yscale('log')
    #plt.xlabel('x')
    #plt.ylabel('z')
    #fig.colorbar(cf4, ax=ax4)
    #fig.colorbar(cf1, ax=[ax1])#, orientation='horizontal', fraction=.1)
    fig.colorbar(cf4, ax=[ax1, ax2,ax3,ax4], orientation='horizontal', fraction=.1)
    plt.show()




simulation(configuration_file, initial_conditions, set_boundary, user_source_func, restore_index=271)
#simulation("/home/sijmejanpaarde/streaming_instability.ini", initial_conditions, set_boundary, restore_index=-1)

visualise("/Users/sjp/Desktop/boxset/vsi.ini", 200)
