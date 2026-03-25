import numpy as np
from numba import jit_module
from mpi4py import MPI

from boxset.conservation_laws.tva_sbox import shear_param, eta, stokes, dust_diff
from boxset.output.parallel import *
from boxset.coords import create_coordinates
from boxset.simulation import simulation, read_config_file

configuration_file = "/Users/sjp/Desktop/boxset/tva_vsi.ini"
config = read_config_file(configuration_file)

eps0 = 3.0

def user_source_func(U, coords, time):
    return np.zeros_like(U)

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
    #state[0,:,0:n_ghost] = np.exp(-0.5*x[0:n_ghost]**2)
    #state[1,:,0:n_ghost] = 0.0
    #state[2,:,0:n_ghost] = -eta*state[0,:,0:n_ghost]
    #state[3,:,0:n_ghost] = 0.0
    #state[4,:,0:n_ghost] = np.exp(-0.5*x[0:n_ghost]**2)

    #state[0,:,len(x)-n_ghost:len(x)] = np.exp(-0.5*x[len(x)-n_ghost:len(x)]**2)
    #state[1,:,len(x)-n_ghost:len(x)] = 0.0
    #state[2,:,len(x)-n_ghost:len(x)] = -eta*state[0,:,len(x)-n_ghost:len(x)]
    #state[3,:,len(x)-n_ghost:len(x)] = 0.0
    #state[4,:,len(x)-n_ghost:len(x)] = np.exp(-0.5*x[len(x)-n_ghost:len(x)]**2)

    eps = eps0*np.exp(-0.5*stokes*x*x/dust_diff)
    deps = -x*stokes*eps/dust_diff
    p = np.exp(dust_diff*(eps-eps0)/stokes - 0.5*x*x)
    rho = p*(1 + eps)
    vx = 2*eta*eps*deps*stokes*x/(1 + eps)**3
    vy = -eta/(1 + eps)
    vz = -eps*stokes*x/(1 + eps)

    state[0,:,0:n_ghost] = rho[0:n_ghost]
    state[1,:,0:n_ghost] = rho[0:n_ghost]*vx[0:n_ghost]
    state[2,:,0:n_ghost] = rho[0:n_ghost]*vy[0:n_ghost]
    state[3,:,0:n_ghost] = rho[0:n_ghost]*vz[0:n_ghost]
    state[4,:,0:n_ghost] = p[0:n_ghost]

    state[0,:,len(x)-n_ghost:len(x)] = rho[len(x)-n_ghost:len(x)]
    state[1,:,len(x)-n_ghost:len(x)] = \
      rho[len(x)-n_ghost:len(x)]*vx[len(x)-n_ghost:len(x)]
    state[2,:,len(x)-n_ghost:len(x)] = \
      rho[len(x)-n_ghost:len(x)]*vy[len(x)-n_ghost:len(x)]
    state[3,:,len(x)-n_ghost:len(x)] = \
      rho[len(x)-n_ghost:len(x)]*vz[len(x)-n_ghost:len(x)]
    state[4,:,len(x)-n_ghost:len(x)] = p[len(x)-n_ghost:len(x)]

    return state

def set_boundary(state, coords, dim, n_ghost):
    if dim == 0:
        return _set_boundary_x(state, coords[0], n_ghost)
    return _set_boundary_z(state, coords[1], n_ghost)

jit_module(nopython=True, error_model="numpy")

def initial_conditions(coords, config):
    n_eq = 5
    noise_level = 1.0e-6

    h = 0.5*noise_level
    l = -0.5*noise_level

    rng = np.random.default_rng(seed=1+MPI.COMM_WORLD.Get_rank())

    U = np.zeros((n_eq, len(coords[0]), len(coords[1])))
    z = coords[1]
    eps = eps0*np.exp(-0.5*stokes*z*z/dust_diff)
    deps = -z*stokes*eps/dust_diff
    p = np.exp(dust_diff*(eps-eps0)/stokes - 0.5*z*z)

    dust_gas_ratio = lambda x: eps0*np.exp(-0.5*stokes*x*x/dust_diff)
    rho_g = \
      lambda x: np.exp(dust_diff*(dust_gas_ratio(x)-eps0)/stokes - 0.5*x*x)
    rho_d = lambda x: dust_gas_ratio(x)*rho_g(x)

    import scipy as sp
    dust_surface_density = sp.integrate.quad(rho_d, -4, 4)[0]
    gas_surface_density = sp.integrate.quad(rho_g, -4, 4)[0]
    print('Metallicity: ', dust_surface_density/gas_surface_density)

    for i in range(0, len(z)):
        #U[0,:,i] = np.exp(-0.5*z[i]**2)
        #U[4,:,i] = U[0,:,i]/1.1
        U[0,:,i] = p[i]*(1.0 + eps[i])
        U[1,:,i] = U[0,:,i]*2*eta*eps[i]*deps[i]*stokes*z[i]/(1 + eps[i])**3
        U[2,:,i] = -U[0,:,i]*eta/(1 + eps[i])
        U[3,:,i] = -U[0,:,i]*eps[i]*stokes*z[i]/(1 + eps[i])
        U[4,:,i] = p[i]

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

    for n in save_index:
        t, U = restore_from_dump(state, n, config['Output']['direc'], pos, global_dims, n_ghost)

        z = coords[1][n_ghost:-n_ghost]
        dens = U[0,n_ghost:-n_ghost,n_ghost:-n_ghost]
        vx = U[1,n_ghost:-n_ghost,n_ghost:-n_ghost]/dens
        vy = U[2,n_ghost:-n_ghost,n_ghost:-n_ghost]/dens
        vz = U[3,n_ghost:-n_ghost,n_ghost:-n_ghost]/dens
        rhog = U[4,n_ghost:-n_ghost,n_ghost:-n_ghost]
        dens = dens - rhog

        contour_plot = True
        if contour_plot is True:
            #dens = dens - np.mean(dens, axis=0)
            vx = vx - np.mean(vx, axis=0)
            vy = vy - np.mean(vy, axis=0)
            vz = vz - np.mean(vz, axis=0)

            print('Total kin: ', np.sum(0.5*(vx*vx + vy*vy + vz*vz)))

            minvel = np.min([vx, vy, vz])-1.0e-14
            maxvel = np.max([vx, vy, vz])+1.0e-14
            levels = np.linspace(minvel, maxvel, 100)


            fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(10,9))

            cf1 = ax1.contourf(x[n_ghost:-n_ghost], y[n_ghost:-n_ghost],
                        np.transpose((dens)),
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
            fig.colorbar(cf4, ax=[ax1,ax2,ax3,ax4], orientation='horizontal', fraction=.1)

        else:
            #plt.plot(z, np.mean(dens/rhog, axis=0))
            plt.plot(z, np.mean(vz, axis=0))
            #eps = 0.1202*np.exp(-0.5*z*z)
            #plt.plot(z, -eps*stokes*z/(1 + eps))
            #plt.plot(z, np.mean(rhog, axis=0))

    #plt.yscale('log')
    #plt.xlabel('x')
    #plt.ylabel('z')
    #fig.colorbar(cf4, ax=ax4)
    #fig.colorbar(cf1, ax=[ax1])#, orientation='horizontal', fraction=.1)
    plt.show()




simulation(configuration_file, initial_conditions, set_boundary, user_source_func, restore_index=-1)

#visualise("/Users/sjp/Desktop/boxset/tva_vsi.ini", [27])
