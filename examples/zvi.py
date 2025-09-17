import numpy as np
from numba import jit_module
from mpi4py import MPI

from boxset.conservation_laws.adi_sbox import gamma, g0
from boxset.utils.kolmogorov_noise import add_kolmogorov_noise
from boxset.domain_decomposition import get_grid_dimensions

def _set_boundary_x(state, x, n_ghost):
    # Periodic boundary conditions
    state[:,0:n_ghost,:,:] = state[:,len(x)-2*n_ghost:len(x)-n_ghost,:,:]
    state[:,len(x)-n_ghost:len(x),:,:] = state[:,n_ghost:2*n_ghost,:,:]

    return state

def _set_boundary_y(state, x, n_ghost):
    # Periodic boundary conditions
    state[:,:,0:n_ghost,:] = state[:,:,len(x)-2*n_ghost:len(x)-n_ghost,:]
    state[:,:,len(x)-n_ghost:len(x),:] = state[:,:,n_ghost:2*n_ghost,:]

    return state

def _set_boundary_z(state, x, n_ghost):
    # Nonreflecting
    state[0,:,:,0:n_ghost] = np.exp(-x[0:n_ghost])
    state[1,:,:,0:n_ghost] = 0.0
    state[2,:,:,0:n_ghost] = 0.0
    state[3,:,:,0:n_ghost] = 0.0
    state[4,:,:,0:n_ghost] = \
      state[0,:,:,0:n_ghost]*g0/(gamma-1)

    state[0,:,:,len(x)-n_ghost:len(x)] = \
      np.exp(-x[len(x)-n_ghost:len(x)])
    state[1,:,:,len(x)-n_ghost:len(x)] = 0.0
    state[2,:,:,len(x)-n_ghost:len(x)] = 0.0
    state[3,:,:,len(x)-n_ghost:len(x)] = 0.0
    state[4,:,:,len(x)-n_ghost:len(x)] = \
      state[0,:,:,len(x)-n_ghost:len(x)]*g0/(gamma-1)

    # Nonreflecting
    #state[0,0:n_ghost] = np.exp(-x[0:n_ghost])
    #state[1,0:n_ghost] = 0.0
    #state[2,0:n_ghost] = 0.0
    #state[3,0:n_ghost] = 0.0
    #state[4,0:n_ghost] = \
    #  state[0,0:n_ghost]*g0/(gamma-1)

    #state[0,len(x)-n_ghost:len(x)] = \
    #  np.exp(-x[len(x)-n_ghost:len(x)])
    #state[1,len(x)-n_ghost:len(x)] = 0.0
    #state[2,len(x)-n_ghost:len(x)] = 0.0
    #state[3,len(x)-n_ghost:len(x)] = 0.0
    #state[4,len(x)-n_ghost:len(x)] = \
    #  state[0,len(x)-n_ghost:len(x)]*g0/(gamma-1)

    return state

def set_boundary(state, coords, dim, n_ghost):
    if dim == 0:
        return _set_boundary_x(state, coords[0], n_ghost)
    if dim == 1:
        return _set_boundary_y(state, coords[1], n_ghost)
    return _set_boundary_z(state, coords[-1], n_ghost)

jit_module(nopython=True, error_model="numpy")

def initial_conditions(coords, config):
    n_eq = 5
    noise_amp = 0.0001

    rng = np.random.default_rng(seed=1)

    U = np.zeros((n_eq, len(coords[0]), len(coords[1]), len(coords[2])))
    z = coords[-1]
    for i in range(0, len(z)):
        U[0,:,:,i] = np.exp(-z[i])

    U = add_kolmogorov_noise(U, coords, noise_amp,
                             get_grid_dimensions(config), rng)
    U[4,...] = 0.5*(U[1,...]**2 + U[2,...]**2 + U[3,...]**2)/U[0,...] +\
      U[0,...]*g0/(gamma-1)

    return U

from boxset.output.parallel import *
from boxset.coords import create_coordinates

def visualise(ini_file, save_index):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import configparser


    config = configparser.ConfigParser()
    config.read(ini_file)

    n_ghost = np.int32(config['Grid']['n_ghost'])

    coords, pos, global_dims = create_coordinates(config)
    x = coords[0]
    y = coords[1]
    z = coords[-1]

    state = initial_conditions(coords, config)

    #t, U = restore_from_dump(state, save_index, config['Output']['direc'], pos, global_dims, n_ghost)
    U = state

    #f = U[0,n_ghost:-n_ghost,n_ghost:-n_ghost]
    #cf = plt.contourf(x[n_ghost:-n_ghost], y[n_ghost:-n_ghost],
    #                  np.transpose(np.log10(f)),
    #                  levels=100, cmap='plasma')

    dens = U[0,n_ghost:-n_ghost,n_ghost:-n_ghost,16]
    vx = U[1,n_ghost:-n_ghost,n_ghost:-n_ghost,16]/dens
    vy = U[2,n_ghost:-n_ghost,n_ghost:-n_ghost,16]/dens
    vz = U[3,n_ghost:-n_ghost,n_ghost:-n_ghost,16]/dens

    minvel = np.min([vx, vy, vz])
    maxvel = np.max([vx, vy, vz])
    print(minvel, maxvel)
    levels = np.linspace(minvel, maxvel, 100)

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(10,9))

    cf1 = ax1.contourf(x[n_ghost:-n_ghost], y[n_ghost:-n_ghost],
                      np.transpose(np.log10(dens)),
                      levels=100, cmap='plasma')

    cf2 = ax2.contourf(x[n_ghost:-n_ghost], y[n_ghost:-n_ghost],
                   np.transpose(vx),
                   levels=levels, cmap='bwr')
    cf3 = ax3.contourf(x[n_ghost:-n_ghost], y[n_ghost:-n_ghost],
                   np.transpose(vy),
                   levels=levels, cmap='bwr')
    cf4 = ax4.contourf(x[n_ghost:-n_ghost], y[n_ghost:-n_ghost],
                   np.transpose(vz),
                   levels=levels, cmap='bwr')

    plt.show()

    #plt.plot(z, np.mean(U[3,:,:,:]/U[0,:,:,:], axis=(0,1)))
    #plt.plot(z, U[3,:]/U[0,:])

    #plt.yscale('log')
    #plt.xlabel('x')
    #plt.ylabel('z')
    #fig.colorbar(cf4, ax=ax4)
    #fig.colorbar(cf1, ax=[ax1])#, orientation='horizontal', fraction=.1)
    #fig.colorbar(cf4, ax=[ax1, ax2,ax3,ax4], orientation='horizontal', fraction=.1)

    x = x[n_ghost:-n_ghost]
    y = y[n_ghost:-n_ghost]
    z = z[n_ghost:-n_ghost]

    dens = U[0,n_ghost:-n_ghost,n_ghost:-n_ghost,n_ghost:-n_ghost]
    vx = U[1,n_ghost:-n_ghost,n_ghost:-n_ghost,n_ghost:-n_ghost]/dens
    vy = U[2,n_ghost:-n_ghost,n_ghost:-n_ghost,n_ghost:-n_ghost]/dens
    vz = U[3,n_ghost:-n_ghost,n_ghost:-n_ghost,n_ghost:-n_ghost]/dens

    hatvx = np.fft.rfftn(vx, s=vx.shape, axes=[0,1,2])
    hatvy = np.fft.rfftn(vy, s=vy.shape, axes=[0,1,2])
    hatvz = np.fft.rfftn(vz, s=vz.shape, axes=[0,1,2])

    kx = np.fft.fftfreq(len(x), d=x[1] - x[0])
    ky = np.fft.fftfreq(len(y), d=y[1] - y[0])
    kz = np.fft.rfftfreq(len(z), d=z[1] - z[0])

    for i in range(0, len(kx)):
        for j in range(0, len(ky)):
            for k in range(0, len(kz)):
                if np.abs(kx[i]) < np.max(kx) and np.abs(ky[j]) < np.max(ky) and np.abs(kz[k]) < np.max(kz):
                    kk = np.sqrt(kx[i]*kx[i] + ky[j]*ky[j] + kz[k]*kz[k])
                    Ekin = 2*np.pi*kk*kk*(np.abs(hatvx[i,j,k])**2 +
                                          np.abs(hatvy[i,j,k])**2 +
                                          np.abs(hatvz[i,j,k])**2)
                    if Ekin < 200:
                        print(kx[i],ky[j],kz[k], Ekin)
                    if kk != 0:
                        plt.plot([kk], [Ekin], linestyle=None, marker='o')
    plt.yscale('log')

    k = np.linspace(1,32, 100)
    plt.plot(k, 100000*k**(-5/3))

    plt.show()


from boxset.simulation import simulation

#simulation("/Users/sjp/Desktop/boxset/zvi.ini", initial_conditions, set_boundary, restore_index=-1)

visualise("/Users/sjp/Desktop/boxset/zvi.ini", 0)
