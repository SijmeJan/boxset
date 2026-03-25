import numpy as np
from numba import jit_module

def _set_boundary_x(state, x, n_ghost):
    # Periodic boundary conditions
    state[:,0:n_ghost,:] = state[:,len(x)-2*n_ghost:len(x)-n_ghost,:]
    state[:,len(x)-n_ghost:len(x),:] = state[:,n_ghost:2*n_ghost,:]

    return state

def _set_boundary_y(state, x, n_ghost):
    state[:,:,0:n_ghost] = 0.0
    state[:,:,len(x)-n_ghost:len(x)] = 0.0

    return state

def set_boundary(state, coords, dim, n_ghost):
    if dim == 0:
        return _set_boundary_x(state, coords[0], n_ghost)
    return _set_boundary_y(state, coords[1], n_ghost)

def user_source_func(state, coords, time):
    return 0*state

jit_module(nopython=True, error_model="numpy")

def initial_conditions(coords, config):
    n_eq = 1

    U = np.zeros((n_eq, len(coords[0]), len(coords[1])))


    U[0,10:int(len(coords[0])/2),len(coords[1])-11:-1] = 1.0
    U[0,int(len(coords[0])/2):-10,1:11] = 1.0

    return U


from boxset.output.parallel import *
from boxset.coords import create_coordinates

def visualise(direc, save_index):
    import matplotlib.pyplot as plt
    import configparser

    config = configparser.ConfigParser()
    config.read(direc + 'dust_boltz.ini')

    n_ghost = np.int32(config['Grid']['n_ghost'])

    coords, pos, global_dims, periodic_flags = create_coordinates(config)
    x = coords[0]
    y = coords[1]

    state = initial_conditions(coords, config)

    t, U = restore_from_dump(state, 0, config['Output']['direc'], pos, global_dims, n_ghost)


    f, axs = plt.subplots(3, 2, sharex=True)

    fd = U[0,n_ghost:-n_ghost,n_ghost:-n_ghost]
    cf = axs[0,0].contourf(x[n_ghost:-n_ghost],
                           y[n_ghost:-n_ghost],
                           np.transpose(fd),
                           levels=100, cmap='terrain')

    axs[0,0].set_ylabel('Y')
    #plt.colorbar(cf, ax=ax1)

    dens = np.sum(U[0,n_ghost:-n_ghost,n_ghost:-n_ghost], axis=1)
    momx = np.sum(U[0,n_ghost:-n_ghost,n_ghost:-n_ghost]*y[n_ghost:-n_ghost], axis=1)

    axs[0,0].set_title(r'$T=0$')
    axs[1,0].plot(x[n_ghost:-n_ghost], dens)
    axs[1,0].set_ylabel(r'$\rho_{\rm d}$')
    axs[2,0].plot(x[n_ghost:-n_ghost], momx)
    axs[2,0].set_xlabel('X')
    axs[2,0].set_ylabel(r'$\rho_{\rm d}v_{\rm d}$')

    axs[1,0].annotate("", xytext=(0.25, 5), xy=(0.75, 5),
                      arrowprops=dict(arrowstyle="-|>"))
    axs[1,0].annotate("", xytext=(1.75, 5), xy=(1.25, 5),
                      arrowprops=dict(arrowstyle="-|>"))
    axs[1,0].set_ylim([0,22])
    axs[2,0].set_ylim([-8,8])

    t, U = restore_from_dump(state, save_index, config['Output']['direc'], pos, global_dims, n_ghost)

    fd = U[0,n_ghost:-n_ghost,n_ghost:-n_ghost]
    cf = axs[0,1].contourf(x[n_ghost:-n_ghost],
                           y[n_ghost:-n_ghost],
                           np.transpose(fd),
                           levels=100, cmap='terrain')

    axs[0,1].set_ylabel('Y')
    #plt.colorbar(cf, ax=ax1)

    dens = np.sum(U[0,n_ghost:-n_ghost,n_ghost:-n_ghost], axis=1)
    momx = np.sum(U[0,n_ghost:-n_ghost,n_ghost:-n_ghost]*y[n_ghost:-n_ghost], axis=1)

    axs[0,1].set_title(r'$T=1.3$')
    axs[1,1].plot(x[n_ghost:-n_ghost], dens)
    axs[1,1].set_ylabel(r'$\rho_{\rm d}$')
    axs[2,1].plot(x[n_ghost:-n_ghost], momx)
    axs[2,1].set_xlabel('X')
    axs[2,1].set_ylabel(r'$\rho_{\rm d}v_{\rm d}$')

    axs[1,1].annotate("", xytext=(0.7, 5), xy=(0.4, 5),
                      arrowprops=dict(arrowstyle="-|>"))
    axs[1,1].annotate("", xytext=(1.3, 5), xy=(1.6, 5),
                      arrowprops=dict(arrowstyle="-|>"))
    axs[1,1].set_ylim([0,22])
    axs[2,1].set_ylim([-8,8])

    axs[1,1].plot([1,1],[0,22], linestyle='--')
    axs[2,1].plot([0.01,1.99],[0,0], linestyle='--')

    #for ax in axs.flat:
    #    ax.set(xlabel='x-label', ylabel='y-label')

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()
    plt.show()

from boxset.simulation import simulation

#simulation("/Users/sjp/Desktop/boxset/dust_boltz.ini", initial_conditions, set_boundary, user_source_func, restore_index=-1)

for n in range(1,10):
    print(n, (n-1)**n * n**(n-1))

visualise('/Users/sjp/Desktop/boxset/', 13)
