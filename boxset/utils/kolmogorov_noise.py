import numpy as np
from numba import jit_module


def add_kolmogorov_noise(state, coords, amplitude, global_dims, rng):
    max_k = []
    for dim in range(0, len(coords)):
        max_k.append(int((global_dims[dim])/2))

    # Normalized coordinates so that Lx = Ly = Lz = 1
    x = coords[0]/global_dims[0]/(coords[0][1] - coords[0][0])
    y = coords[1]/global_dims[1]/(coords[1][1] - coords[1][0])
    z = coords[2]/global_dims[2]/(coords[2][1] - coords[2][0])

    for kx in range(-max_k[0], max_k[0]):
        for ky in range(-max_k[1], max_k[1]):
            for kz in range(1, max_k[2]):
                k = np.sqrt(kx*kx + ky*ky + kz*kz)

                E = amplitude*k**(-5/3)
                theta_1 = rng.uniform(low=-np.pi, high=np.pi)
                theta_2 = rng.uniform(low=-np.pi, high=np.pi)
                phi = rng.uniform(low=0, high=2*np.pi)

                alpha = np.sqrt(0.5*E*k**(-2)/np.pi)*np.exp(1j*theta_1)*np.cos(phi)
                beta = np.sqrt(0.5*E*k**(-2)/np.pi)*np.exp(1j*theta_2)*np.sin(phi)

                hatvx= (alpha*k*ky + beta*kx*kz)/(k*np.sqrt(kx*kx + ky*ky) + 1.0e-30)
                hatvy= (beta*ky*kz - alpha*k*kx)/(k*np.sqrt(kx*kx + ky*ky) + 1.0e-30)
                hatvz = -(beta*np.sqrt(kx*kx + ky*ky))/k
                if kx == 0 and ky == 0:
                    hatvx= (alpha + beta*kz/k)/np.sqrt(2)
                    hatvy= (beta*kz/k - alpha)/np.sqrt(2)

                for n in range(0, len(coords[2])):
                    for m in range(0, len(coords[1])):
                        for l in range(0, len(coords[0])):
                            e = np.exp(-2j*np.pi*(kx*x[l] + ky*y[m] + kz*z[n]))
                            state[1,l,m,n] += state[0,l,m,n]*2*np.real(hatvx*e)
                            state[2,l,m,n] += state[0,l,m,n]*2*np.real(hatvy*e)
                            state[3,l,m,n] += state[0,l,m,n]*2*np.real(hatvz*e)

    # kz = 0
    for kx in range(-max_k[0], max_k[0]):
        for ky in range(1, max_k[1]):
            k = np.sqrt(kx*kx + ky*ky)

            E = amplitude*k**(-5/3)
            theta_1 = rng.uniform(low=-np.pi, high=np.pi)
            theta_2 = rng.uniform(low=-np.pi, high=np.pi)
            phi = rng.uniform(low=0, high=2*np.pi)

            alpha = np.sqrt(0.5*E*k**(-2)/np.pi)*np.exp(1j*theta_1)*np.cos(phi)
            beta = np.sqrt(0.5*E*k**(-2)/np.pi)*np.exp(1j*theta_2)*np.sin(phi)

            hatvx= alpha*k*ky/(k*np.sqrt(kx*kx + ky*ky) + 1.0e-30)
            hatvy= -alpha*k*kx/(k*np.sqrt(kx*kx + ky*ky) + 1.0e-30)
            hatvz = -(beta*np.sqrt(kx*kx + ky*ky))/k

            for n in range(0, len(coords[2])):
                for m in range(0, len(coords[1])):
                    for l in range(0, len(coords[0])):
                        e = np.exp(-2j*np.pi*(kx*x[l] + ky*y[m]))
                        state[1,l,m,n] += state[0,l,m,n]*2*np.real(hatvx*e)
                        state[2,l,m,n] += state[0,l,m,n]*2*np.real(hatvy*e)
                        state[3,l,m,n] += state[0,l,m,n]*2*np.real(hatvz*e)

    # kz = ky = 0
    for kx in range(1, max_k[0]):
        k = np.sqrt(kx*kx)

        E = amplitude*k**(-5/3)
        theta_1 = rng.uniform(low=-np.pi, high=np.pi)
        theta_2 = rng.uniform(low=-np.pi, high=np.pi)
        phi = rng.uniform(low=0, high=2*np.pi)

        alpha = np.sqrt(0.5*E*k**(-2)/np.pi)*np.exp(1j*theta_1)*np.cos(phi)
        beta = np.sqrt(0.5*E*k**(-2)/np.pi)*np.exp(1j*theta_2)*np.sin(phi)

        hatvx= 0.0
        hatvy= -alpha*k*kx/(k*np.sqrt(kx*kx) + 1.0e-30)
        hatvz = -(beta*np.sqrt(kx*kx))/k

        for n in range(0, len(coords[2])):
            for m in range(0, len(coords[1])):
                for l in range(0, len(coords[0])):
                    e = np.exp(-2j*np.pi*kx*x[l])
                    state[1,l,m,n] += state[0,l,m,n]*2*np.real(hatvx*e)
                    state[2,l,m,n] += state[0,l,m,n]*2*np.real(hatvy*e)
                    state[3,l,m,n] += state[0,l,m,n]*2*np.real(hatvz*e)

    return state

jit_module(nopython=True, error_model="numpy")
