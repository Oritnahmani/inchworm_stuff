import numpy as np
import itertools
import scipy
import matplotlib.pyplot as plt
import scipy.constants
from hamiltonian import get_hamiltonian_matrix, get_ladder_operators, get_noninteracting_hamiltonian, add_interactions_to_hamiltonian, generate_interaction_tensor
#from perturbation_theory_hartree_fock import calc_green_fun_hartree,  calc_green_fun_fock
from green_function_zeroorder import green, Green
from numpy.linalg import eig
from scipy.linalg import expm
from time import perf_counter
from copy import copy
from GF import GreensFunction
from Dayson_series import calc_sigma_c, calc_bold_green_c, calc_dyson_c
from green_function_zertoorder_mitsubara import Green_iomega
import h5py
from config_params import SystemParameters

def hdf5_file_create(array):
    f = h5py.File('G_tau.hdf5','w')
    g_tau = f.create_group("G_tau_two_particles")
    g_tau.create_dataset('my_array', data=array)

    return g_tau


def fft_on_tensor(file_path,tau,beta):
    with h5py.File(file_path, 'r') as f:
        group = f['G_tau_two_particles']
        data = group['my_array'][:]
        expk = np.exp(1j * np.pi * tau[:-1] / beta)[:, None, None]
        g_tilde = data[:-1] *expk
        g_omega = beta * np.fft.fft(g_tilde,axis=0)[:len(tau)//2]
    return g_omega

def tail(file_path,tau,beta, omega,g_omega):
    with h5py.File(file_path, 'r') as f:
        group = f['G_tau_two_particles']
        data = group['my_array'][:]
        dgtau_dtau = np.gradient(data, tau)
        dgtau_dtau_squered = np.gradient(dgtau_dtau, tau)
        c_1 = ( data[len(data) -1 ] - data[0] ) / ( omega * 1j )
        c_2 = ( dgtau_dtau[len(data) -1 ] - dgtau_dtau[0] ) / ( (omega * 1j)**2 )
        c_3 = ( dgtau_dtau_squered[len(data) -1 ] - dgtau_dtau_squered[0] ) / ( (omega * 1j)**3)
        tail = c_1 + c_2 + c_3 
    return(tail)

# def full_fft(tail,g_omega):
#     g_omega_full = g_omega






# # From imaginary time to imaginary frequency
# def ft(g: np.ndarray, tau: np.ndarray, wn: np.ndarray, beta: float, tail=[0, 1,0, 0]) -> np.ndarray:

#     # get the shape of the green's function for one omega
#     shape = get_shape(g[1])
#     g = g - hf_exp_tau1(shape, tau, beta, tail)

#     # mapping the fourier transform
#     expk = np.exp(1j * np.pi * tau[:-1] / beta)[:, None, None]
#     g_cut = g[:-1]*expk

#     g_omega = beta * np.fft.ifft(g_cut, axis=0)[:len(wn)]

#     # Adding the tail
#     g_omega = g_omega + hf_exp_freq1(shape, wn, tail)
#     return g_omega








if __name__ == '__main__':
    epsilon = 0
    v = 1
    J = 1.0
    n = 2
    beta = 1.0
    num_times = 100
    omegas = np.zeros(num_times//2)
    for i in range(len(omegas)):
        omegas[i] = ( ( 2 * i + 1 ) * np.pi ) / beta 
    dt = beta/num_times
    t1 = np.linspace(0, beta, num_times)
    t3 = np.linspace(0, beta, num_times)
    params = SystemParameters(n, dt, beta, t1)
    t_start = perf_counter()
    h = get_hamiltonian_matrix(epsilon, v, n)
    a, adag = get_ladder_operators(n)
    H0 = get_noninteracting_hamiltonian(h, a, adag)


    G_0_tau = GreensFunction(np.array([green(h, beta, t) for t in t1]))
    G_0_omega = np.array([Green_iomega(h, a, adag, beta, omega) for omega in omegas])


    iter_num = 10
    # G = calc_bold_green_c(iter_num,G_0, U,G.__copy__(),params)
    # h5_filename = 'my_data.h5'
    # print(G.G[:,:,:])
    g_tau = hdf5_file_create(G_0_tau.G[:,:,:])
    g_omega = fft_on_tensor('G_tau.hdf5',t1,beta)
    g_omega_final = np.array([tail('G_tau.hdf5',dt,beta) for omega in omegas])

    plt.figure()
    plt.plot(G_0_tau.G[:,0,0])

    plt.figure()
    plt.plot(omegas, G_0_omega[:,0,0].imag ,marker=".", label="exact?")

    plt.plot(omegas, g_omega[:,0,0].imag , 'k--',label="FT(G_tau)")
    plt.legend()
    plt.show()