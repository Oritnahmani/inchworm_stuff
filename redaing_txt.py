import numpy as np
# import itertools
import scipy
import matplotlib.pyplot as plt
import scipy.constants
# import h5py
from green_mbtools.pesto import mb
from mbanalysis import ir


def read_greenfunction_from_txt(number_of_orbitals, time_filename,green_path):
    t_list = []
    with open(time_filename) as k:
        for line in k:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # t_str, ij_str = line.split()
            t_str = line.split()[0]
            t_list.append(float(t_str))
    t_arr = np.array(t_list)    
    t_shape = len(t_arr)
    green_tau = np.zeros((t_shape, number_of_orbitals, number_of_orbitals), dtype=complex)
    for i in range(number_of_orbitals):
        for j in range(number_of_orbitals):
            # data = np.loadtxt(f'/home/orit/VS_codes1/example/G_{i}_{j}.dat')
            with open(f'{green_path}/G_{i}_{j}.dat') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    t_str, ij_str = line.split()            # split into 2 parts
                    # TODO mabey there is a problem
                    green_tau[:, i, j] = np.array([complex(x) for x in ij_str.strip().split(',')])
    return green_tau, t_arr


def read_delta_tau_from_txt(delta_file,t_arr,number_of_orbitals):
    delta_tau = np.zeros((t_arr.shape[0],number_of_orbitals, number_of_orbitals), dtype=complex)
    with open(delta_file) as k:
        for raw_line in k:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            # parts = line.split()

            # for l in range(t_arr.shape[0]):
            #     for i in range(number_of_orbitals):
            #         for m in range(number_of_orbitals):
            l_str, i_str, m_str, ij_str_re, ij_str_im = line.split()
            # l_str , i_str , m_str , ij_str_re , ij_str_im = line.split()
            l = int(l_str)
            i = int(i_str)
            m = int(m_str)
            delta_tau[l, int(i), int(m)] = complex(float(ij_str_re) + 1j * float(ij_str_im))
    return delta_tau

    
def read_hopping_from_txt(hopping_file, number_of_orbitals):
    hopping = np.zeros(( number_of_orbitals, number_of_orbitals), dtype=complex)
    with open(hopping_file) as k:
        for raw_line in k:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            i_str, m_str, ij_str_re, ij_str_im = line.split()
            i = int(i_str)
            m = int(m_str)
            hopping[ int(i), int(m)] = complex(float(ij_str_re) + 1j * float(ij_str_im))
    return hopping
           



if __name__ == '__main__':
    number_of_orbitals = 4
    # for i in range(number_of_orbitals):
    #     for j in range(number_of_orbitals):
    time_filename = f'/home/orit/VS_codes1/example/G_0_0.dat'
    delta_file = '/home/orit/VS_codes1/example/delta.txt'
    hopping_file = '/home/orit/VS_codes1/example/hopping.txt'
    green_tau, t_arr = read_greenfunction_from_txt(number_of_orbitals, time_filename,'/home/orit/VS_codes1/example')
    delta_tau = read_delta_tau_from_txt(delta_file, t_arr, number_of_orbitals)
    hopping = read_hopping_from_txt(hopping_file, number_of_orbitals)
    print (hopping)
    # print(t_arr.shape)
    # print(green_tau)