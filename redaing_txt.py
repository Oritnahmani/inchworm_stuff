import numpy as np
# import itertools
import scipy
import matplotlib.pyplot as plt
import scipy.constants
# import h5py
from green_mbtools.pesto import mb
from mbanalysis import ir


def read_greenfunction_from_txt(number_of_orbitals, time_filename):
    with open(time_filename) as k:
        for line in k:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            t_str, ij_str = line.split()
            t_shape = len(t_str)
    green = np.zeros((t_shape, number_of_orbitals, number_of_orbitals))
    for i in range(number_of_orbitals):
        for j in range(number_of_orbitals):
            # data = np.loadtxt(f'/home/orit/VS_codes1/example/G_{i}_{j}.dat')
            times = []
            with open(f'/home/orit/green_fun/inchworm/example/G_{i}_{j}.dat') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    t_str, ij_str = line.split()            # split into 2 parts
                    green[:, i, j] = np.array([complex(x) for x in ij_str.strip().split(',')])
    return green_tau

def read_delta_tau_from_txt(delta_file):
    delta_tau = np.zeros((number_of_orbitals, number_of_orbitals))
    with open(delta_file) as k:
        for line in k:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            i , j , ij_str = line.split()
            delta_tau[int(i), int(j)] = complex(ij_str)
    return delta_tau

def fourier_trans_for_all(delta_tau,green_tau):
    

           



if __name__ == '__main__':
    number_of_orbitals = 4
    # for i in range(number_of_orbitals):
    #     for j in range(number_of_orbitals):
    time_filename = f'/home/orit/green_fun/inchworm/example/G_{0}_{0}.dat'
    delta_file = '/home/orit/green_fun/inchworm/example/delta.txt'
    green = read_greenfunction_from_txt(number_of_orbitals, time_filename)
    # delta_tau = read_delta_tau_from_txt(time_filename)
    # print(green)