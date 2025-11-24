import numpy as np
import itertools
import scipy
import matplotlib.pyplot as plt
import scipy.constants
import h5py
from green_mbtools.pesto import mb
from mbanalysis import ir


def read_array_from_txt(number_of_orbitals):
    data = np.array
    for i in range(number_of_orbitals):
        for j in range(number_of_orbitals):
            data = np.loadtxt(f'/home/orit/VS_codes1/example/G_{i}_{j}.dat')
    return data



if __name__ == '__main__':
    number_of_orbitals = 3
    # for i in range(number_of_orbitals):
    #     for j in range(number_of_orbitals):
    # filename = f'/home/orit/VS_codes1/example/G_{i}_{j}.dat'
    
