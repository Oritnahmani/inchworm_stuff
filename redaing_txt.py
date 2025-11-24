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
            # data = np.loadtxt(f'/home/orit/VS_codes1/example/G_{i}_{j}.dat')
            times = []
            with open(f'/home/orit/VS_codes1/example/G_{i}_{j}.dat') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    t_str, ij_str = line.split()            # split into 2 parts
                    t = float(t_str)
                    times.append(t)
    return times



if __name__ == '__main__':
    number_of_orbitals = 3
    # for i in range(number_of_orbitals):
    #     for j in range(number_of_orbitals):
    # filename = f'/home/orit/VS_codes1/example/G_{i}_{j}.dat'
    times = read_array_from_txt(number_of_orbitals)
    print(times)