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
            t_str = t_str.strip().strip('[],')   # remove [ ] and commas
            if t_str:                            # skip empty tokens
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

def read_delta_tau_from_txt(delta_file: str, number_of_orbitals: int,beta: float, endpoint: bool = False):
    # ---------- pass 1: determine max l ----------
    l_max = -1
    with open(delta_file, "r") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 5:
                continue  # or raise, if you prefer strict
            l = int(parts[0])
            if l > l_max:
                l_max = l

    if l_max < 0:
        raise ValueError(f"No data lines found in {delta_file}")

    n_tau = l_max + 1

    # ---------- allocate ----------
    delta_tau = np.zeros((n_tau, number_of_orbitals, number_of_orbitals), dtype=complex)

    # ---------- pass 2: fill ----------
    with open(delta_file, "r") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            l_str, i_str, m_str, re_str, im_str = line.split()
            l = int(l_str)
            i = int(i_str)
            m = int(m_str)
            delta_tau[l, i, m] = float(re_str) + 1j * float(im_str)

    tau_delta = np.linspace(0.0, beta, n_tau, endpoint=endpoint)


    return delta_tau, tau_delta

    
def read_hopping_from_txt(hopping_file):
    entries = []
    max_index = -1

    with open(hopping_file) as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            i_str, j_str, re_str, im_str = line.split()
            i = int(i_str)
            j = int(j_str)

            val = complex(float(re_str), float(im_str))
            entries.append((i, j, val))

            max_index = max(max_index, i, j)

    norb = max_index + 1
    hopping = np.zeros((norb, norb), dtype=complex)

    for i, j, val in entries:
        hopping[i, j] = val
    return hopping
           



if __name__ == '__main__':
    number_of_orbitals = 4
    # for i in range(number_of_orbitals):
    #     for j in range(number_of_orbitals):
    time_filename = f'/home/orit/VS_codes1/example/G_0_0.dat'
    delta_file = '/home/orit/VS_codes1/example/delta.txt'
    hopping_file = '/home/orit/VS_codes1/example/hopping.txt'
    hopping = read_hopping_from_txt(hopping_file)
    number_of_orbitals = hopping.shape[0]
    green_tau, t_arr = read_greenfunction_from_txt(number_of_orbitals, time_filename,'/home/orit/VS_codes1/example')
    delta_tau = read_delta_tau_from_txt(delta_file, t_arr, number_of_orbitals)
    print (hopping)
    # print(t_arr.shape)
    # print(green_tau)