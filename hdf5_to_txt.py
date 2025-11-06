import numpy as np
import itertools
import scipy
import matplotlib.pyplot as plt
import scipy.constants
import h5py




def readef read_GW_file(sim_h5):
    with h5py.File(inputh5_path, 'r') as f:
        t = f['sim.h5/iter11/Selfenergy/mesh'][:]

        

    return t



if __name__ == '__main__':
    tau_grid_path = '/home/orit/VS_codes1/green-mbtools/tests/test_data/ir_grid/1e4.h5'
    GW_result_path = '/home/orit/VS_codes1/green-mbtools/tests/test_data/H2_GW/sim.h5'