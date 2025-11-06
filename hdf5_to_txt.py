import numpy as np
import itertools
import scipy
import matplotlib.pyplot as plt
import scipy.constants
import h5py




def time_to_txtfile(sim_h5):
    with h5py.File(GW_result_path, 'r') as f:
        t = f['sim.h5/iter11/Selfenergy/mesh'][:]

        

    return t



if __name__ == '__main__':
    GW_result_path = '/home/orit/VS_codes1/green-mbtools/tests/test_data/H2_GW/sim.h5'