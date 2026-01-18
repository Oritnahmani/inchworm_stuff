import numpy as np
import itertools
import scipy
import matplotlib.pyplot as plt
import scipy.constants
import h5py



def time_to_txtfile(GW_result_path, normalize=True):
    
    with h5py.File(GW_result_path, "r") as f:
        # --- find last iteration ---
        if "iter" in f:
            it = int(f["/iter"][()])
        else:
            iters = [int(k[4:]) for k in f.keys() if k.startswith("iter")]
            if not iters:
                raise KeyError("No iter* groups found in GW HDF5 file")
            it = max(iters)

        mesh_path = f"/iter{it}/Selfenergy/mesh"
        if mesh_path not in f:
            raise KeyError(f"Mesh not found at {mesh_path}")

        t = f[mesh_path][:]

    # --- normalize or not ---
    tau = t / t[-1] if normalize else t

    with open('time_intervals.txt', "w") as f:
        for v in tau:
            f.write(f"{v:.16e}\n")

# def time_to_txtfile(GW_result_path):
#     with h5py.File(GW_result_path, 'r') as f:
#         t = f['/iter11/Selfenergy/mesh'][:]
#     f.close()
#     with open('time_intervals.txt', 'w') as f:
#         f.write(str(t/t[-1]))
#     f.close()

if __name__ == '__main__':
    GW_result_path = '/home/orit/VS_codes1/NiO_GW_iter14.h5'
    time_to_txtfile(GW_result_path)
