import numpy as np
# import itertools
import scipy
import matplotlib.pyplot as plt
import scipy.constants
# import h5py
from green_mbtools.pesto import mb
from mbanalysis import ir
import os

def read_greenfunction_from_txt_spin_blocked(time_filename: str, green_path: str, endpoint: bool = False,
                                             tol: float = 1e-12):
    """
    Reads files green_path/G_{i}_{j}.dat where i,j are *combined* indices.
    Assumes *spin-blocked* ordering:
        combined index = spin * norb + orbital
    so:
        spin = idx // norb
        orb  = idx %  norb

    Each file has two columns: tau  value  (value assumed real; stored as complex)

    Returns:
      G_tau: (ntime, 2, norb, norb) complex
      tau  : (ntime,) float
    """

    # ---- read tau grid from time_filename (first column) ----
    tau = []
    with open(time_filename, "r") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            tau.append(float(line.split()[0]))
    tau = np.asarray(tau, dtype=float)
    ntime = tau.size
    if ntime == 0:
        raise ValueError(f"No tau points found in {time_filename}")

    # ---- infer combined dimension by scanning available filenames ----
    max_idx = -1
    for name in os.listdir(green_path):
        if not (name.startswith("G_") and name.endswith(".dat")):
            continue
        parts = name[:-4].split("_")  # "G_i_j"
        if len(parts) != 3:
            continue
        try:
            i = int(parts[1]); j = int(parts[2])
        except ValueError:
            continue
        max_idx = max(max_idx, i, j)

    if max_idx < 0:
        raise ValueError(f"No G_i_j.dat files found in {green_path}")

    n_combined = max_idx + 1
    if n_combined % 2 != 0:
        raise ValueError(f"Combined dimension {n_combined} is odd; cannot split into 2 spins.")

    norb = n_combined // 2
    nspin = 2

    # ---- allocate output ----
    G_tau = np.zeros((ntime, nspin, norb, norb), dtype=complex)

    # ---- fill from files ----
    for i in range(n_combined):
        for j in range(n_combined):
            path = os.path.join(green_path, f"G_{i}_{j}.dat")
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing file: {path}")

            arr = np.loadtxt(path)
            if arr.ndim == 1:
                vals = np.array([arr[1]], dtype=float)
            else:
                vals = arr[:, 1]

            if vals.size != ntime:
                raise ValueError(
                    f"Time grid mismatch in {path}: got {vals.size} points, expected {ntime}"
                )

            # ---- spin-blocked mapping ----
            si, sj = i // norb, j // norb
            oi, oj = i % norb,  j % norb

            if si >= nspin or sj >= nspin:
                raise ValueError(
                    f"Index-to-spin mapping out of range: i={i} -> si={si}, j={j} -> sj={sj}, "
                    f"with norb={norb}, nspin={nspin}"
                )

            # only store spin-diagonal blocks
            if si == sj:
                G_tau[:, si, oi, oj] = vals.astype(complex)
            else:
                # warn if sizable spin-flip blocks exist
                vmax = float(np.max(np.abs(vals)))
                if vmax > tol:
                    print(f"Warning: spin-flip Green block in file G_{i}_{j}.dat (max |G| = {vmax})")

    return G_tau, tau



def read_greenfunction_from_txt_spin(time_filename: str, green_path: str, endpoint: bool = False):
    """
    Reads files green_path/G_{i}_{j}.dat where i,j are *combined* indices (spin+orbital).
    Each file has two columns: tau  value   (value is real in your example; can be complex if needed)

    Returns:
      G_tau: (ntime, 2, norb, norb) complex
      tau  : (ntime,) float
    """

    # ---- read tau grid from time_filename (first column) ----
    tau = []
    with open(time_filename, "r") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            tau.append(float(line.split()[0]))
    tau = np.asarray(tau, dtype=float)
    ntime = tau.size
    if ntime == 0:
        raise ValueError(f"No tau points found in {time_filename}")

    # ---- infer combined dimension by scanning available filenames ----
    # expects names like G_0_0.dat
    max_idx = -1
    for name in os.listdir(green_path):
        if not (name.startswith("G_") and name.endswith(".dat")):
            continue
        parts = name[:-4].split("_")  # strip ".dat", split "G_i_j"
        if len(parts) != 3:
            continue
        try:
            i = int(parts[1]); j = int(parts[2])
        except ValueError:
            continue
        max_idx = max(max_idx, i, j)

    if max_idx < 0:
        raise ValueError(f"No G_i_j.dat files found in {green_path}")

    n_combined = max_idx + 1
    if n_combined % 2 != 0:
        raise ValueError(f"Combined dimension {n_combined} is odd; cannot split into 2 spins.")

    norb = n_combined // 2
    nspin = 2

    # ---- allocate output ----
    G_tau = np.zeros((ntime, nspin, norb, norb), dtype=complex)

    # ---- fill from files ----
    for i in range(n_combined):
        for j in range(n_combined):
            path = os.path.join(green_path, f"G_{i}_{j}.dat")
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing file: {path}")

            # read second column (value) and ignore first (tau)
            arr = np.loadtxt(path)
            if arr.ndim == 1:
                # handle single-line edge case
                vals = np.array([arr[1]], dtype=float)
            else:
                vals = arr[:, 1]

            if vals.size != ntime:
                raise ValueError(
                    f"Time grid mismatch in {path}: got {vals.size} points, expected {ntime}"
                )

            si, sj = i % 2, j % 2
            oi, oj = i // 2, j // 2

            # only store spin-diagonal blocks
            if si == sj:
                G_tau[:, si, oi, oj] = vals.astype(complex)
            else:
                # optional: warn if there is a sizable spin-flip term
                if np.max(np.abs(vals)) > 1e-12:
                    print(f"Warning: spin-flip Green block in file G_{i}_{j}.dat (max |G| = {np.max(np.abs(vals))})")

    return G_tau, tau


def read_delta_tau_from_txt_spin(delta_file: str, beta: float, endpoint: bool = False):
    # ---------- pass 1: determine max l and max combined index ----------
    l_max = -1
    max_index = -1

    with open(delta_file, "r") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            l = int(parts[0])
            i = int(parts[1])
            m = int(parts[2])
            l_max = max(l_max, l)
            max_index = max(max_index, i, m)

    if l_max < 0:
        raise ValueError(f"No data lines found in {delta_file}")

    n_tau = l_max + 1
    n_combined = max_index + 1

    if n_combined % 2 != 0:
        raise ValueError("Combined dimension must be even (2 * norb).")

    norb = n_combined // 2
    nspin = 2

    # ---------- allocate ----------
    # shape: (tau_index, spin, orb_i, orb_j)
    delta_tau = np.zeros((n_tau, nspin, norb, norb), dtype=complex)

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
            val = float(re_str) + 1j * float(im_str)

            si, sj = i % 2, m % 2
            oi, oj = i // 2, m // 2

            # assuming spin-conserving hybridization
            if si == sj:
                delta_tau[l, si, oi, oj] = val
            else:
                # optional sanity check
                if abs(val) > 1e-12:
                    print(f"Warning: spin-flip delta at l={l} (i,m)=({i},{m}) = {val}")

    tau_delta = np.linspace(0.0, beta, n_tau, endpoint=endpoint)
    return delta_tau, tau_delta


    


def read_hopping_from_txt_spin(hopping_file):
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

    # combined dimension = 2 * norb
    n_combined = max_index + 1
    if n_combined % 2 != 0:
        raise ValueError("Combined dimension must be even (spin-degenerate).")

    norb = n_combined // 2
    nspin = 2

    hopping = np.zeros((nspin, norb, norb), dtype=complex)

    for i, j, val in entries:
        si = i % 2
        sj = j % 2
        oi = i // 2
        oj = j // 2

        # assuming spin-conserving hopping
        if si == sj:
            hopping[si, oi, oj] = val
        else:
            # optional sanity check
            if abs(val) > 1e-12:
                print(f"Warning: spin-flip hopping at ({i},{j}) = {val}")

    return hopping

    
           



if __name__ == '__main__':
    number_of_orbitals = 4
    beta = 10.0
    # for i in range(number_of_orbitals):
    #     for j in range(number_of_orbitals):
    time_filename = f'/home/orit/VS_codes1/G_0_0.dat'
    delta_file = '/home/orit/VS_codes1/delta.txt'
    hopping_file = '/home/orit/VS_codes1/hopping.txt'
    # number_of_orbitals = hopping.shape[0]
    green_tau, t_arr = read_greenfunction_from_txt_spin(time_filename,'/home/orit/VS_codes1')
    delta_tau, tau_delta = read_delta_tau_from_txt_spin(delta_file,beta)
    hopping = read_hopping_from_txt_spin(hopping_file)

    print(green_tau)
