import argparse
import time
import re
from pathlib import Path
import numpy as np
import scipy.interpolate
import h5py
from green_mbtools.pesto import mb
from mbanalysis import ir
from inchworm_stuff.redaing_txt import read_greenfunction_from_txt, read_delta_tau_from_txt, read_hopping_from_txt
from data_analyzing_from_mbpt.processing_after_inchworm import read_mu, interpolation, fourier_transform, dyson_green_to_sigma_split_omega , save_sigma_split_to_hdf5, read_beta_from_seet_sbatch



def build_full_space_sigma_from_impurity(
    *,
    sigma_imp: np.ndarray,         # (nomega, ns, nao_imp, nao_imp)
    uu: np.ndarray,                # (nao_imp, nao_full) OR (nao_full, nao_imp)
    X_k: np.ndarray,               # (nk, nao_full, nao_full)
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      sigma_full_orth: (nomega, ns, nao_full, nao_full)
      sigma_full_ao:   (nomega, ns, nk, nao_full, nao_full)
    """
    nomega, ns, nao_imp, _ = sigma_imp.shape
    nk, nao_full, _ = X_k.shape

    # --- embed impurity -> full orth: U^† Σ_imp U ---
    # try both UU orientations
    try:
        # UU: (p,i)
        sigma_full_orth = np.einsum(
            "pi, wspq, qj -> wsij",
            uu.conj(), sigma_imp, uu,
            optimize=True
        )
    except ValueError:
        # UU: (i,p)
        sigma_full_orth = np.einsum(
            "ip, wspq, jq -> wsij",
            uu.conj(), sigma_imp, uu,
            optimize=True
        )

    # --- rotate orth -> AO(k) ---
    sigma_full_ao = np.zeros((nomega, ns, nk, nao_full, nao_full), dtype=np.complex128)
    for w in range(nomega):
        for s in range(ns):
            sigma_full_ao[w, s] = np.einsum(
                "kab, bc, kdc -> kad",
                X_k, sigma_full_orth[w, s], X_k.conj(),
                optimize=True
            )

    return sigma_full_orth, sigma_full_ao






def insert_sigma_into_seet_file(
    *,
    results_file: Path,
    iteration: int,
    sigma_add_ao: np.ndarray,   # (nomega, ns, nk, nao_full, nao_full)
    mixing: float,
):
    with h5py.File(results_file, "r+") as fs:
        group = fs[f"iter{iteration}/Selfenergy"]
        sigma_in = group["data"][()]

        if sigma_in.shape != sigma_add_ao.shape:
            raise ValueError(f"Shape mismatch: SEET {sigma_in.shape} vs add {sigma_add_ao.shape}")

        sigma_in += mixing * sigma_add_ao
        group["data"][...] = sigma_in





def main():
    ap = argparse.ArgumentParser(description="Wait for inchworm G files, then compute selfenergy_iw.")
    ap.add_argument("--sbatch-dir", type=Path, default=None,
                help="Directory containing the SEET sbatch file (default: run-dir)")
    ap.add_argument("--sbatch-name", type=Path, default=Path("sbatch_seet"),
                help="Filename of the SEET sbatch script inside sbatch-dir")

    ap.add_argument("--run-dir",type=Path, default=".", help="Directory where inchworm output files are located.")
    ap.add_argument("--time_intervals",type=Path, default="time_intervals.txt", help="Path (relative to run-dir) for time_intervals.txt")
    ap.add_argument("--delta-file",type=Path, default="delta.txt", help="Path (relative to run-dir) for delta.txt")
    ap.add_argument("--hopping-file", type=Path, default="hopping.txt", help="Path (relative to run-dir) for hopping.txt")
    # Mu inputs (these are in your original script; make them arguments so it works on cluster)
    #TODO
    ap.add_argument("--nio-gw-h5", type=Path, default="NiO_GW_iter14.h5", help="Path to NiO_GW_iter*.h5 (default: NiO_GW_iter14.h5)")
    ap.add_argument("--input-h5", type=Path, default="input.h5", help="Path to input.h5 (grid info)")
    # TODO
    ap.add_argument("--ir-grid", type=Path, default="1e5.h5" , help="Path to IR grid h5 file, e.g. 1e5.h5")

    # Green naming
    ap.add_argument("--g-dir", type=Path, default=None,
                help="Directory containing G_{i}_{j}.dat files (default: run-dir)")
    ap.add_argument("--g-pattern", type=str, default="G_{i}_{j}.dat",
                help='Green filename pattern, e.g. "G_{i}_{j}.dat"')


    # Output
    ap.add_argument("--out-npy", default="selfenergy_iw.npy", help="Output numpy file (saved in run-dir).")
    ap.add_argument("--mixing", type=float, default=0.5, help="Mixing parameter for updating selfenergy.")
    args = ap.parse_args()


    run_dir = args.run_dir.expanduser().resolve() 
    def resolve(p: Path) -> Path: 
        p = p.expanduser() 
        return (run_dir / p).resolve() if not p.is_absolute() else p.resolve() 



    time_filename = resolve(args.time_intervals) 
    delta_file = resolve(args.delta_file) 
    hopping_file = resolve(args.hopping_file) 
    nio_gw_h5 = resolve(args.nio_gw_h5) 
    input_h5 = resolve(args.input_h5) 
    ir_grid = resolve(args.ir_grid)

    g_dir = resolve(args.g_dir) if args.g_dir is not None else run_dir
    print(g_dir)
    sbatch_dir = resolve(args.sbatch_dir) if args.sbatch_dir is not None else run_dir
    sbatch_path = (sbatch_dir / args.sbatch_name).expanduser().resolve()


    # 1) Wait until ALL G_{i}_{j} files exist & stable
    # g_files = find_g_files(run_dir, args.orbitals, args.g_pattern)
    # print(f"[watch] Waiting for {len(g_files)} Green files like: {run_dir / args.g_pattern.format(i=0,j=0)}")
    # wait_for_files(g_files, poll_s=5.0, stable_checks=2, stable_interval_s=2.0)
    # print("[watch] Green files present and stable.")

    # 2) Compute selfenergy

    beta = read_beta_from_seet_sbatch(sbatch_path)
    mu = read_mu(str(nio_gw_h5))



    hopping = read_hopping_from_txt(str(hopping_file))
    num_orbitals = hopping.shape[0]

    green_tau, t_arr = read_greenfunction_from_txt(num_orbitals, str(time_filename), str(g_dir))
    delta_tau, tau_delta_original = read_delta_tau_from_txt(str(delta_file), num_orbitals,beta)

    new_delta_tau =  interpolation(tau_delta_original, delta_tau, t_arr, kind="linear")


    delta_omega, green_omega = fourier_transform(beta, str(ir_grid), new_delta_tau, green_tau)

    selfenergy_iw, sigma_static, sigma_dynamic_iw, my_ir = dyson_green_to_sigma_split_omega(
    beta=beta,
    green_omega=green_omega,
    number_of_orbitals=num_orbitals,
    ir_grid_path=str(ir_grid),
    mu=mu,
    delta_omega=delta_omega,
    hopping=hopping,
    avg_slice=None,          # or e.g. slice(10, 200)
    use_weights=False)


    sigma_file = run_dir / "selfenergy_split.h5"
    save_sigma_split_to_hdf5(
        sigma_file,
        sigma_static,
        sigma_dynamic_iw,
        sigma_iw=selfenergy_iw
    )