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



def open_seet_and_insert_new_results(*, results_file, iteration, sigma_inchworm, uu_trans, X_k, mixing, ns_imp):
    fsimseet = h5py.File(results_file, 'r+')
    group = fsimseet['iter{}/Selfenergy'.format(iteration)]
    sigma_in = group['data'][()]
    nomega, ns, nk, nao_full, _ = sigma_in.shape
    assert ns_imp == ns, "Spin dimension mismatch"
    nimp = len(uu_trans)
    sigma_local_orth = np.zeros((nomega, ns, nao_full, nao_full), dtype=np.complex128)
    for i in range(nimp):
        sigma_local_orth += np.einsum('pi, tspq, qj -> wsij',
                                  uu_trans[i].conj(), sigma_inchworm[i], uu_trans[i],optimize=True)

    sigma_loc_ao = np.zeros((nw, ns, nk, nao_full, nao_full), dtype=np.complex128)
    for w in range(nw):
        for s in range(ns):
            sigma_loc_ao[w, s] = np.einsum(
                'kab, bc, kdc -> kad',
                X_k,
                sigma_local_orth[w, s],
                X_k.conj(),
                optimize=True
            )

    sigma_in += mixing * sigma_loc_ao

    group['data'][...] = sigma_in
    fsimseet.close()











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