from pathlib import Path
import numpy as np
import h5py
from data_analyzing_from_inchworm import processing_after_inchworm as proc


def load_transform_data_ibz(transform_file: Path, input_file: Path):
    with h5py.File(transform_file, "r") as ft:
        nimp = int(ft["nimp"][()])
        X_inv_k_full = ft["X_inv_k"][()]
        uu_trans = [(ft[f"{i}/UU"][()] + 0j) for i in range(nimp)]

    with h5py.File(input_file, "r") as fin:
        ir_list = fin["grid/ir_list"][()]

    X_inv_k_ibz = X_inv_k_full[ir_list]
    return nimp, uu_trans, X_inv_k_ibz


def embed_all_impurities_to_full_orth(
    *,
    sigma_imp_list,   # list of arrays, each (ntau, ns, nao_imp_i, nao_imp_i)
    uu_trans,         # list of arrays, one per impurity
):
    """
    Embed all impurity dynamic self-energies into the full orthogonal space and sum them.

    Returns
    -------
    sigma_full_orth : np.ndarray
        Shape (ntau, ns, nao_full, nao_full)
    """
    if len(sigma_imp_list) != len(uu_trans):
        raise ValueError(
            f"Need one sigma per impurity: got {len(sigma_imp_list)} sigmas and {len(uu_trans)} UU blocks."
        )

    # Infer output size from first projector and first sigma
    sigma0 = sigma_imp_list[0]
    ntau, ns = sigma0.shape[:2]

    # Figure out nao_full from UU orientation
    uu0 = uu_trans[0]
    nao_full = max(uu0.shape)

    sigma_full_orth = np.zeros((ntau, ns, nao_full, nao_full), dtype=np.complex128)

    for sigma_imp, uu in zip(sigma_imp_list, uu_trans):
        # sigma_imp: (w, s, p, q)
        try:
            # UU: (p, i)
            sigma_full_orth += np.einsum(
                "pi, wspq, qj -> wsij",
                uu.conj(),
                sigma_imp,
                uu,
                optimize=True,
            )
        except ValueError:
            # UU: (i, p)
            sigma_full_orth += np.einsum(
                "ip, wspq, jq -> wsij",
                uu.conj(),
                sigma_imp,
                uu,
                optimize=True,
            )

    return sigma_full_orth


def embed_all_sigma_inf_to_full_orth(
    *,
    sigma_inf_list,   # list of arrays, each (ns, nao_imp_i, nao_imp_i)
    uu_trans,         # list of arrays
):
    """
    Embed all impurity static self-energies Sigma1 into the full orthogonal space and sum them.

    Returns
    -------
    sigma_inf_full_orth : np.ndarray
        Shape (ns, nao_full, nao_full)
    """
    if len(sigma_inf_list) != len(uu_trans):
        raise ValueError(
            f"Need one sigma_inf per impurity: got {len(sigma_inf_list)} static blocks and {len(uu_trans)} UU blocks."
        )

    ns = sigma_inf_list[0].shape[0]
    uu0 = uu_trans[0]
    nao_full = max(uu0.shape)

    sigma_inf_full_orth = np.zeros((ns, nao_full, nao_full), dtype=np.complex128)

    for sigma_inf, uu in zip(sigma_inf_list, uu_trans):
        try:
            # UU: (p, i)
            sigma_inf_full_orth += np.einsum(
                "pi, spq, qj -> sij",
                uu.conj(),
                sigma_inf,
                uu,
                optimize=True,
            )
        except ValueError:
            # UU: (i, p)
            sigma_inf_full_orth += np.einsum(
                "ip, spq, jq -> sij",
                uu.conj(),
                sigma_inf,
                uu,
                optimize=True,
            )

    return sigma_inf_full_orth


def rotate_dynamic_orth_to_ao_k(*, sigma_full_orth: np.ndarray, X_k: np.ndarray):
    """
    Rotate dynamic sigma from orthogonal basis to AO basis per k-point.

    sigma_full_orth: (ntau, ns, nao_full, nao_full)
    X_k:             (nk, nao_full, nao_full)

    returns:
    sigma_full_ao:   (ntau, ns, nk, nao_full, nao_full)
    """
    ntau, ns, nao_full, _ = sigma_full_orth.shape
    nk, nao_full2, _ = X_k.shape
    if nao_full != nao_full2:
        raise ValueError(f"nao_full mismatch: sigma has {nao_full}, X_k has {nao_full2}")

    X_k_H = X_k.conj().transpose(0, 2, 1)

    sigma_full_ao = np.zeros((ntau, ns, nk, nao_full, nao_full), dtype=np.complex128)
    for w in range(ntau):
        sigma_full_ao[w] = np.einsum(
            "kab, sbc, kcd -> skad",
            X_k,
            sigma_full_orth[w],
            X_k_H,
            optimize=True,
        )
    return sigma_full_ao





def rotate_dynamic_orth_to_ao_k(*, sigma_full_orth: np.ndarray, X_k: np.ndarray):
    ntau, ns, nao_full, _ = sigma_full_orth.shape
    nk, nao_full2, _ = X_k.shape
    if nao_full != nao_full2:
        raise ValueError(f"nao_full mismatch: sigma has {nao_full}, X_k has {nao_full2}")

    X_k_H = X_k.conj().transpose(0, 2, 1)

    sigma_full_ao = np.zeros((ntau, ns, nk, nao_full, nao_full), dtype=np.complex128)
    for w in range(ntau):
        sigma_full_ao[w] = np.einsum(
            "kab, sbc, kcd -> skad",
            X_k,
            sigma_full_orth[w],
            X_k_H,
            optimize=True,
        )
    return sigma_full_ao


def rotate_static_orth_to_ao_k(*, sigma_inf_full_orth: np.ndarray, X_k: np.ndarray):
    """
    Rotates static sigma from orthogonal basis to AO basis per k-point.
    """
    ns, nao_full, _ = sigma_inf_full_orth.shape
    nk, _, _ = X_k.shape

    X_k_H = X_k.conj().transpose(0, 2, 1)
    sigma_inf_full_ao = np.zeros((ns, nk, nao_full, nao_full), dtype=np.complex128)
    
    sigma_inf_full_ao[...] = np.einsum(
        "kab, sbc, kcd -> skad",
        X_k,
        sigma_inf_full_orth,
        X_k_H,
        optimize=True,
    )
    return sigma_inf_full_ao


def insert_sigma_into_seet_file(
    *,
    results_file: Path,
    iteration: int,
    sigma_add_ao: np.ndarray,       
    sigma_inf_add_ao: np.ndarray,   
    mixing: float,
):
    """
    Directly writes/rewrites the solver results into the current iter{iteration}.
    If you are just starting and the iteration doesn't exist, it creates it dynamically.
    """
    # Open in 'a' mode so it can read, write, or create groups if missing
    with h5py.File(results_file, "a") as fs:
        iter_key = f"iter{iteration}"

        # Create the iteration group if it's the first time running
        if iter_key not in fs:
            print(f"Creating missing iteration group: {iter_key}")
            iter_grp = fs.create_group(iter_key)
        else:
            iter_grp = fs[iter_key]

        # Ensure the Selfenergy subgroup path exists
        if "Selfenergy" not in iter_grp:
            se_grp = iter_grp.create_group("Selfenergy")
        else:
            se_grp = iter_grp["Selfenergy"]

        # Helper function to completely rewrite/create a clean dataset
        def rewrite_dataset(group, name, data):
            if name in group:
                del group[name]  # Wipe old data structure to avoid size mismatches
            group.create_dataset(name, data=data)

        # Completely rewrite the solver outputs into the current iteration datasets
        rewrite_dataset(se_grp, "data", sigma_add_ao)
        rewrite_dataset(iter_grp, "Sigma1", sigma_inf_add_ao)

    print(f"Successfully rewrote solver results into {results_file} -> {iter_key}")


def main(): 
    ap = argparse.ArgumentParser(description="Embed and rotate solver results back to main problem.")
    ap.add_argument("--transform-file", type=Path, required=True,
                    help="Path to transform.h5 (contains nimp, X_k, UU)")
    ap.add_argument("--results-file", type=Path, required=True,
                    help="SEET results HDF5 file to update")
    ap.add_argument("--iteration", type=int, required=True,
                    help="SEET iteration index to update (iter{iteration}/Selfenergy)")
    ap.add_argument("--impurity-index", type=int, default=0,
                    help="Which impurity block in transform.h5 to use")
    ap.add_argument("--save-full-sigma", type=Path, default=None,
                    help="Optional: save Sigma_full_orth and Sigma_full_ao here")

    args = ap.parse_args()

    sigma_imp_all, sigma_inf_all = proc.run_processing(args)

    sigma_imp_list = [sigma_imp_all[i] for i in range(sigma_imp_all.shape[0])]
    sigma_inf_list = [sigma_inf_all[i] for i in range(sigma_inf_all.shape[0])]

    nimp, uu_trans, X_k = load_transform_data_ibz(
        args.transform_file,
        args.input_h5,
    )

    if len(sigma_imp_list) != nimp:
        raise ValueError(
            f"transform.h5 says nimp={nimp}, but processing returned "
            f"{len(sigma_imp_list)} impurity blocks"
        )

    sigma_full_orth = embed_all_impurities_to_full_orth(
        sigma_imp_list=sigma_imp_list,
        uu_trans=uu_trans,
    )

    sigma_inf_full_orth = embed_all_sigma_inf_to_full_orth(
        sigma_inf_list=sigma_inf_list,
        uu_trans=uu_trans,
    )

    sigma_full_ao = rotate_dynamic_orth_to_ao_k(
        sigma_full_orth=sigma_full_orth,
        X_k=X_k,
    )

    sigma_inf_full_ao = rotate_static_orth_to_ao_k(
        sigma_inf_full_orth=sigma_inf_full_orth,
        X_k=X_k,
    )

    if args.save_full_sigma is not None:
        with h5py.File(args.save_full_sigma, "w") as f:
            f.create_dataset("Sigma_full_orth_tau", data=sigma_full_orth)
            f.create_dataset("Sigma_full_ao_tau", data=sigma_full_ao)
            f.create_dataset("Sigma1_full_orth", data=sigma_inf_full_orth)
            f.create_dataset("Sigma1_full_ao", data=sigma_inf_full_ao)

    insert_sigma_into_seet_file(
        results_file=args.results_file,
        iteration=args.iteration,
        sigma_add_ao=sigma_full_ao,
        sigma_inf_add_ao=sigma_inf_full_ao,
        mixing=args.mixing,
    )


if __name__ == "__main__":
    main()