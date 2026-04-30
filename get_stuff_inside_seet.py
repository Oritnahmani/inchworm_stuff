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


def rotate_static_orth_to_ao_k(*, sigma_inf_full_orth: np.ndarray, X_k: np.ndarray):
    """
    Rotate static Sigma1 from orthogonal basis to AO basis per k-point.

    sigma_inf_full_orth: (ns, nao_full, nao_full)
    X_k:                 (nk, nao_full, nao_full)

    returns:
    sigma_inf_full_ao:   (ns, nk, nao_full, nao_full)
    """
    ns, nao_full, _ = sigma_inf_full_orth.shape
    nk, nao_full2, _ = X_k.shape
    if nao_full != nao_full2:
        raise ValueError(f"nao_full mismatch: sigma_inf has {nao_full}, X_k has {nao_full2}")

    X_k_H = X_k.conj().transpose(0, 2, 1)
    return np.einsum(
        "kab, sbc, kcd -> skad",
        X_k,
        sigma_inf_full_orth,
        X_k_H,
        optimize=True,
    )


def insert_sigma_into_seet_file(
    *,
    results_file: Path,
    iteration: int,
    sigma_add_ao: np.ndarray,       # (ntau, ns, nk, nao_full, nao_full)
    sigma_inf_add_ao: np.ndarray,   # (ns, nk, nao_full, nao_full)
    mixing: float,
):
    """
    Update iter{iteration}/Selfenergy/data and iter{iteration}/Sigma1.
    If iter{iteration} does not exist, initialize it by copying iter{iteration-1}.
    """
    with h5py.File(results_file, "r+") as fs:
        new_iter_key = f"iter{iteration}"
        prev_iter_key = f"iter{iteration - 1}"

        if new_iter_key not in fs:
            if prev_iter_key not in fs:
                raise KeyError(
                    f"{new_iter_key} not found, and cannot initialize it because {prev_iter_key} is also missing."
                )

            prev_group = fs[prev_iter_key]
            new_group = fs.create_group(new_iter_key)

            # copy everything from previous iteration into the new one
            for name in prev_group.keys():
                prev_group.copy(name, new_group, name=name)

            # update the top-level current-iteration marker if present
            if "iter" in fs:
                fs["iter"][...] = iteration

        sigma_group = fs[f"{new_iter_key}/Selfenergy"]
        sigma_in = sigma_group["data"][()]
        sigma_inf_in = fs[f"{new_iter_key}/Sigma1"][()]

        if sigma_in.shape != sigma_add_ao.shape:
            raise ValueError(
                f"Dynamic sigma shape mismatch: SEET {sigma_in.shape} vs add {sigma_add_ao.shape}"
            )
        if sigma_inf_in.shape != sigma_inf_add_ao.shape:
            raise ValueError(
                f"Static sigma shape mismatch: SEET {sigma_inf_in.shape} vs add {sigma_inf_add_ao.shape}"
            )

        sigma_group["data"][...] = sigma_in + mixing * sigma_add_ao
        fs[f"{new_iter_key}/Sigma1"][...] = sigma_inf_in + mixing * sigma_inf_add_ao


def main():
    ap = proc.build_argparser()
    ap.add_argument("--transform-file", type=Path, required=True)
    ap.add_argument("--results-file", type=Path, required=True)
    ap.add_argument("--iteration", type=int, required=True)
    ap.add_argument("--mixing", type=float, default=0.5)
    ap.add_argument("--save-full-sigma", type=Path, default=None)
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