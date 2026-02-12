import argparse
import time
import re
from pathlib import Path
import numpy as np
import h5py
import data_analyzing_from_inchworm.processing_after_inchworm as proc



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
    # 1) Reuse the processing script's argument parser
    ap = proc.build_argparser()

    # 2) Add SEET-specific arguments
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

    # 3) Run inchworm post-processing using the other script
    sigma_imp = proc.run_processing(args)   # (nomega, ns, nao_imp, nao_imp)

    # 4) Load transformation matrices
    with h5py.File(args.transform_file, "r") as ft:
        X_k = ft["X_k"][()]
        uu = ft[f"{args.impurity_index}/UU"][()] + 0j

    # 5) Build full-space sigma
    sigma_full_orth, sigma_full_ao = build_full_space_sigma_from_impurity(
        sigma_imp=sigma_imp,
        uu=uu,
        X_k=X_k
    )

    # 6) Optional: save “whole space” sigma for debugging
    if args.save_full_sigma is not None:
        with h5py.File(args.save_full_sigma, "w") as f:
            f.create_dataset("Sigma_imp_iw", data=sigma_imp)
            f.create_dataset("Sigma_full_orth_iw", data=sigma_full_orth)
            f.create_dataset("Sigma_full_ao_iw", data=sigma_full_ao)

    # 7) Insert into SEET file using mixing already in args
    insert_sigma_into_seet_file(
        results_file=args.results_file,
        iteration=args.iteration,
        sigma_add_ao=sigma_full_ao,
        mixing=args.mixing
    )
