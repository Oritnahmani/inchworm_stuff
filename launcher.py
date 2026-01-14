import argparse
import os
import re
import subprocess
import time
from pathlib import Path
import h5py
import json


SBATCH_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=inchworm
#SBATCH --error=error.txt
#SBATCH --output=output.txt
#SBATCH --partition={partition}
#SBATCH --nodes={nodes}
#SBATCH --ntasks={ntasks}
##SBATCH --cpus-per-task=128
#SBATCH --time=336:00:00
#SBATCH --hint=nomultithread
#SBATCH --exclusive
#SBATCH --chdir={run_dir}

export OMP_NUM_THREADS=1

EquilibriumInchworm_dir=/gcohenlab/data/dolevg/src/eq_inchworm/EquilibriumInchworm
EquilibriumInchworm_bin=$EquilibriumInchworm_dir/build/inchworm

date
hostname

ml purge
ml gnu8 openmpi4 hdf5

echo "Running on:" >> execution_log.txt
mpirun --mca btl ofi hostname >> execution_log.txt
mpirun --mca btl ofi $EquilibriumInchworm_bin run.param --run.type=bare_prop --mc.max_order=1000 >> execution_log.txt
mpirun --mca btl ofi $EquilibriumInchworm_bin run.param --run.type=inch_prop >> execution_log.txt
mpirun --mca btl ofi $EquilibriumInchworm_bin run.param --run.type=inch_gf >> execution_log.txt
"""

RUNPARAM_TEMPLATE = """beta = {beta}
work_in_local_eigenbasis = true

[hamiltonian]
orbitals = {orbitals}
type = file
file.hopping = "hopping.txt"
file.Uijkl = "Uijkl.txt"
energy_shift = 0

[hybridization]
type = file
ntau = {hyb_ntau}
nblock = 2
norbital = 2
optimization_type = 3

[mc]
num_steps = 100000000
num_equilibration_steps = 1
num_decorrelation_steps = 1
max_runtime = 30
seed = 568460
max_order = 50
max_order_block = 8

[inchworm]
use_bare_prop = true
ntau_max_bare = 1

[bare_propagator]
ntau = 20000

[propagator]
ntau = 150

[gf]
type_of_discretization = "file"
file_of_discretization = "time_intervals.txt"
ntau = 143

[output]
"""

def _norm_state(s: str) -> str:
    """
    Normalize sinfo node states; clusters sometimes add suffixes like 'mix+', 'idle~', etc.
    """
    return s.strip().lower().strip("+~*")

def query_nodes(partitions: list[str] | None = None) -> list[dict]:
    """
    Return list of node dicts:
      {"node": str, "partitions": [str], "cpus": int, "state": str}
    """
    fmt = "%N|%P|%c|%t"
    cmd = ["sinfo", "-N", "-h", "-o", fmt]
    if partitions:
        cmd = ["sinfo", "-N", "-h", "-p", ",".join(partitions), "-o", fmt]

    res = subprocess.run(cmd, check=True, capture_output=True, text=True)
    nodes = []
    for line in res.stdout.splitlines():
        node, part, cpus, state = line.strip().split("|")
        parts = [p.replace("*", "") for p in part.split(",")]
        nodes.append(
            {"node": node, "partitions": parts, "cpus": int(cpus), "state": _norm_state(state)}
        )
    return nodes

def pick_best_nodes_auto(
    nodes: list[dict],
    prefer_states=("idle", "mix"),
    max_nodes: int | None = None,
    min_cpus: int | None = None,
    prefer: str = "max_total",   # "max_total" or "highest_cpus"
) -> dict:
    """
    Pick a *homogeneous* set of nodes (same CPU count) and decide how many nodes to use.

    Returns a dict:
      {
        "nodes": ["nodeA", "nodeB", ...],
        "cpus_per_node": 128,
        "num_nodes": 4,
        "total_tasks": 512
      }

    Heuristics:
    - Filter by state in prefer_states
    - Optionally filter by min_cpus
    - Group by cpus_per_node
    - Choose best CPU bucket either by:
        * max_total: maximize (cpus_per_node * available_node_count)
        * highest_cpus: choose highest cpus_per_node (even if fewer nodes)
    - Use up to max_nodes from that bucket
    """
    prefer_set = {s.lower() for s in prefer_states}

    candidates = [n for n in nodes if n["state"] in prefer_set]
    if not candidates:
        # Fall back: if nothing idle/mix, consider all nodes (job may queue)
        candidates = nodes

    if min_cpus is not None:
        candidates = [n for n in candidates if n["cpus"] >= min_cpus]

    if not candidates:
        raise RuntimeError("No candidate nodes after filtering (partition/state/min_cpus).")

    buckets = defaultdict(list)
    for n in candidates:
        buckets[n["cpus"]].append(n["node"])

    # Decide which bucket to use
    if prefer == "highest_cpus":
        chosen_cpus = max(buckets.keys())
    elif prefer == "max_total":
        # maximize cpus * count; tie-breaker: higher cpus
        chosen_cpus = max(buckets.keys(), key=lambda c: (c * len(buckets[c]), c))
    else:
        raise ValueError("prefer must be 'max_total' or 'highest_cpus'")

    node_list = sorted(buckets[chosen_cpus])  # stable ordering
    if max_nodes is not None:
        node_list = node_list[:max_nodes]

    num_nodes = len(node_list)
    total_tasks = chosen_cpus * num_nodes

    return {
        "nodes": node_list,
        "cpus_per_node": chosen_cpus,
        "num_nodes": num_nodes,
        "total_tasks": total_tasks,
    }



def write_text(path: Path, text: str, make_executable: bool = False) -> None:
    path.write_text(text)
    if make_executable:
        path.chmod(0o750)


def sbatch_submit(script_path: Path, cwd: Path) -> int:
    # --parsable prints jobid (sometimes "jobid;cluster")
    res = subprocess.run(
        ["sbatch", "--parsable", str(script_path)],
        cwd=str(cwd),
        check=True,
        capture_output=True,
        text=True,
    )
    return int(res.stdout.strip().split(";")[0])

def _latest_iter_group(h5: h5py.File) -> str:
    """
    Find the largest 'iterN' group present in the file and return its name, e.g. 'iter11'.
    """
    iters = []
    for k in h5.keys():
        if k.startswith("iter"):
            try:
                iters.append((int(k[4:]), k))  # (N, "iterN")
            except ValueError:
                pass
    if not iters:
        raise KeyError("No iter* groups found at H5 root (expected groups like iter0, iter1, ...).")
    return max(iters)[1]


def timetxt_and_beta(results_path: Path, run_dir: Path, normalize_to_unit: bool = True) -> float:
    """
    Reads tau mesh from the GW results H5 file, writes run_dir/time_intervals.txt,
    and returns beta (assumed tau[-1]).
    """
    results_path = Path(results_path)
    run_dir = Path(run_dir)

    with h5py.File(results_path, "r") as f:
        iter_group = _latest_iter_group(f)  # e.g. "iter11"
        mesh_path = f"/{iter_group}/Selfenergy/mesh"
        if mesh_path not in f:
            raise KeyError(f"Missing dataset {mesh_path} in {results_path}")

        tau = f[mesh_path][:]

    beta = float(tau[-1])

    x = (tau / beta) if normalize_to_unit else tau

    out = run_dir / "time_intervals.txt"
    out.write_text("\n".join(f"{v:.16e}" for v in x) + "\n")

    return beta




def main():
    ap = argparse.ArgumentParser(description="Generate run.param + sbatch, submit, wait, then plot.")
    ap.add_argument("--run-dir", default=".", help="Directory to write files and run the job.")
    ap.add_argument("--orbitals", type=int, default=4, help="Value for [hamiltonian].orbitals in run.param.")
    ap.add_argument("--hyb-ntau", type=int, default=1001, help="Value for [hybridization].ntau in run.param.")
    ap.add_argument("--submit", action="store_true", help="Actually call sbatch (otherwise only generate files).")
    ap.add_argument("--poll", type=float, default=10.0, help="Polling seconds while waiting for job.")
    ap.add_argument("--plot-after", action="store_true", help="Run plot_inch_gf.py after job completes.")
    ap.add_argument("--results-path", default=None, help="results file from previous GW calculation.")
    args = ap.parse_args()
    beta = timetxt_and_beta(args.results_path, args.run_dir)

    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)


    

    # 1) Write run.param


    runparam_path = run_dir / "run.param"
    write_text(
        runparam_path,
        RUNPARAM_TEMPLATE.format(beta=args.beta, orbitals=args.orbitals, hyb_ntau=args.hyb_ntau),
    )
    print(f"[write] {runparam_path}")

    nodes = query_nodes(partitions=["gcohen_2023"])
    best = pick_best_node(nodes)

    partition = best["partitions"][0]
    ntasks_per_node = best["cpus"]

    pin_node = True
    nodelist_line = f"#SBATCH --nodelist={best['node']}" if pin_node else ""


    # 2) Write sbatch script
    sbatch_path = run_dir / "inchworm.sbatch"
    write_text(
        sbatch_path,
    SBATCH_TEMPLATE.format(
        run_dir=str(run_dir),
        partition=partition,
        ntasks_per_node=ntasks_per_node,
        nodelist_line=nodelist_line,
    ),
)

    if not args.submit:
        print("[info] Not submitting (use --submit to sbatch).")
        return

    # 3) Submit
    job_id = sbatch_submit(sbatch_path, cwd=run_dir)
    print(f"[submit] job id: {job_id}")

    # 4) Wait for completion
    final_state = wait_for_job(job_id, poll_s=args.poll)
    print(f"[done] job {job_id} finished with state: {final_state}")

    if final_state != "COMPLETED":
        print("[warn] Job did not complete successfully; skipping plot.")
        return

    if not args.plot_after:
        print("[info] Not plotting (use --plot-after).")
        return

    # 5) Run plot script AFTER completion.
    # We know the sbatch defines EquilibriumInchworm_dir, but that variable exists only inside the job shell.
    # So we replicate the same directory here in Python:
    eq_dir = Path("/gcohenlab/data/dolevg/src/eq_inchworm/EquilibriumInchworm")
    plot_script = eq_dir / "plot_inch_gf.py"

    if not plot_script.exists():
        raise FileNotFoundError(f"Plot script not found: {plot_script}")

    print(f"[plot] Running: python {plot_script}")
    subprocess.run(["python", str(plot_script)], cwd=str(run_dir), check=True)
    print("[plot] Done.")


if __name__ == "__main__":
    main()