import argparse
import os
import re
import subprocess
import time
from pathlib import Path


SBATCH_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=inchworm
#SBATCH --error=error.txt
#SBATCH --output=output.txt
#SBATCH --partition=gcohen_2023
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
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





def main():
    ap = argparse.ArgumentParser(description="Generate run.param + sbatch, submit, wait, then plot.")
    ap.add_argument("--run-dir", default=".", help="Directory to write files and run the job.")
    ap.add_argument("--beta", type=float, default=100.0, help="Value for beta in run.param.")
    ap.add_argument("--orbitals", type=int, default=4, help="Value for [hamiltonian].orbitals in run.param.")
    ap.add_argument("--hyb-ntau", type=int, default=1001, help="Value for [hybridization].ntau in run.param.")
    ap.add_argument("--submit", action="store_true", help="Actually call sbatch (otherwise only generate files).")
    ap.add_argument("--poll", type=float, default=10.0, help="Polling seconds while waiting for job.")
    ap.add_argument("--plot-after", action="store_true", help="Run plot_inch_gf.py after job completes.")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    # 1) Write run.param
    runparam_path = run_dir / "run.param"
    write_text(
        runparam_path,
        RUNPARAM_TEMPLATE.format(beta=args.beta, orbitals=args.orbitals, hyb_ntau=args.hyb_ntau),
    )
    print(f"[write] {runparam_path}")

    # 2) Write sbatch script
    sbatch_path = run_dir / "inchworm.sbatch"
    write_text(
        sbatch_path,
        SBATCH_TEMPLATE.format(run_dir=str(run_dir)),
        make_executable=False,  # not required for sbatch
    )
    print(f"[write] {sbatch_path}")

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