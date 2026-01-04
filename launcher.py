#!/usr/bin/env python3
import argparse
import subprocess
import time
from pathlib import Path
from typing import Iterable, Tuple

REQUIRED_FILES = ("hopping.txt", "delta.txt", "Uijkl.txt")

SLURM_TEMPLATE = """#!/bin/bash -l
#SBATCH --job-name=eqiw
#SBATCH -p gcohen_intel
#SBATCH -o outfile
#SBATCH -e errfile
#SBATCH -n 1300
#SBATCH --exclude=compute-0-16
#SBATCH --mem={memory}
#SBATCH --time={walltime}
#SBATCH --chdir={run_dir}

####  mem="{memory}"
####  walltime={walltime}

####module load mpi/openmpi-1.10.4 gcc/gcc-7.3.0 python/anaconda_python-3.6.1
#module load anaconda2   ### be careful anaconda will override modules....
#module load openmpi/1.10.7 hdf5/1.10.1 fftw/3.3.4

#module load hdf5-1.10.6-gcc-9.1.0-cybjw4a fftw-3.3.8-gcc-9.1.0-jsnca6v
#export CC=$(which gcc)
#export CXX=$(which g++)
#export HDF5_USE_FILE_LOCKING=FALSE
#EquilibriumInchworm_dir=$HOME/src2/multiorbital/EquilibriumInchworm/
#EquilibriumInchworm_dir=$HOME/src/multiorbital/EquilibriumInchworm/
EquilibriumInchworm_dir=/gcohenlabstorage/eeitan/eq_inchworm_from_dolev_2025/EquilibriumInchworm/
EquilibriumInchworm_bin=$EquilibriumInchworm_dir/build/inchworm

date
#echo 'SLURM_JOBID=' $SLURM_JOBID
hostname
pwd

#module list

#. /a/home/cc/chemist/eeitan/eeitan/venv/bin/activate
ml purge
ml git/2.30.2   gnu8/8.3.0   openmpi4/4.1.0  hdf5/1.10.5   fftw/3.3.9  boost/1.71.0   eigen/3.3.9  ALPSCore/2.3.2-master12.06.22  cmake/3.23.2  openblas/0.3.7
ml
rm -f results.h5
echo 'Evaluating hybridization...' > execution_log.txt

echo 'Starting QMC...' >> execution_log.txt
mpirun --mca btl ofi --oversubscribe -n $SLURM_NTASKS $EquilibriumInchworm_bin run.param_G_sparseTau --run.type=bare_prop --mc.max_order=1000 >> execution_log.txt
mpirun --mca btl ofi --oversubscribe -n $SLURM_NTASKS $EquilibriumInchworm_bin run.param_G_sparseTau --run.type=inch_prop >> execution_log.txt
echo '$EquilibriumInchworm_dir/plot_prop.py'
python $EquilibriumInchworm_dir/plot_prop.py &

mpirun --mca btl ofi --oversubscribe -n $SLURM_NTASKS $EquilibriumInchworm_bin run.param_G_sparseTau --run.type=inch_gf --mc.max_runtime=20 >> execution_log.txt

echo '$EquilibriumInchworm_dir/plot_inch_gf.py'
python $EquilibriumInchworm_dir/plot_inch_gf.py &
"""

def files_present(paths: Iterable[Path]) -> bool:
    return all(p.exists() for p in paths)

def snapshot(paths: Iterable[Path]) -> Tuple[Tuple[int, int], ...]:
    """
    Return (size, mtime_ns) per file. If missing, returns (-1, -1).
    """
    out = []
    for p in paths:
        try:
            st = p.stat()
            out.append((st.st_size, st.st_mtime_ns))
        except FileNotFoundError:
            out.append((-1, -1))
    return tuple(out)

def stable_files(paths: Iterable[Path], checks: int, interval_s: float, require_nonempty: bool) -> bool:
    """
    Consider files 'stable' if their (size,mtime) is unchanged across `checks` consecutive snapshots.
    """
    last = snapshot(paths)
    for _ in range(checks):
        time.sleep(interval_s)
        cur = snapshot(paths)
        if cur != last:
            return False
        last = cur

    if require_nonempty:
        for p in paths:
            try:
                if p.stat().st_size <= 0:
                    return False
            except FileNotFoundError:
                return False
    return True

def write_slurm(run_dir: Path, slurm_name: str, memory: str, walltime: str) -> Path:
    slurm_path = run_dir / slurm_name
    content = SLURM_TEMPLATE.format(memory=memory, walltime=walltime, run_dir=str(run_dir))
    slurm_path.write_text(content)
    # Make it executable (not required for sbatch, but convenient)
    slurm_path.chmod(0o750)
    return slurm_path

def sbatch(slurm_path: Path, cwd: Path) -> int:
    # --parsable returns "12345" or "12345;cluster"
    res = subprocess.run(
        ["sbatch", "--parsable", str(slurm_path)],
        cwd=str(cwd),
        check=True,
        capture_output=True,
        text=True,
    )
    job_id = res.stdout.strip().split(";")[0]
    return int(job_id)

def main():
    ap = argparse.ArgumentParser(description="Wait for input files, generate slurm script, and sbatch it.")
    ap.add_argument("--dir", default=".", help="Folder to watch / run in (default: current dir).")
    ap.add_argument("--memory", required=True, help='Slurm memory, e.g. "64G" or "64000M".')
    ap.add_argument("--walltime", required=True, help='Slurm time, e.g. "02:00:00" or "2-00:00:00".')
    ap.add_argument("--slurm-name", default="eqiw.sbatch", help="Output slurm filename.")
    ap.add_argument("--poll", type=float, default=5.0, help="Polling interval seconds.")
    ap.add_argument("--stable-checks", type=int, default=2,
                    help="How many consecutive stable snapshots to require.")
    ap.add_argument("--stable-interval", type=float, default=2.0,
                    help="Seconds between stability snapshots.")
    ap.add_argument("--require-nonempty", action="store_true",
                    help="Require input files to be non-empty before submitting.")
    ap.add_argument("--marker", default=".eqiw_submitted",
                    help="Marker file to prevent re-submitting.")
    args = ap.parse_args()

    run_dir = Path(args.dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    marker = run_dir / args.marker
    if marker.exists():
        print(f"[skip] Marker exists: {marker} (already submitted). Delete it to allow re-submit.")
        return

    required_paths = [run_dir / name for name in REQUIRED_FILES]
    print(f"[watch] Watching {run_dir}")
    print(f"[watch] Required: {', '.join(REQUIRED_FILES)}")

    while True:
        if files_present(required_paths):
            if stable_files(required_paths, checks=args.stable_checks,
                            interval_s=args.stable_interval, require_nonempty=args.require_nonempty):
                break
            else:
                print("[watch] Files exist but still changing; waiting...")
        time.sleep(args.poll)

    slurm_path = write_slurm(run_dir, args.slurm_name, args.memory, args.walltime)
    print(f"[write] Wrote slurm file: {slurm_path}")

    job_id = sbatch(slurm_path, cwd=run_dir)
    marker.write_text(str(job_id) + "\n")
    print(f"[submit] Submitted job {job_id}")
    print(f"[done] Marker written: {marker}")

if __name__ == "__main__":
    main()
