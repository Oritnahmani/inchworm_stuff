import argparse
import subprocess
from pathlib import Path

REQUIRED_FILES = ("hopping.txt", "delta.txt", "Uijkl.txt")

SEET_TEMPLATE = """#!/bin/bash -l
#SBATCH --job-name=seet
#SBATCH --error=error_%j.txt
#SBATCH --output=output_%j.txt
#SBATCH --partition=gcohen_2023
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
##SBATCH --cpus-per-task=128
#SBATCH --time=336:00:00
#SBATCH --hint=nomultithread
#SBATCH --exclusive
#SBATCH --chdir={run_dir}

export OMP_NUM_THREADS=1
source /gcohenlabstorage/oritnahmani/software/spack_with_Gaurav/spack_init.sh
ml git
spack env activate green
BETA={beta}
export HDF5_USE_FILE_LOCKING=FALSE

mpirun -n 128  $GREENSEET_ROOT/bin/embedding.exe --scf_type=GW --BETA $BETA --grid_file $GREENSEET_ROOT/share/ir/1e5.h5 --itermax 1 --results_file sim_seet.h5 --weak_results ../mbpt_with_mixining_6/NiO_GW.h5 --embedding_type SEET --mixing_type CDIIS --diis_start 2 --diis_size 5 --mixing_weight 0.3 --jobs SC --seet_input ../init_seet/transform.h5 --bath_file bath.txt --impurity_solver_exec  $GREENSEET_ROOT/seet_solvers/bin/ed_solver --impurity_solver_params " --arpack.NEV=8 --arpack.NCV=20 --lanc.NOMEGA=1000 --FREQ_FILE=$GREENSEET_ROOT/share/ir/1e5.h5 --FREQ_PATH=/fermi/ngrid " --dc_data_prefix "../init_seet/dc_int"  --dc_data_path_prefix "../init_seet/dc_int"  --seet_root_dir "./seet"  --spin_symm true --impurity_solver inchworm
"""

EQIW_TEMPLATE = """#!/bin/bash -l
#SBATCH --job-name=eqiw
#SBATCH -p gcohen_intel
#SBATCH -o outfile
#SBATCH -e errfile
#SBATCH -n 1300
#SBATCH --exclude=compute-0-16
#SBATCH --time={walltime}
#SBATCH --chdir={run_dir}

EquilibriumInchworm_dir=/gcohenlabstorage/eeitan/eq_inchworm_from_dolev_2025/EquilibriumInchworm/
EquilibriumInchworm_bin=$EquilibriumInchworm_dir/build/inchworm

date
hostname
pwd

ml purge
ml git/2.30.2   gnu8/8.3.0   openmpi4/4.1.0  hdf5/1.10.5   fftw/3.3.9  boost/1.71.0   eigen/3.3.9  ALPSCore/2.3.2-master12.06.22  cmake/3.23.2  openblas/0.3.7
ml

# Safety: wait until SEET has produced the required files (and they're non-empty)
for f in hopping.txt delta.txt Uijkl.txt; do
  while [ ! -s "$f" ]; do
    echo "Waiting for $f ..."
    sleep 10
  done
done

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

def write_text(path: Path, text: str) -> None:
    path.write_text(text)
    path.chmod(0o750)

def sbatch(script: Path, cwd: Path, extra_sbatch_args=None) -> int:
    extra_sbatch_args = extra_sbatch_args or []
    res = subprocess.run(
        ["sbatch", "--parsable", *extra_sbatch_args, str(script)],
        cwd=str(cwd),
        check=True,
        capture_output=True,
        text=True,
    )
    job_id = res.stdout.strip().split(";")[0]
    return int(job_id)

def main():
    ap = argparse.ArgumentParser(
        description="Submit SEET job, then submit EQIW job after SEET completes successfully."
    )
    ap.add_argument("--run-dir", default=".", help="Directory where both jobs will run / where files are created.")
    ap.add_argument("--seet-script", default="seet.sbatch", help="Filename for generated SEET sbatch script.")
    ap.add_argument("--eqiw-script", default="eqiw.sbatch", help="Filename for generated EQIW sbatch script.")
    ap.add_argument("--beta", default="100", help="Value for BETA in the SEET job.")
    ap.add_argument("--memory", default=None, help='EQIW memory (optional). If omitted, do not set #SBATCH --mem and use cluster defaults.')
    ap.add_argument("--walltime", required=True, help='EQIW time, e.g. "02:00:00" or "2-00:00:00".')
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    # Write SEET script
    seet_path = run_dir / args.seet_script
    write_text(seet_path, SEET_TEMPLATE.format(run_dir=str(run_dir), beta=args.beta))
    print(f"[write] {seet_path}")

    # Write EQIW script
    eqiw_path = run_dir / args.eqiw_script
    write_text(eqiw_path, EQIW_TEMPLATE.format(run_dir=str(run_dir), memory=args.memory, walltime=args.walltime))
    print(f"[write] {eqiw_path}")

    # Submit SEET
    seet_job = sbatch(seet_path, cwd=run_dir)
    print(f"[submit] SEET job id: {seet_job}")

    # Submit EQIW with dependency on SEET success
    eqiw_job = sbatch(eqiw_path, cwd=run_dir, extra_sbatch_args=[f"--dependency=afterok:{seet_job}"])
    print(f"[submit] EQIW job id: {eqiw_job} (depends on afterok:{seet_job})")

    print("[done] Pipeline submitted.")
    print(f"       SEET must generate: {', '.join(REQUIRED_FILES)} in {run_dir}")

if __name__ == "__main__":
    main()


