#!/bin/bash
#SBATCH --job-name=v4_m06_A0350_eps1p2
#SBATCH --output=cp2k_run_%j.out
#SBATCH --error=cp2k_run_%j.err
#SBATCH --time=03:00:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=8
#SBATCH --account=e89-camm
#SBATCH --partition=standard
#SBATCH --qos=standard

module load epcc-job-env
module load cp2k/cp2k-2024.3
export OMP_NUM_THREADS=8
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
export OMP_PLACES=cores

srun --distribution=block:block --hint=nomultithread cp2k.psmp -i cp2k.inp -o cp2k.out
