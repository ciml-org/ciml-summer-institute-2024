#!/usr/bin/env bash

#SBATCH --job-name=hvd-tf2-train-cnn-cifar-gpu
#SBATCH --account=gue998
#SBATCH --partition=gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=368G
#SBATCH --gpus=4
#SBATCH --time=00:10:00
#SBATCH --output=%x.o%j.%N

declare -xr LUSTRE_PROJECT_DIR="/expanse/lustre/projects/${SLURM_ACCOUNT}/${USER}"
declare -xr LUSTRE_SCRATCH_DIR="/expanse/lustre/scratch/mkandes/temp_project"
declare -xr LOCAL_SCRATCH_DIR="/scratch/${USER}/job_${SLURM_JOB_ID}"
declare -xr SINGULARITY_CONTAINER_DIR='/cm/shared/apps/containers/singularity'

declare -xr SINGULARITY_MODULE='singularitypro/3.11'
declare -xr COMPILER_MODULE='gcc/10.2.0'
declare -xr MPI_MODULE='openmpi/4.1.3'

module reset
module load "${SINGULARITY_MODULE}"
module load "${COMPILER_MODULE}"
module load "${MPI_MODULE}"
module list
export OMPI_MCA_btl='self,vader'
export UCX_TLS='shm,rc,ud,dc'
export UCX_NET_DEVICES='mlx5_0:1'
export UCX_MAX_RNDV_RAILS=1
printenv

time -p mpirun -n "${SLURM_NTASKS}" singularity exec --nv "${SINGULARITY_CONTAINER_DIR}/tensortflow-latest.sif" python3 -u hvd-tf2-train-cnn-cifar.py
