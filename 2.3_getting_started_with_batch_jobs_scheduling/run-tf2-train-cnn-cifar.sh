#!/usr/bin/env bash

#SBATCH --job-name=train-cnn-cifar-c10-fp32-e42-bs256-tensorflow-22.08-tf2-py3-1v100
#SBATCH --account=use300
#SBATCH --partition=gpu-debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10
#SBATCH --cpus-per-task=1
#SBATCH --mem=92G
#SBATCH --gpus=1
#SBATCH --time=00:05:00
#SBATCH --output=%x.o%j.%N

declare -xr LOCAL_SCRATCH_DIR="/scratch/${USER}/job_${SLURM_JOB_ID}"
declare -xr LUSTRE_PROJECTS_DIR="/expanse/lustre/projects/${SLURM_JOB_ACCOUNT}/${USER}"
declare -xr LUSTRE_SCRATCH_DIR="/expanse/lustre/scratch/${USER}/temp_project"
declare -xr SINGULARITY_CONTAINER_DIR='/cm/shared/apps/containers/singularity'

declare -xr SINGULARITY_MODULE='singularitypro/3.9'

module purge
module load "${SINGULARITY_MODULE}"
module list
export KERAS_HOME="${LOCAL_SCRATCH_DIR}"
printenv

time -p singularity exec --bind "${KERAS_HOME}:/tmp" --nv "${SINGULARITY_CONTAINER_DIR}/tensorflow/tensorflow_22.08-tf2-py3.sif" \
  python3 -u tf2-train-cnn-cifar.py --classes 10 --precision fp32 --epochs 42 --batch_size 256
