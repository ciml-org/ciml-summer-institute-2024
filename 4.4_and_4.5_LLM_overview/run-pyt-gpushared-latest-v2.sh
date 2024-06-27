#!/usr/bin/env bash
#SBATCH --job-name=hface-gpu
#SBATCH --account=gue998
# -----------------------------
#SBATCH --partition=gpu-shared
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=92G
#SBATCH --output=slurmoutgpudebug.hface.%x.o%j.%N.txt
# ----------------------
#SBATCH --time=00:30:00

# if you are using the ciml24 reservation, use these
#SBATCH --reservation=ciml24gpu
#SBATCH --qos=gpu-shared-eot

module purge
module load gpu
module load slurm

module load singularitypro/3.11  #container
module list


# Set up paths for python to find pacakges
# Use these lines if you move ~/.local (which is default location) to Local_HFgpu-latest
#      or if you installed packages directly to Local_HFgpu-latest 
#export PYTHONPATH=/home/$USER/HFace/Local_HFgpu-latest/lib/python3.10/site-packages/:$PYTHONPATH
#export PATH=/home/$USER/HFace/Local_HFgpu_latest/local/bin:$PATH
#echo "--------------- paths -------------------"
#echo $PYTHONPATH
#echo $PATH

# Set up cache location ----------------------------
#      use this if you are not using the default ~/.cache folder
#      export HF_HOME=/home/$USER/cache
#  ---- OR --------
#You can run hugging face login first (it will put the token in .cache folder)
# and also run it within singularity (b/c it was installed that way)

#singularity exec --nv --bind /expanse,/scratch /cm/shared/apps/containers/singularity/pytorch/pytorch-latest.sif /home/$USER/HFace/Local_HFgpu-latest/bin/huggingface-cli login --token  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 
# ----------------------------------------------------

#Now you can run the rag example
singularity exec --nv --bind /expanse,/scratch /cm/shared/apps/containers/singularity/pytorch/pytorch-latest.sif python3 Rag_example_forbatch_v2.py > stdout_answer.txt

