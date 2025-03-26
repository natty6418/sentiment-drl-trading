#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=3:00:00
#SBATCH --mem=30GB                # Allocate 30GB memory
#SBATCH --gres=gpu:rtx8000:1       # Request 1 RTX8000 GPU
#SBATCH --job-name=torch

module purge

singularity exec --nv \
    --overlay /scratch/nm3833/pytorch-example/my_pytorch.ext3:ro \
    /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
    /bin/bash -c "source /ext3/env.sh;
                  source activate py310_env;
                  python fineTunning.py"
