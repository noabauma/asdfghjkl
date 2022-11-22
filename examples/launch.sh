#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --output=slurm_output.txt
#SBATCH --error=slurm_error.txt
#SBATCH --constraint=gpu
#SBATCH --partition=debug
#SBATCH --time=00:15:00
#SBATCH --account=g34

export MASTER_ADDR=$(hostname)
export MASTER_PORT=1234

srun python train_resnet50_with_kfac.py 