#!/bin/bash
#SBATCH -J sweep_small
#SBATCH -p medium
#SBATCH -N 1
#SBATCH --gres=gpu:tesla:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16g
#SBATCH --time=8:00:00
#SBATCH -o %N.%J.VAE_test_loader.out
#SBATCH -e %N.%J.VAE_test_loader.err

# Necessary to access existing modules on the cluster
source /etc/profile.d/lmod.sh
source /etc/profile.d/zz_hpcnow-arch.sh

# Load Anaconda Module, Activate Conda CLI and Activate Environment
module load Anaconda3/2020.02
export PATH="/soft/easybuild/x86_64/software/Anaconda3/2020.02/bin:$PATH"
export PATH="$HOME/.conda/envs/GrooveTransformer/bin:$PATH"
source /soft/easybuild/x86_64/software/Anaconda3/2020.02/etc/profile.d/conda.sh
conda activate GrooveTransformer

# Login to WANDB
export WANDB_API_KEY=API_KEY
python -m wandb login

# Run your codes here

cd GrooveTransformer
#wandb agent mmil_vae_g2d/SmallSweeps_MGT_VAE/bib6bpsb
python train.py