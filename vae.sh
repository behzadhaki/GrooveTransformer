#!/bin/bash
#SBATCH -J sweep_small
#SBATCH -p high
#SBATCH -N 1
#SBATCH --gres=gpu:quadro:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=16g
#SBATCH --time=24:00:00
#SBATCH -o %N.%J.VAE_test_loader.out
#SBATCH -e %N.%J.VAE_test_loader.err

# Necessary to access existing modules on the cluster

export PATH="$HOME/miniconda_envs/anaconda3/bin:$PATH"
export PATH="$HOME/miniconda_envs/anaconda3/envs/GrooveTransformer:$PATH"
source activate GrooveTransformer

# Login to WANDB
cd GrooveTransformer

export WANDB_API_KEY="API_KEY"
python -m wandb login

wandb agent mmil_vae_g2d/SmallSweeps_MGT_VAE/b339vcez
# python train.py