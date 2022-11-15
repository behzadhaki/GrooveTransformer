#!/bin/bash
#SBATCH -J sweep_small
#SBATCH -p medium
#SBATCH -N 1
#SBATCH --gres=gpu:tesla:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16g
#SBATCH -o /homedtic/bhaki/GrooveTransformer/error/%N.%J.VAE_test_loader.out
#SBATCH -e /homedtic/bhaki/GrooveTransformer/error/%N.%J.VAE_test_loader.err

export PATH="$HOME/project/anaconda3/bin:$PATH"
export PATH="$/homedtic/bhaki/project/anaconda3/envs/torch_thesis:$PATH"
export PATH="$/homedtic/bhaki:$PATH"

cd GrooveTransformer

source activate torch_thesis
wandb agent mmil_vae_g2d/SmallSweeps_MGT_VAE/7295cxer
#python sweep_tester_VAE.py