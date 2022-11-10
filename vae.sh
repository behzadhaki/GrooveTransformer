#!/bin/bash
#SBATCH -J sweep_small
#SBATCH -p long
#SBATCH -N 1
#SBATCH --gres=gpu:tesla:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16g
#SBATCH -o /homedtic/hperez/error/%N.%J.VAE_test_loader.out
#SBATCH -e /homedtic/hperez/error/%N.%J.VAE_test_loader.err

export PATH="$HOME/GrooveTransformer/VarGrvTrnsfmr/bin:$PATH"

cd GrooveTransformer
source VarGrvTrnsfmr/bin/activate
wandb agent mmil_vae_g2d/sweeps_small/oacvriid
#python sweep_tester_VAE.py