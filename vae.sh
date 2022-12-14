#!/bin/bash
#SBATCH -J sweep_small
#SBATCH -p medium
#SBATCH -N 1
#SBATCH --gres=gpu:tesla:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16g
#SBATCH -o /homedtic/hperez/GrooveTransformer/error/%N.%J.VAE_test_loader.out
#SBATCH -e /homedtic/hperez/GrooveTransformer/error/%N.%J.VAE_test_loader.err

export PATH="$HOME/GrooveTransformer/VarGrvTrnsfmr/bin:$PATH"

cd GrooveTransformer
source VarGrvTrnsfmr/bin/activate
wandb agent mmil_vae_g2d/SmallSweeps_MGT_VAE/bib6bpsb
#python sweep_tester_VAE.py