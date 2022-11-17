#!/bin/bash
#SBATCH -J sweep_small
#SBATCH -p medium
#SBATCH -N 1
#SBATCH --gres=gpu:tesla:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16g
#SBATCH --time=8:00:00
#SBATCH -o %N.%J.VAE_test_loader.out
#SBATCH -e %N.%J.VAE_test_loader.err

source /etc/profile.d/lmod.sh
source /etc/profile.d/zz_hpcnow-arch.sh
module load Anaconda3/2020.02
conda activate GrooveTransformer
wandb agent mmil_vae_g2d/SmallSweeps_MGT_VAE/bib6bpsb
#python sweep_tester_VAE.py