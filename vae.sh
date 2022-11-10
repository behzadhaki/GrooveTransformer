#!/bin/bash
#SBATCH -J bhaki_test_cuda
#SBATCH -p short
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8g
#SBATCH -o /homedtic/bhaki/error/%N.%J.VAE_test_loader.out
#SBATCH -e /homedtic/bhaki/error/%N.%J.VAE_test_loader.err
export PATH="$HOME/project/anaconda3/bin:$PATH"
export PATH="$/homedtic/bhaki/project/anaconda3/envs/torch_thesis:$PATH"
source activate torch_thesis
cd GrooveTransformer
python sweep_tester_VAE.py