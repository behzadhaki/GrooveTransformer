#!/bin/bash
#SBATCH -J env_setup
#SBATCH -p medium
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8g
#SBATCH -o %N.%J.env_setup_conda.out
#SBATCH -e %N.%J.env_setup_conda.err

source /etc/profile.d/lmod.sh
source /etc/profile.d/zz_hpcnow-arch.sh

module load Anaconda3/2020.02

# Create a new conda environment with Python 3.9 if it doesn't exist
# UNCOMMENT THE FOLLOWING LINE IF YOU WANT TO CREATE A THE ENVIRONMENT for THE FIRST TIME
# conda create --name GrooveTransformer python=3.9

conda activate GrooveTransformer

conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=10.2 -c pytorch
pip install bokeh==2.4.3
pip install colorcet==3.0.0
pip install fluidsynth==0.2
pip install holoviews==1.15.1
pip install librosa==0.9.2
pip install matplotlib==3.6.0
pip install note_seq==0.0.5
pip install numpy==1.23.3
pip install pandas==1.5.0
pip install pretty_midi==0.2.9
pip install pyFluidSynth==1.3.1
pip install PyYAML==6.0
pip install scikit_learn==1.1.3
pip install scipy==1.9.1
pip install soundfile==0.11.0
pip install tqdm==4.64.1
pip install wandb==0.13.3