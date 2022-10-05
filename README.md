# VariationalMonotonicGrooveTransformer
Variational version of Monotonic Groove Transformer


# Install Environment (Local on Mac/Linux/Windows)

----

1. Setup venv environment called `VarGrvTrnsfmr`

   ```
    python3 -m venv VarGrvTrnsfmr
    source VarGrvTrnsfmr/bin/activate
    pip3 install --upgrade pip
   ```
   
2. install `pytorch` (use cuda if on cluster)


   - Locally, mac/win no cuda 

   ```
    pip3 install torch torchvision torchaudio
   ```

   - Cuda on linux
  
   ```
    pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
   ```

3. install the rest of the libraries

   ```
    pip3 install wandb
    pip3 install note_seq
    pip3 install bokeh
    pip3 install matplotlib
    pip3 install ffmpeg
    pip3 install tqdm
    pip3 install colorcet
   ```

4. Install FluidSynth and pufluidsynth 
   1. go to https://github.com/FluidSynth/fluidsynth/wiki/Download#distributions
   2. install the correct version of FluidSynth  (on mac I used `brew install fluidsynth`)
   3. install pyfluidsynth
   
   ```
             pip3 install pyfluidsynth
   ```


# Install Environment (On UPF Cluster Use Conda)

----

Before installing the environment above, do the following:

1. Start an interactive session

      ```
      srun --nodes=1 --partition=short --gres=gpu:1 --cpus-per-task=4 --mem=8g --pty bash -i
      source /etc/profile.d/lmod.sh
      source /etc/profile.d/zz_hpcnow-arch.sh
      ```

2. We need to load FluidSynth first. This is needed for running `import fluidsynth` after 
installing the `pyFluidSynth` pip3 package. (check available modules using `module av` command)
   
      ```
      module load FluidSynth/2.3.0-GCCcore-10.2.0
      ```

3. Setup venv environment called `VarGrvTrnsfmr`
   
      ```
      conda create --name VarGrvTrnsfmr python=3.6   
      source activate VarGrvTrnsfmr		           
      ```

4. Install following packages
      
      ```
      conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
      pip3 install note_seq
      pip3 install wandb
      pip3 install bokeh
      pip3 install matplotlib
      pip3 install ffmpeg
      pip3 install tqdm
      pip3 install colorcet
      pip3 install visual_midi	
      pip3 install pyfluidsynth 
      ```