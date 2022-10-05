# VariationalMonotonicGrooveTransformer
Variational version of Monotonic Groove Transformer


# Install Environment

----

1. Setup venv environment called `VarGrvTrnsfmr`


    python3 -m venv VarGrvTrnsfmr
    source VarGrvTrnsfmr/bin/activate
    pip3 install --upgrade pip

2. install `pytorch` (use cuda if on cluster)


   - Locally, mac/win no cuda 

    pip3 install torch torchvision torchaudio


   - Cuda on linux
  
    pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116


3. install the rest of the libraries


    pip3 install wandb
    pip3 install note_seq
    pip3 install bokeh
    pip3 install matplotlib
    pip3 install ffmpeg
    pip3 install tqdm
    pip3 install colorcet


4. Install FluidSynth and pufluidsynth 
   1. go to https://github.com/FluidSynth/fluidsynth/wiki/Download#distributions
   2. install the correct version of FluidSynth  (on mac I used `brew install fluidsynth`)
   3. install pyfluidsynth
   
             pip3 install pyfluidsynth



