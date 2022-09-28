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


3. install `wandb`


    pip3 install wandb
