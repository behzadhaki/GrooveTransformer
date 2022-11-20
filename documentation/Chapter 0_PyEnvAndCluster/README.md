# Environment Setup 
# Table of Contents
1. [Required Packages](#1)
2. [CPU Installation](#2)
   1. [venv Installation](#2.1)
   2. [Anaconda Installation](#2.2)
3. [GPU Installation](#3)
   1. [Local GPU Installation](#3.1)
   2. [HPC Cluster GPU Installation](#3.2)
      1. [Anaconda Installation](#3.2.1)

> **Note** If you want to set up on the clusters, first refer to the **`CLUSTER GUIDE`** available [HERE](HPC_Cluster_Guide.md) 

# 1. Required Packages <a name="1"></a>
The repository was last tested with 
the following versions of the [packages](../../demos/installation/requirements_gpu.txt):

```commandline
pytorch==1.12.1 
torchvision==0.13.1 
torchaudio==0.12.1 
cudatoolkit=10.2    ----> GPU ONLY!
bokeh==2.4.3
colorcet==3.0.0
fluidsynth==0.2
holoviews==1.15.1
librosa==0.9.2
matplotlib==3.6.0
note_seq==0.0.5
numpy==1.23.3
pandas==1.5.0
pretty_midi==0.2.9
pyFluidSynth==1.3.1
PyYAML==6.0
scikit_learn==1.1.3
scipy==1.9.1
soundfile==0.11.0
tqdm==4.64.1
wandb==0.13.3
```

> **Note** In order to be able to use pyFluidSynth, you need to install fluidsynth software
> on your machine. pyFluidSynth needs fluidsynth to be installed on your machine in order to
> work. You can install fluidsynth using the following command:
> On Mac, you can do this by running 
> ```commandline
> brew install fluidsynth
> ```
> Alternatively, on Mac and Linux, you can install fluidsynth using the following command:
> ```commandline
> conda install -c conda-forge fluidsynth
> ```
> In this repo, the fluidsynth package is used to generate audio from a number of different symbolic
> representations such as `MIDI`, `Note_Sequence`, and `HVO_Sequence`. 

> **Warning** If you decide to install fluidsynth using the conda-forge channel, we recommend that you 
> prepare your conda environment first using the guides in section [2.2](#2.2) or [3.1](#2.3)

### 2. CPU Installation  <a name="2"></a>
#### 2.1  venv Installation <a name="2.1"></a>
First create a virtual environment:
```commandline
python3 -m venv GrooveTransformerVenv
```
Then activate the virtual environment and install the required packages:
```commandline
source GrooveTransformerVenv/bin/activate
```
To install torch, navigate to the [pytorch website](https://pytorch.org/get-started/locally/) 
and select the appropriate installation command for your system. See figure below for an example:

<img src="assets/torch_install.png" width="500" >

Then while the virtual environment is activated, run the following command to install the required packages:

```commandline
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
```

#### 2.2 [`Anaconda`](https://www.anaconda.com/) Installation <a name="2.2"></a>
To install the packages using conda, run the following command:

```commandline
conda create --name GrooveTransformer python=3.9
source activate GrooveTransformer

conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1  -c pytorch
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
```

### 3. GPU Installation <a name="3"></a>

#### 3.1 GPU Installation on Local Machines <a name="3.1"></a>
The gpu installation is similar to the cpu installation, except that you need to install the
`cudatoolkit` package.

If you are using `conda`, you can install the `cudatoolkit` package using the following command:

```commandline
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=10.2 -c pytorch
```

If you are using `pip`, you can install the `cudatoolkit` package using the pytorch suggested command 
found [here](https://pytorch.org/get-started/locally/).

#### 3.2 Installation on HPC Clusters <a name="3.2"></a>

The installations here are specific to the [`UPF DTIC HPC`](https://guiesbibtic.upf.edu/recerca/hpc) clusters. 
If you haven't used the clusters before, you can find a guide on how to use them [here](HPC_Cluster_Guide.md).





##### 3.2.1 [`Miniconda`](with local conda installation) <a name="3.2.1"></a>

###### Local Installation of Miniconda3 on the Cluster

> **Note** To install Miniconda3, first login to the cluster. 
> After logging into the login nodes (no need to connect to computational nodes yet), run the following commands
> ```terminal  
>  mkdir ~/miniconda_envs
>  cd ~/miniconda_envs
>  wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
>  bash Miniconda3-latest-Linux-x86_64.sh -b -p ~/miniconda_envs/anaconda3
>  ```

> **Note** Everytime you want to use the conda environment, you need to activate it first locate the path
> to the conda executable and activate the environment. For example
> ```terminal
>   export PATH="$HOME/miniconda_envs/anaconda3/bin:$PATH"
> ```
> This will allow your terminal to be able to find the conda executable.  

###### Creating a Miniconda Environment
Assuming that you have already installed Miniconda3, and located the path to the conda executable using the
`export PATH="$HOME/miniconda_envs/anaconda3/bin:$PATH"` command, you can create a conda environment using the following command:

```terminal
cd ~/miniconda_envs
conda create -y -n GrooveTransformer python=3.9 anaconda 
```

Once finished, check that the environment has been created by running the following command:

```terminal
conda info --envs
```

> **Warning** Open your `.bashrc` file and add make sure no conda commands are being executed when you open a new terminal.
> If you have any conda commands in your `.bashrc` file (should be at the very bottom), remove them.
> You can use `vim .bashrc`, then press `i` to enter insert mode, 
> then use the arrow keys to navigate to the bottom of the file. Once done, press `esc` to exit insert mode,
> then type `:wq` to save and quit.

###### Activating the Miniconda Environment 
In order to activate the environment, you need to locate the path to the conda environment as well as 
the path to the conda executable. 

```terminal
export PATH="$HOME/miniconda_envs/anaconda3/bin:$PATH"
export PATH="$HOME/miniconda_envs/anaconda3/envs/GrooveTransformer:$PATH"
source activate GrooveTransformer
```



###### Installing the Required Packages
Once the environment is activated, you can install the required packages using the following command:

```terminal
pip install torch==1.12.0+cu102 --extra-index-url https://download.pytorch.org/whl/cu102
pip3 install ffmpeg==1.4
pip3 install holoviews==1.15.1
pip3 install note-seq==0.0.5
pip3 install wandb==0.13.3

# conda install -c conda-forge fluidsynth                     # Only if you want to synthesize hvo_sequences                     
# pip3 install pyFluidSynth                                    # Only if you want to synthesize hvo_sequences
```


###### Batch Script for Miniconda Installation, Environment Creation, and Package Installation

The following batch scripts can be used to install Miniconda3, create a conda environment, 
and install the required packages. 

The installation can take a long time, so it is recommended to run 
the installation in 3 separate sessions, that is to say, run the [first script](#step1MinInstall) in one session,
then once it is finished, run the [second script](#step2condaCreate) in another session, and finally run the
[third script](#step3PipInstall) in a third session.

Alternatively, you can run the entire installation in one session using the [full batch script](#fullBatch).

> **Note** A batch script is a file that contains a series of commands that are executed one after the other.
> In order to run a batch script, you need to create an empty `[file_name].sh` file, and then copy the contents of the
> batch script into the `file.sh` file. Finally, you can sumbit the batch script to the cluster using the
> `sbatch file.sh` command.
> 
> To summarize,
> ```shell
> touch install_conda.sh
> 
> vim install_conda.sh
>    
>  # in vim: 
>     # 1.   press i to enter insert mode
>     # 2.   copy the contents of the batch script into the file
>     # 3.   press esc to exit insert mode
>     # 4.   type :wq to save and quit
> 
> sbatch install_conda.sh
> 
>```

- ##### Step1: Miniconda3 Installation <a name="step1MinInstall"></a>
```bash
#!/bin/bash
#SBATCH -J env_setup
#SBATCH -p high
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8g
#SBATCH --time=24:00:00
#SBATCH -o %N.%J.env_setup.out
#SBATCH -e %N.%J.env_setup.er

# Step 1. Intalling Miniconda
# ----------------------------------------------------
mkdir ~/miniconda_envs
cd ~/miniconda_envs
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p ~/miniconda_envs/anaconda3
```

- ##### Step2: Environment Creation <a name="step2condaCreate"></a>
```bash
#!/bin/bash
#SBATCH -J env_setup
#SBATCH -p high
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8g
#SBATCH --time=24:00:00
#SBATCH -o %N.%J.env_setup.out
#SBATCH -e %N.%J.env_setup.er

# Step 2. Create a new Conda Env
# ----------------------------------------------------
export PATH="$HOME/miniconda_envs/anaconda3/bin:$PATH"
cd ~/miniconda_envs
conda create -y -n GrooveTransformer python=3.9
```

- ##### Step3: Package Installation <a name="step3PipInstall"></a>
```bash
#!/bin/bash
#SBATCH -J env_setup
#SBATCH -p high
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8g
#SBATCH --time=24:00:00
#SBATCH -o %N.%J.env_setup.out
#SBATCH -e %N.%J.env_setup.er

# Step 3.  Install Required Packages
# ----------------------------------------------------
# 1. Activate Environment
# ---------------------
export PATH="$HOME/miniconda_envs/anaconda3/bin:$PATH"
export PATH="$HOME/miniconda_envs/anaconda3/envs/GrooveTransformer:$PATH"
source activate GrooveTransformer

# 2. Essential Packages
#    ------------------
pip install torch==1.12.0+cu102 --extra-index-url https://download.pytorch.org/whl/cu102
pip3 install ffmpeg==1.4
pip3 install holoviews==1.15.1
pip3 install wandb==0.13.3

# 3. MIDI/AUDIO Features (If needed)
#    -------------------------------
pip3 install note-seq==0.0.5
conda install -c conda-forge fluidsynth
pip3 install pyFluidSynth
```

- #### **Full Batch Script**  <a name="fullBatch"></a>

------

```bash
#!/bin/bash
#SBATCH -J env_setup
#SBATCH -p high
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8g
#SBATCH --time=24:00:00
#SBATCH -o %N.%J.env_setup.out
#SBATCH -e %N.%J.env_setup.er

# Step 1. Intalling Miniconda
# ----------------------------------------------------
mkdir ~/miniconda_envs
cd ~/miniconda_envs
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p ~/miniconda_envs/anaconda3

# Step 2. Create a new Conda Env
# ----------------------------------------------------
export PATH="$HOME/miniconda_envs/anaconda3/bin:$PATH"
cd ~/miniconda_envs
conda create -y -n GrooveTransformer python=3.9


# Step 3.  Install Required Packages
# ----------------------------------------------------
# 1. Activate Environment
# ---------------------
export PATH="$HOME/miniconda_envs/anaconda3/bin:$PATH"
export PATH="$HOME/miniconda_envs/anaconda3/envs/GrooveTransformer:$PATH"
source activate GrooveTransformer

# 2. Essential Packages
#    ------------------
pip install torch==1.12.0+cu102 --extra-index-url https://download.pytorch.org/whl/cu102
pip3 install ffmpeg==1.4
pip3 install holoviews==1.15.1
pip3 install wandb==0.13.3

# 3. MIDI/AUDIO Features (If needed)
#    -------------------------------
pip3 install note-seq==0.0.5
conda install -c conda-forge fluidsynth
pip3 install pyFluidSynth
```
































<!---
##### 3.2.2 Anaconda (using lmod softwares available on the cluster) <a name="3.2.2"></a>

>  **Warning** You can prepare the environment using `Anaconda`. This can be done either using an 
> [interactive session](https://guiesbibtic.upf.edu/recerca/hpc/interactive-jobs), or by submitting a 
> [remote job](https://guiesbibtic.upf.edu/recerca/hpc/basic-jobs) using `sbatch` command. 
> That said, the installation may take a long time, as a result, we highly suggest preparing the environment 
> remotely

A template shell file is available in [demos/installation/env_setup_conda_gpu.sh](../../demos/installation/env_setup_conda_gpu.sh).
The content of this file is as follows:

```shell
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

activate GrooveTransformer

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
```

> **Note** If you don't have previously created a conda environment, you need to uncomment the 
> `conda create --name GrooveTransformer python=3.9` line, or simply create the environment using the following
> set of commands in an interactive session prior to submitting the above shell file remotely
> ```shell
> # Get access to a cpu-only node on the short partition
> srun --nodes=1 --partition=short --cpus-per-task=4 --mem=8g --pty bash -i
> 
> # Get access to existing modules available on the cluster  
> source /etc/profile.d/lmod.sh
> source /etc/profile.d/zz_hpcnow-arch.sh
> 
> # Load Anaconda
> module load Anaconda3/2020.02
> 
> # Create GrooveTransformer Conda Environment
> conda create --name GrooveTransformer python=3.9
>
> ```

> **Warning** It is possible that the Anaconda3/2020.02 module is updated and this version is no longer
> available. In such case, run `module avail` or  `module spider Anaconda` to see which versions are available.
> 
> **Note** Make sure you have sourced `/etc/profile.d/lmod.sh` and  `/etc/profile.d/zz_hpcnow-arch.sh` to be able 
> to use the `module` command. 






##### 3.2.3 venv <a name="3.2.3"></a>
Login to a computation node [**interactively using srun**](https://github.com/behzadhaki/GrooveTransformer/blob/main/documentation/Chapter%200_PyEnvAndCluster/HPC_Cluster_Guide.md#22-interactive-sessions-)
```commandline
srun ....
```
Create a folder called venv in your $HOME$ directory, then create a 
```commandline
cd ~
mkdir venv
cd venv
python3 -m venv GrooveTransformer
```
Then activate the virtual environment and install the required packages:
```commandline
cd ..
source ~/venv/GrooveTransformer/bin/activate
```

Following the activation of the virtual environment, you can install the required packages using the following commands:

```commandline
pip3 install --upgrade pip
pip install torch==1.10.2+cu102 --extra-index-url https://download.pytorch.org/whl/cu113
pip3 install bokeh==2.3.3
pip3 install colorcet==3.0.0
pip3 install fluidsynth==0.2
pip3 install holoviews==1.15.1
pip3 install librosa==0.9.2
pip3 install matplotlib==3.6.0
pip3 install note_seq==0.0.5
pip3 install numpy==1.23.3
pip3 install pandas==1.5.0
pip3 install pretty_midi==0.2.9
pip3 install pyFluidSynth==1.3.1
pip3 install PyYAML==6.0
pip3 install scikit_learn==1.1.3
pip3 install scipy==1.9.1
pip3 install soundfile==0.11.0
pip3 install tqdm==4.64.1
pip3 install wandb==0.13.3
```
--->