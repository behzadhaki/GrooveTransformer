# Preprocessing Data for Training / Testing / Evaluating
## Instructions for preprocessing data into HVO_Sequence format to be used in training/evaluation pipelines

----

## Groove Midi Dataset Using `TFDS`

To download and preprocess the groove midi dataset with the metadata available, the easiest way is to use 
[ TensorFlow Datasets (TFDS)](https://www.tensorflow.org/datasets). We will be using the `midionly` sets available
in [Groove TFDS](https://www.tensorflow.org/datasets/catalog/groove). For the initial experiments, 
we use the [groove/2bar-midionly](https://www.tensorflow.org/datasets/catalog/groove#groove2bar-midionly) 
set.

For preprocessing the set, do the following:

1. activate magenta conda env (for installation, see [Preparing the Conda environment](#env_instructions))
    
    [mac] source /Users/[username]/miniconda3/etc/profile.d/conda.sh
    source activate magenta		        	# Activate env
    
2. run compile.py
   

    python3 compile.py


## Preparing the Conda environment<a name="env_instructions"></a>

--- 

1. create conda environment 


    conda create --name magenta python=3.6   	# Create env
    
2. activate environment


    [mac] source /Users/[username]/miniconda3/etc/profile.d/conda.sh
    [cluste] export PATH="$HOME/project/anaconda3/bin:$PATH"

    source activate magenta		        	# Activate env

3. install packages and add env to jupyter kernels


    pip install visual_midi				                        
    pip install tables 					                        
    pip install magenta==1.1.7 --use-deprecated=legacy-resolver	
    pip install note_seq				                        
    pip install pandas
    pip install ipyparams
    conda install jupyter
    conda install -c anaconda ipykernel
    python3 -m ipykernel install --user --name=magenta


