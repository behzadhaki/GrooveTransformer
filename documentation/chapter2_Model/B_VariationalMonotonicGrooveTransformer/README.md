# Chapter 2 - Models: Instantiating, Storing, Loading, Generating

----

# Table of Contents
1. [Introduction](#1)
2. [MonotonicGrooveVAE.GrooveTransformerEncoderVAE](#2)
   1. [Instantiation](#2_i)
   2. [Storing](#2_ii)
   3. [Loading](#2_iii)
   4. [Pretrained Versions](#2_iv)
   5. [Generation](#2_v)


## 1. Introduction <a name="1"></a>


## 2. `MonotonicGrooveVAE.GrooveTransformerEncoderVAE` <a name="3"></a>
This is a model that can take in a piano-roll-like monotonic groove and output a piano-roll-like drum pattern.
This model consists of the encoder part of the original transformer model, and a decoding part implemented with the encoder part. 

We have many pretrained versions of this model available (see [Pretrained Versions](#2_iv)). I suggest reading
this [document](https://behzadhaki.com/blog/2022/trainingGrooveTransformer/) 
to better understand the training/evaluation process.



If you use this model, please cite the following [paper](
https://behzadhaki.com/assets/pdf/Haki_2022__Real-Time_Drum_Accompaniment_Using_Transformer_Architecture.pdf):
```citation
@article{hakireal,
  title={Real-Time Drum Accompaniment Using Transformer Architecture},
  author={Haki, Behzad and Nieto, Marina and Pelinski, Teresa and Jord{\`a}, Sergi}
  booktitle={Proceedings of the 3rd Conference on AI Music Creativity, AIMC}
  year={2022}
}
```

### 2.i Instantiation <a name="2_i"></a>
A groove transformer consisting of the 
[transformerEncoder](https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoder.html#torch.nn.TransformerEncoder)
only section of the original transformer

Source code available [here](../../demos/model/VariationalMonotonicGrooveTransformer/GrooveTransformerEncoderVAE_test.py)

```python
params = {
  'd_model_enc': 128,
  'd_model_dec': 512,
  'embedding_size_src': 9,
  'embedding_size_tgt': 27,
  'nhead_enc': 2,
  'nhead_dec': 4,
  'dim_feedforward_enc': 16,
  'dim_feedforward_dec': 32,
  'num_encoder_layers': 3,
  'num_decoder_layers': 5,
  'dropout': 0.1,
  'latent_dim': 32,
  'max_len_enc': 32,
  'max_len_dec': 32,
  'device': 'cpu',
  'o_activation': 'sigmoid',
  'batch_size': 8 }

# test transformer

from model import GrooveTransformerEncoderVAE
device = 'cuda' if torch.cuda.is_available() else 'cpu'
config.update({'device': device})

TM = GrooveTransformer(config)
```

### 2.ii Storing <a name="2_ii"></a>
The models have a `save` method which can be used to store the model parameters. 
The `save` method takes in a  `**.pth` file path where the model attributes are to be stored. 
The model parameters as well as the model state dictionary are stored in the stored file.

```python
model_path = "model/misc/???/rand_model.pth"
TEM.save(model_path)
```

Using this method, a `**.json` file is also created which stores the model parameters. The data stored in 
this json file is already available in the dictionary stored in the `.pth` file. The json file is created
for conveniently inspecting the model params.

### 2.iii Loading <a name="2_iii"></a>

```python
## 4. Loading a Stored Model <a name="4"></a>

Source code available [here](../../testers/model/monotonic_groove_transformer_v1/LoaderSamplerDemo.py)

from helpers import load_variational_mgt_model
import torch

# Model path and model_param dictionary
model_name = f"{wandb_project}/{run_name}_{run_id}/{ep_}"
model_path = f"misc/VAE/{model_name}.pth"


# 1. LOAD MODEL
device = 'cuda' if torch.cuda.is_available() else 'cpu'
GrooveTransformer = load_variational_mgt_model(model_path, device=device)

```

### 2.iv Pretrained Versions <a name="3_iv"></a>
Four pretrained versions of this model are available. The models are trained according to the documents discussed above
in the introduction section. The models are available in the `model/saved/monotonic_groove_transformer_v1` directory.

The models are:


To load the model, use the `load_variational_mgt_model` 

```python





```
### 2.v Generation <a name="3_v"></a>
Source code available [here](../../demos/model/monotonic_groove_transformer_v1/LoaderSamplerDemo.py)

Create am input groove ([create a HVO_Sequence instance](https://github.com/behzadhaki/GrooveTransformer/blob/main/documentation/chapter1_Data/README.md#create-a-score-),
[load a midi file](https://github.com/behzadhaki/GrooveTransformer/blob/main/documentation/chapter1_Data/README.md#load-from-midi-), 
or [grab one from the HVO_Sequence datasets as below](https://github.com/behzadhaki/GrooveTransformer/blob/main/documentation/chapter1_Data/README.md#load-from-midi-)
```python
from data.dataLoaders import load_gmd_hvo_sequences
test_set = load_gmd_hvo_sequences(
    "data/gmd/resources/storedDicts/groove_2bar-midionly.bz2pickle", "gmd", "data/dataset_json_settings/4_4_Beats_gmd.json", [4],
    "ROLAND_REDUCED_MAPPING", "train")
input_hvo_seq = test_set[np.random.randint(0, len(test_set))]
input_groove_hvo = torch.tensor(input_hvo_seq.flatten_voices(), dtype=torch.float32)
```

Pass groove to model and sample a drum pattern
```python


```


Inspect generations by synthesizing to audio [link to documentation], 
store to midi [link to documentation] , or plot pianorolls [link to documentation]
```python


```
