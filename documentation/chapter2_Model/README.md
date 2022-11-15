# Chapter 2 - Models: Instantiating, Storing, Loading, Generating

----

# Table of Contents
1. [Introduction](#1)
2. [BasicGrooveTransformer.GrooveTransformer](#2)
   1. [Instantiation](#2_i)
3. [BasicGrooveTransformer.GrooveTransformerEncoder](#3)
   1. [Instantiation](#3_i)
   2. [Storing](#3_ii)
   3. [Loading](#3_iii)
   4. [Pretrained Versions](#3_iv)
   5. [Generation](#3_v)


## 1. Introduction <a name="1"></a>

## 2. `BasicGrooveTransformer.GrooveTransformer`  <a name="2"></a>
This model is a full encoder/decoder transformer model similar to the original transformer in the paper [Attention is All You Need](https://arxiv.org/abs/1706.03762). It consists of a [transformerEncoder](https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoder.html#torch.nn.TransformerEncoder) and a [transformerDecoder](https://pytorch.org/docs/stable/generated/torch.nn.TransformerDecoder.html#torch.nn.TransformerDecoder).
The only thing is that this model is designed to work with piano-roll-like data. 

We have not yet trained any versions of this model, but the implementation is complete and ready to be trained. 



### 2.i Instantiation <a name="2_i"></a>
A groove transformer similar to the original transformer consisting of
[transformerEncoder](https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoder.html#torch.nn.TransformerEncoder) 
and [transformerDecoder](https://pytorch.org/docs/stable/generated/torch.nn.TransformerDecoder.html#torch.nn.TransformerDecoder).

Source code available [here](../../demos/model/monotonic_groove_transformer_v1/BasicGrooveTransformer_test.py)

```python
# Instantiating a model
from model.Base.BasicGrooveTransformer import GrooveTransformer

params = {
    "d_model": 128,
    "nhead": 4,
    "dim_forward": 256,
    "dropout": 0.1,
    "num_encoder_layers": 6,
    "num_decoder_layers": 6,
    "max_len": 32,
    "N": 64,  # batch size
    "embedding_size_src": 16,  # input dimensionality at each timestep
    "embedding_size_tgt": 27  # output dimensionality at each timestep
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'

TM = GrooveTransformer(params["d_model"], params["embedding_size_src"], params["embedding_size_tgt"],
                       params["nhead"], params["dim_feedforward"], params["dropout"],
                       params["num_encoder_layers"], params["num_decoder_layers"], params["max_len"], device)
```

## 3. `BasicGrooveTransformer.GrooveTransformerEncoder` <a name="3"></a>
This is a model that can take in a piano-roll-like monotonic groove and output a piano-roll-like drum pattern.
This model consists of only the encoder part of the original transformer model. 

We have many pretrained versions of this model available (see [Pretrained Versions](#3_iv)). I suggest reading
this [document](https://behzadhaki.com/blog/2022/trainingGrooveTransformer/) 
to better understand the training/evaluation process.

Moreover, a real-time plugin has already developed using this model. 
The plugin is available [here](https://github.com/behzadhaki/GrooveTransformer)

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

### 3.i Instantiation <a name="3_i"></a>
A groove transformer consisting of the 
[transformerEncoder](https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoder.html#torch.nn.TransformerEncoder)
only section of the original transformer

Source code available [here](../../demos/model/monotonic_groove_transformer_v1/BasicGrooveTransformer_test.py)

```python
from model.Base.BasicGrooveTransformer import GrooveTransformerEncoder

params = {
    'd_model': 512,
    'embedding_size_src': 27,
    'embedding_size_tgt': 27,
    'nhead': 1,
    'dim_feedforward': 64,
    'dropout': 0.25542373735391866,
    'num_encoder_layers': 10,
    'max_len': 32,
    'device': 'gpu'
}

from model import GrooveTransformerEncoder

device = 'cuda' if torch.cuda.is_available() else 'cpu'

TEM = GrooveTransformerEncoder(params["d_model"], params["embedding_size_src"], params["embedding_size_tgt"],
                               params["nhead"], params["dim_feedforward"], params["dropout"],
                               params["num_encoder_layers"], params["max_len"], device)
```

### 3.ii Storing <a name="3_ii"></a>
The models have a `save` method which can be used to store the model parameters. 
The `save` method takes in a  `**.pth` file path where the model attributes are to be stored. 
The model parameters as well as the model state dictionary are stored in the stored file.

```python
TEM.save("model/misc/rand_model.pth")
```

Using this method, a `**.json` file is also created which stores the model parameters. The data stored in 
this json file is already available in the dictionary stored in the `.pth` file. The json file is created
for conveniently inspecting the model params.

### 3.iii Loading <a name="3_iii"></a>
```python
## 4. Loading a Stored Model <a name="4"></a>

Source code available [here](../../testers/model/monotonic_groove_transformer_v1/LoaderSamplerDemo.py)

```python
from model.modelLoadesSamplers import load_groove_transformer_encoder_model
from model.saved.monotonic_groove_transformer_v1.params import model_params
import torch
import numpy as np

# Model path and model_param dictionary
model_name = "colorful_sweep_41"
model_path = f"model/saved/monotonic_groove_transformer_v1/{model_name}.model"
model_param = model_params[model_name]

# 1. LOAD MODEL
GrooveTransformer = load_groove_transformer_encoder_model(model_path, model_param)
checkpoint = torch.load(model_path, map_location=model_param['device'])
```

### 3.iv Pretrained Versions <a name="3_iv"></a>
Four pretrained versions of this model are available. The models are trained according to the documents discussed above
in the introduction section. The models are available in the `model/saved/monotonic_groove_transformer_v1` directory.

The models are:
- `misunderstood_bush_246`  --> https://wandb.ai/mmil_tap2drum/transformer_groove_tap2drum/runs/vxuuth1y
- `rosy_durian_248`         --> https://wandb.ai/mmil_tap2drum/transformer_groove_tap2drum/runs/2cgu9h6u
- `hopeful_gorge_252`       --> https://wandb.ai/mmil_tap2drum/transformer_groove_tap2drum/runs/v7p0se6e
- `solar_shadow_247`        --> https://wandb.ai/mmil_tap2drum/transformer_groove_tap2drum/runs/35c9fysk

To load the model, use the `load_groove_transformer_encoder_model` method from 
the `modelLoadesSamplers` module as discussed above. For example, to load the `misunderstood_bush_246` model,
use the following [code](../../demos/model/monotonic_groove_transformer_v1/load_pretrained_versions_available.py):

```python

```python
from model.Base.modelLoadesSamplers import load_groove_transformer_encoder_model

model_path = f"model/saved/monotonic_groove_transformer_v1/latest/misunderstood_bush_246.pth"
GrooveTransformer = load_groove_transformer_encoder_model(model_path)
```
### 3.v Generation <a name="3_v"></a>
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
from model.modelLoadesSamplers import get_prediction
voice_thresholds = [0.5] * 9           # per voice sampling thresholds
voice_max_count_allowed = [32] * 9     # per voice max number of hits allowed
output_hvo = get_prediction(GrooveTransformer, input_groove_hvo, voice_thresholds,
                            voice_max_count_allowed, return_concatenated=True)
```


Inspect generations by synthesizing to audio [link to documentation], 
store to midi [link to documentation] , or plot pianorolls [link to documentation]
```python
from hvo_sequence.hvo_seq import zero_like
input = input_hvo_seq
groove = zero_like(input_hvo_seq)                        # create template for groove hvo_sequence object
groove.hvo = input_groove_hvo.cpu().detach().numpy()                     # add score
output = zero_like(input_hvo_seq)                        # create template for output hvo_sequence object
output.hvo = output_hvo[0, :, :].cpu().detach().numpy()                    # add score


input.to_html_plot("in.html", show_figure=True)
groove.to_html_plot("groove.html", show_figure=True)
output.to_html_plot("output.html", show_figure=True)
```
