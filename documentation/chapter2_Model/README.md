# Chapter 2 - Models: Instantiating, Storing, Loading, Generating

----

# Table of Contents
1. [Introduction](#1)
2. [Instantiating a Model](#2)
   1. [BasicGrooveTransformer.GrooveTransformer](#2_i)
   2. [BasicGrooveTransformer.GrooveTransformerEncoder](#2_ii)
3. [Storing a Model](#3)
4. [Loading a Stored Model](#4)
5. [Generation using a Model](#5)

## 1. Introduction <a name="1"></a>

------------------------------------------------------------------

## 2. Instantiating a Model <a name="2"></a>

------------------------------------------------------------------

### 2.i `BasicGrooveTransformer.GrooveTransformer`  <a name="2_i"></a>

A groove transformer similar to the original transformer consisting of
[transformerEncoder](https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoder.html#torch.nn.TransformerEncoder) 
and [transformerDecoder](https://pytorch.org/docs/stable/generated/torch.nn.TransformerDecoder.html#torch.nn.TransformerDecoder).

Source code available [here](../../testers/model/monotonic_groove_transformer_v1/BasicGrooveTransformer_test.py)
```
# Instantiating a model
from model.src.BasicGrooveTransformer import GrooveTransformer

params = {
    "d_model": 128,
    "nhead": 4,
    "dim_forward": 256,
    "dropout": 0.1,
    "num_encoder_layers": 6,
    "num_decoder_layers": 6,
    "max_len": 32,
    "N": 64,             # batch size
    "embedding_size_src": 16,   # input dimensionality at each timestep
    "embedding_size_tgt": 27    # output dimensionality at each timestep
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'

TM = GrooveTransformer(params["d_model"], params["embedding_size_src"], params["embedding_size_tgt"],
                       params["nhead"], params["dim_feedforward"], params["dropout"],
                       params["num_encoder_layers"], params["num_decoder_layers"], params["max_len"], device)
```

### 2.ii `BasicGrooveTransformer.GrooveTransformerEncoder` <a name="2_ii"></a>
A groove transformer consisting of the 
[transformerEncoder](https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoder.html#torch.nn.TransformerEncoder)
only section of the original transformer

Source code available [here](../../testers/model/monotonic_groove_transformer_v1/BasicGrooveTransformer_test.py)

```
from model.src.BasicGrooveTransformer import GrooveTransformerEncoder

params = {
    "d_model": 128,
    "nhead": 4,
    "dim_forward": 256,
    "dropout": 0.1,
    "num_layers": 6,
    "max_len": 32,
    "N": 64,  # batch size
    "embedding_size": 27
}


TEM = GrooveTransformerEncoder(params["d_model"], params["embedding_size"], params["embedding_size"],
                               params["nhead"], params["dim_forward"], params["dropout"],
                               params["num_layers"], params["max_len"], device)
```

## 3. Storing a Model <a name="3"></a>

------------------------------------------------------------------

### _todo: implement a save method_ 


## 4. Loading a Stored Model <a name="4"></a>

------------------------------------------------------------------
Source code available [here](../../testers/model/monotonic_groove_transformer_v1/LoaderSamplerDemo.py)

```
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


## 5. Generation using a Model <a name="5"></a>

------------------------------------------------------------------
Source code available [here](../../testers/model/monotonic_groove_transformer_v1/LoaderSamplerDemo.py)

Create am input groove ([create a HVO_Sequence instance](https://github.com/behzadhaki/VariationalMonotonicGrooveTransformer/blob/main/documentation/chapter1_Data/README.md#create-a-score-),
[load a midi file](https://github.com/behzadhaki/VariationalMonotonicGrooveTransformer/blob/main/documentation/chapter1_Data/README.md#load-from-midi-), 
or [grab one from the HVO_Sequence datasets as below](https://github.com/behzadhaki/VariationalMonotonicGrooveTransformer/blob/main/documentation/chapter1_Data/README.md#load-from-midi-)
```
from data.dataLoaders import load_gmd_hvo_sequences
test_set = load_gmd_hvo_sequences(
    "data/gmd/resources/storedDicts/groove_2bar-midionly.bz2pickle", "gmd", "dataset_setting.json", [4],
    "ROLAND_REDUCED_MAPPING", "train")
input_hvo_seq = test_set[np.random.randint(0, len(test_set))]
input_groove_hvo = torch.tensor(input_hvo_seq.flatten_voices(), dtype=torch.float32)
```

Pass groove to model and sample a drum pattern
```
from model.modelLoadesSamplers import get_prediction
voice_thresholds = [0.5] * 9           # per voice sampling thresholds
voice_max_count_allowed = [32] * 9     # per voice max number of hits allowed
output_hvo = get_prediction(GrooveTransformer, input_groove_hvo, voice_thresholds,
                            voice_max_count_allowed, return_concatenated=True)
```


Inspect generations by synthesizing to audio [link to documentation], 
store to midi [link to documentation] , or plot pianorolls [link to documentation]
```
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
