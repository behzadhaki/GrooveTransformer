

## 1. Introduction <a name="1"></a>

VAEDER (Variational Auto Encoder for Disentangled Expressive Rhythms)
is designed to convert a monophonic, tapped rhythm, as well as a number of control parameters
(density, intensity, genre) into a 9-voice drum performance. 

In this guide, we will briefly walk through the basics of loading, saving, serializing and generating with VAEDER. 

## 2. Instantiation <a name="1"></a>

We use a dictionary to store all the necessary parameters. Below is an example
of a typical instantiation.

```python
from model import GrooveControl_VAE
config = config = {
    'd_model_enc': 16,
    'd_model_dec': 16,
    'embedding_size_src': 3,
    'embedding_size_tgt': 27,
    'nhead_enc': 2,
    'nhead_dec': 4,
    'dim_feedforward_enc': 128,
    'dim_feedforward_dec': 256,
    'num_encoder_layers': 2,
    'num_decoder_layers': 8,
    'latent_dim': 512,

    'dropout': 0.2,
    'velocity_dropout': 0.3,
    'offset_dropout': 0.5,

    'max_len_enc': 32,
    'max_len_dec': 32,

    'device': 'cpu',
    'o_activation': 'tanh',

    'use_in_attention': True,
    'n_continuous_params': 2,
    'n_genres': 16}

model = GrooveControl_VAE(config)
```

### 2.1 Loading a checkpoint <a name="2_i"></a>

We provide a utility function to quickly load a pre-trained model,
including its genre mapping dictionary. 

```python
from helpers import load_vaeder_model
model_path = "checkpoints/model.pth" #replace with path to checkpoint
model = load_vaeder_model(model_path)
```

## 3. Saving The Model<a name="1"></a>
Similar to above, you can easily save the model in pth format:
```python
model.save("my_model.pth")
```
### C++ Serialization <a name="2_i"></a>
Serializing a model in pytorch allows you to load it in Torchscript,
a C++ version of pytorch that is helpful for realtime audio applications.
```python
name = "groover"
save_path = "models/serialized"

model.serialize_whole_model(name, save_path)
```
This will save the model in the .pt format, which can then be loaded in C++ applications, such as NeuralMidiFx.


## 4. Generation<a name="1"></a>

The model takes several inputs. First let's define the shape of each tensor, with N being the batch size.

- HVO: [N, 32, 27]
- Density: [N]
- Intensity: [N]
- Genre: [N, n_genres]

If you are sampling from a random z vector:
- z: [N, latent_dim]


Here, we will demonstrate how to generate a drum pattern from a test set tapped input, as well
as how to generate from a random z vector. In both cases we can save it as a MIDI file for further evaluation.

First, let's grab a random sample from our test set:
```python
import torch
from data.dataLoaders import load_gmd_hvo_sequences
test_set = load_gmd_hvo_sequences(
    "data/gmd/resources/storedDicts/groove_2bar-midionly.bz2pickle", "gmd", "data/dataset_json_settings/4_4_Beats_gmd.json", [4],
    "ROLAND_REDUCED_MAPPING", "train")
input_hvo_seq = test_set[np.random.randint(0, len(test_set))]
input_hvo = torch.tensor(input_hvo_seq.flatten_voices(), dtype=torch.float32)
```

Now let's set the control parameters
```python
density = torch.Tensor([0.75])
intensity = torch.Tensor([0.3])

genre_tag = "rock"
genre_dict = model.get_genre_dict()
genre_id = genre_dict[genre_tag]
genre = torch.nn.functional.one_hot(torch.tensor([genre_id]),
                                                  num_classes=len(genre_dict)).to(dtype=torch.float32)
```

Run the forward pass of our model, using the 'predict' function to include sampling.

```python
hvo, _, _, _ = model.predict(input_hvo, density, intensity, genre, return_concatenated=True)
hvo = torch.squeeze(hvo, dim=0).numpy()
```

Save it as a MIDI file for inspection and playback

```python
from copy import deepcopy
output_seq = deepcopy(input_hvo_seq)
output_seq.hvo = hvo
filename = "my_midi_file.mid"
output_seq.save_hvo_to_midi(filename=filename)
```

### Generating from a random latent vector <a name="2_i"></a>

We can use the same method described above, except with a random z vector instead of an 
input drum pattern. This bypasses the encoder part of our model. 

Different models have a different latent dimensionality, so we need to obtain this first. 
```python
latent_dim = model.get_latent_dim()
z = torch.rand(1, latent_dim).to(dtype=torch.float32)
hvo = model.decode(z, density, intensity, genre)
```
You can save this as a MIDI file with the same process highlighted above. 