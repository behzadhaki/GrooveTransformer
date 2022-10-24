## Chapter 1 - Data Representation & Datasets 

----

# Table of Contents
1. [Introduction](#1)
2. [Data Representation](#2)
   1. [HVO_Sequence](#2_1)
   2. [Example Code](#2_2)
3. [Datasets](#3)
   1. [Groove Midi Dataset](#3_1)
      1. [Load dataset as a dictionary](#3_1_1)
      2. [Extract `HVO_Sequence` objects from dataset dictionaries](#3_1_2)
      3. [**_Load GMD Dataset in `HVO_Sequence` format using a single command !!!_**](#3_1_3)
   2. [Process and Load as torch.utils.data.Dataset](#3_2)

   
## 1. Introduction <a name="1"></a>

---


<!--- ====================================================================================================== -->

## 2. Data Representation <a name="2"></a>

---

A custom datatype called `HVO_Sequence` is used for representing a drum pattern as a piano roll, 
using a sequence of equally (or in some cases unequally) fixed grid lines, velocities and offsets relative to the gridlines.



### 2.1 HVO_Sequence <a name="2_1"></a>

## Data Representation

When dealing with transformer architectures, commonly the input/output
space of possibilities is quantized, and then tokenized so as to learn a
meaningful representation space in which each of the input/output tokens
is embedded. However, for this work, we purposefully decided to replace
the tokenized representation of events with that of a direct
representation. To this end, we used the same representation as proposed
by [Gillick et. al. ](https://arxiv.org/abs/1905.06118). In this representation (from
now on, called HVO, denoting hits, velocities, and offsets), the input
and output sequences are directly represented by three stacked T × M
matrices, where T corresponds to the number of time-steps, in this case,
32 (2 bars with 16 sub-divisions each), and M corresponds to the number
of instruments, in this case, 9. The three matrices are deﬁned as
follows:

-   **Hits**: Binary-valued matrix that indicates the presence (1) or
    absence (0) of a drum hit.

-   **Velocity**: Continuous-valued matrix of velocity levels in the
    range \[0, 1\]

-   **Oﬀsets**: Continuous-valued matrix of oﬀset deviations from the
    nearest 16th note grid line, in the range \[−0.5, 0.5\] where ±0.5
    implies mid-way between a grid line and the following/preceding
    gridline

This results in an HVO matrix of dimension 32 × 27. An example of an HVO
representation (with 4 voices and 4 timesteps) is shown in the following image

![](https://assets.pubpub.org/0suupfy3/31643617978705.jpeg)<a name=nraqniwzatq></a>

<figcaption align = "center"><b>An example of HVO derivation from the piano rolls
</b></figcaption>


### 2.2 Example Code <a name="2_2"></a>

> **Note:** All the code examples in this section are available  [here](../../testers/HVO_Sequence/demo.py)


#### **create a score** <a name="createHVO"></a>

```
from hvo_sequence.hvo_seq import HVO_Sequence
from hvo_sequence.drum_mappings import ROLAND_REDUCED_MAPPING
from hvo_sequence.io_helpers import note_sequence_to_hvo_sequence, midi_to_hvo_sequence

import pretty_midi, note_seq

hvo_seq = HVO_Sequence(drum_mapping=ROLAND_REDUCED_MAPPING)

# ----------------------------------------------------------------
# -----------           CREATE A SCORE              --------------
# ----------------------------------------------------------------

# Add two time_signatures
t_sig = [4, 4]
beat_div_factor = [4]           # divide each quarter note in 4 divisions
t_stamp = 0
hvo_seq.add_time_signature(t_stamp, t_sig[0], t_sig[0], beat_div_factor)

# Add two tempos
hvo_seq.add_tempo(0, 50)

# Create a random hvo for 32 time steps and 9 voices
hvo_seq.random(32, 9)
```

#### **Access data using the .get() or .hvo method**
```

# ----------------------------------------------------------------
# -----------           Access Data                 --------------
# ----------------------------------------------------------------
hvo_seq.get("h")    # get hits
hvo_seq.get("v")    # get vel
hvo_seq.get("o")    # get offsets

hvo_seq.get("vo")    # get vel with offsets
hvo_seq.get("hv0")    # get hvwith offsets replaced as 0
hvo_seq.get("ovhhv0")    # use h v o and 0 to create any tensor
```
#### **Plot piano roll** <a name="pianoroll"></a>
```
# ----------------------------------------------------------------
# -----------           Plot PianoRoll              --------------
# ----------------------------------------------------------------
hvo_seq.to_html_plot("test.html", show_figure=True)
```

#### **save to midi** <a name="saveMidi"></a>
```
# ----------------------------------------------------------------
# -----------           Synthesize/Export           --------------
# ----------------------------------------------------------------
# Export to midi
hvo_seq.save_hvo_to_midi("misc/test.mid")
```

#### **convert to note_sequence**   <a name="toNoteSequence"></a>
```
# Export to note_sequece
hvo_seq.to_note_sequence(midi_track_n=10)
```

#### **Synthesize to (or save as) audio using a SoundFont** <a name="synthesize"></a>
```
# Synthesize to audio
audio = hvo_seq.synthesize(sr=44100, sf_path="hvo_sequence/soundfonts/Standard_Drum_Kit.sf2")
```
```
# Synthesize to audio and auto save
hvo_seq.save_audio(filename="misc/temp.wav", sr=44100,
                   sf_path="hvo_sequence/soundfonts/Standard_Drum_Kit.sf2")
```

#### **Load from midi** <a name="loadFromMidi"></a>
```
# ----------------------------------------------------------------
# -----------           Load from Midi             --------------
# ----------------------------------------------------------------
hvo_seq = midi_to_hvo_sequence('misc/test.mid', ROLAND_REDUCED_MAPPING, [4])

```

----

<!--- ====================================================================================================== -->

## 3. Datasets <a name="3"></a>

### 3.1 Groove Midi Dataset <a name="3_1"></a>
Magenta's [Groove MIDI Dataset
(GMD](https://www.tensorflow.org/datasets/catalog/groove#groove2bar-midionly)), a dataset containing roughly 13.6 hours of
drum performances in the format of beats and fills, classified by genre
and mostly in 4/4 time signature. 

![](https://assets.pubpub.org/f67l9ngf/21642506281264.png)
<figcaption align = "center"><b>Genre distribution in GMD (beats in 4-4 meter)
</b></figcaption>

The dataset in midi format is available [here](../../data/gmd/resources/source_dataset). 
The dataset can also be found as a [Groove TFDS](https://www.tensorflow.org/datasets/catalog/groove) in [ TensorFlow Datasets (TFDS)](https://www.tensorflow.org/datasets)

In order to avoid installing tensorflow, the groove TFDS subsets have been downloaded and stored as dictionaries of metadatas and midi files. 
Access these pickled dictionaries [here](../../data/gmd/resources/storedDicts):
    
```
1. groove_full-midionly.bz2pickle
2. groove_2bar-midionly.bz2pickle
3. groove_4bar-midionly.bz2pickle
```

These files are simply dictionaries of the following format
```
{
    'train', 'test' or 'validation':
        {
            'drummer': ...,
            'session': ...,
             'loop_id': ...,
             'master_id': ...,
             'style_primary': ...,
             'style_secondary': ..., 
             'bpm': ...,
             'beat_type': ...,
             'time_signature': ...,
             'full_midi_filename': ...,
             'full_audio_filename': ...,
              'midi': bytes,
              'note_sequence': ...,
        }
}
```

#### 3.1.1 Load dataset as a dictionary <a name="3_1_1"></a>       

> **Note:** All the code examples in this section are available [here](../../testers/data/demo.py)


```
from data.dataLoaders import load_original_gmd_dataset_pickle

# Load 2bar gmd dataset as a dictionary
gmd_dict = load_original_gmd_dataset_pickle(
    gmd_pickle_path="data/gmd/resources/storedDicts/groove_2bar-midionly.bz2pickle")


gmd_dict.keys()
# dict_keys(['train', 'test', 'validation'])
          
gmd_dict['train'].keys()        
#dict_keys(['drummer', 'session', 'loop_id', 'master_id', 'style_primary', 'style_secondary', 
'bpm', 'beat_type', 'time_signature', 'full_midi_filename', 'full_audio_filename', 
'midi', 'note_sequence'])

```
 

#### 3.1.2 Extract `HVO_Sequence` objects from dataset dictionaries above  <a name="3_1_2"></a>

> **Note:** All the code examples in this section are available [here](../../testers/data/demo.py)


```
# Extract HVO_Sequences from the dictionaries
from data.dataLoaders import extract_hvo_sequences_dict, get_drum_mapping_using_label

hvo_dict = extract_hvo_sequences_dict (
    gmd_dict=gmd_dict,
    beat_division_factor=[4],
    drum_mapping=get_drum_mapping_using_label("ROLAND_REDUCED_MAPPING"))
```

The resulting `hvo_dict` is a dictionary of the following format
```
{
    'train':
        list of HVO_Seqs corresponding to gmd_dict['train']['midi'][idx]
    'test':
        list of HVO_Seqs corresponding to gmd_dict['test']['midi'][idx]
    'validation':
        list of HVO_Seqs corresponding to gmd_dict['validation']['midi'][idx]    
}
```

#### 3.1.3 Load GMD Dataset in `HVO_Sequence` format using a single command !!!  <a name="3_1_3"></a>

> **Note:** All the code examples in this section are available [here](../../testers/data/demo.py)


```
from data.dataLoaders import load_gmd_hvo_sequences

train_set = load_gmd_hvo_sequences(
    dataset_setting_json_path = "data/dataset_json_settings/4_4_Beats_gmd.json", 
    subset_tag = "train", 
    force_regenerate=False)
```

### 3.2 Process and Load as torch.utils.data.Dataset <a name="3_2"></a>

