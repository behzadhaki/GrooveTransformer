# Chapter 3 - Evaluation Tools
## Part A - GrooveEvaluator Basics

-----

# Table of Contents
[GrooveEvaluator Basics](#2)
1. [Prepapre the data used for Evaluation](#2_i)
2. [Initialization](#2_ii)
3. [Preparing Predictions](#2_iii)
   1. [Get Ground Truth Samples](#2_iii_a)
   2. [Pass Samples to Model](#2_iii_b)
   3. [Add Predictions to Evaluator](#2_iii_c)
4. [Saving and Loading](#2_iv)

## 2. GrooveEvaluator Basics <a name="2"></a>

### _All codes provided below are also available [here](../../testers/evaluator/01_Basics_demo.py)_


---

### 2.1. Prepapre the data used for Evaluation <a name="2_i"></a>

First, Load the dataset (read more about loading the dataset [here](https://github.com/behzadhaki/VariationalMonotonicGrooveTransformer/blob/main/documentation/chapter1_Data/README.md#313-load-gmd-dataset-in-hvo_sequence-format-using-a-single-command---))
```python
from data.dataLoaders import load_gmd_hvo_sequences

dataset_setting_json_path = "data/dataset_json_settings/4_4_Beats_gmd.json"
test_set = load_gmd_hvo_sequences(
    "data/gmd/resources/storedDicts/groove_2bar-midionly.bz2pickle",
    "gmd", dataset_setting_json_path, "test")
```

Then, if you want to inspect different subsets of the data, specify how the dataset needs to be split into smaller subsets.
```python
list_of_filter_dicts_for_subsets = []
styles = [
    "afrobeat", "afrocuban", "blues", "country", "dance", "funk", "gospel", "highlife", "hiphop", "jazz",
    "latin", "middleeastern", "neworleans", "pop", "punk", "reggae", "rock", "soul"]
for style in styles:
    list_of_filter_dicts_for_subsets.append(
        {"style_primary": [style], "beat_type": ["beat"], "time_signature": ["4-4"]}
    )
```

> **Note**
> In the above case, "beat_type" and "time_signature" are unnecessary as the original dataset (specified in `data/dataset_json_settings/4_4_Beats_gmd.json`) 
> is already filtered to only include `beat` and `4-4` patterns. However, if you want to inspect a subset of the dataset that is not  
> filtered in the original dataset, you can specify the filters here. As a result, in this example, it makes more sense to use the following filters
> 

```python
list_of_filter_dicts_for_subsets = []
styles = [
    "afrobeat", "afrocuban", "blues", "country", "dance", "funk", "gospel", "highlife", "hiphop", "jazz",
    "latin", "middleeastern", "neworleans", "pop", "punk", "reggae", "rock", "soul"]
for style in styles:
    list_of_filter_dicts_for_subsets.append(
        {"style_primary": [style]}
    )
```


### 2.2. Initialization <a name="2_ii"></a>

To instantiate the GrooveEvaluator, you need to specify the following parameters:

- **hvo_sequences_list**: A 1D list of HVO_Sequence objects corresponding to ground truth data
- **list_of_filter_dicts_for_subsets**: (Default: None, means use all data without subsetting) The filter dictionaries using which the dataset will be subsetted into different groups. Note that the HVO_Sequence objects must contain `metadata` attributes with the keys specified in the filter dictionaries.
- **_identifier**: A string label to identify the set of HVO_Sequence objects. This is used to name the output files.
- **n_samples_to_use**: (Default: -1, means use all data) The number of samples to use for evaluation in case you don't want to use all the samples. THese are randomly selected.
         (it is recommended to use the entirety of the dataset, if smaller subset is needed, process them externally prior to Evaluator initialization)
- **max_hvo_shape**: (Default: (32, 27)) The maximum shape of the HVO array. This is used to trim/pad the HVO arrays to the same shape.
- **need_heatmap**: (Default: True) Whether to generate velocity timing heatmaps
- **need_global_features**: (Default: True) Whether to generate global features plots
- **need_piano_roll**: (Default: True) Whether to generate piano roll plots
- **need_audio**: (Default: True) Whether to generate audio files
- **n_samples_to_synthesize**: (Default: "all") The number of samples to synthesize audio files for. If "all", all samples will be synthesized.
- **n_samples_to_draw_pianorolls**: (Default: "all") The number of samples to draw piano rolls for. If "all", all samples will be drawn.
- **disable_tqdm**: (Default: False) Whether to disable tqdm progress bars


> **Warning**
> The **_pianoroll and audio sample generations are extremely slow_**. If these are needed during the training process, 
> make sure you only use a small number of samples for generating these. 
> This can be done by manually specifying the n_samples_to_synthesize and n_samples_to_draw_pianorolls parameters. 
> These parameters are **per subset** of the dataset!

> **Recommendation**
> Talk about smaller downsampled datasets for different tasks
> 
> 



### 2.3. Preparing Predictions <a name="(#2_iii)"></a>

Some of the available evaluation methods can be run on the ground truth data. However, most of the methods require 
predictions. As a result, prior to running the evaluation, you need to : 
    (1) prepare the inputs to the model using the [ground truth data](#2_iv_a) available in the evaluator, 
    (2) [pass the inputs to a model](#2_iv_b) and format the results as a **[Batch Size, Time Steps, Num Voices * 3]** numpy array, 
    (3) [pass the formatted results to the evaluator](#2_iv_c) 


Here is a visual guide to using the evaluator for evaluating predictions
<img src="img.png" width="600">



#### 2.3.1. Get Ground Truth Samples  <a name="2_iii_a"></a>
Get the ground truth samples as a numpy array displaying piano rolls in HVO format.

```python
evaluator_test_set.get_ground_truth_hvos_array()
```

> **Warning** In groove transformer models, we don't want to feed this tensor to the model. 
> Instead, we want to feed the tapified versions (monotonic groove).
> To do so, we can get the ground truth HVO_Sequence samples and use the internal `flatten_voices` 
> method to get the groove.

```python
import numpy as np
input = np.array([hvo_seq.flatten_voices(voice_idx=2) for hvo_seq in evaluator_test_set.get_ground_truth_hvo_sequences()])
```

#### 2.3.2 Pass Samples to Model <a name="2_iii_b"></a>
from data.dataLoaders import load_gmd_hvo_sequences
test_set = load_gmd_hvo_sequences(
    "data/gmd/resources/storedDicts/groove_2bar-midionly.bz2pickle", "gmd", "data/dataset_json_settings/4_4_Beats_gmd.json", [4],
    "ROLAND_REDUCED_MAPPING", "train")
predicted_hvos_array = model.predict(input)

```python
```

#### 2.3.3 Add Predictions to Evaluator <a name="2_iii_c"></a>

```python
evaluator_test_set.add_predictions(predicted_hvos_array)
```

### 2.4. Saving and Loading <a name="2_iv"></a>

To save the model, use the `dump` method. You can the `path` parameter to specify the directory in which you want to store the evaluator. Also,
you can use the `fname`, to add aditional information.

```python
evaluator_test_set.dump(path="misc", fname="")

# Output >> Dumped Evaluator to path/test_set_full_fname.Eval.bz2
```

> **Note**
> During training replace `fname` with the epoch number and the training step number to be able to easily identify the evaluator.

To load a saved model, use the full path along with the `load_evaluator` method

```python
from eval.GrooveEvaluator.src.evaluator import load_evaluator
evaluator_test_set = load_evaluator("path/test_set_full_fname.Eval.bz2")
```

