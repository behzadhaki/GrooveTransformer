# Chapter 3 - Evaluation Tools (Part B - Accessing Evaluation Results)

-----

# Table of Contents
3. [Accessing Evaluation Results](#3)
   1. [Results for general inspection](#3_i)
   2. [Get Evaluation Results for `WandB`](#3_ii)
      
   
## 2. GrooveEvaluator <a name="2"></a>

---

### 2.4. Evaluating Predictions <a name="2_iv"></a>

Here is a visual guide to using the evaluator for evaluating predictions
<img src="img.png" width="600">

#### 2.4.1. Get Ground Truth Samples  <a name="2_iv_a"></a>
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

#### 2.4.2 Pass Samples to Model <a name="2_iv_b"></a>
```python
predicted_hvos_array = model.predict(input)
```

#### 2.4.3 Add Predictions to Evaluator <a name="2_iv_c"></a>

```python
evaluator_test_set.add_predictions(predicted_hvos_array)
```

#### 2.4.4. Get Evaluation Results <a name="2_iv_d"></a>
The evaluation results can be obtained as a dictionary to be further inspected. Also, they can be obtained as a WANDB ready dictionary to be logged to WANDB.


##### 2.4.4.1 Results for general inspection <a name="2_iv_d_i"></a>
Get the results using the `get_logging_dict` method.

```python
_gt_logging_data, _predicted_logging_data = evaluator_test_set.get_logging_dict()
```

> **Note** Many times during training, you don't need to re-log the ground truth data. 
> In such cases, you can only get the logging dict for the predicted data.
> ```python
>     _predicted_logging_data = evaluator_test_set.get_logging_dict(need_groundTruth=False)
>```


The resulting dictionaries (`_gt_logging_data`, `_predicted_logging_data`) have the following format:

``` python
{
    'velocity_heatmaps': bokeh.models.layouts.Tabs Figure,
     'global_feature_pdfs': bokeh.models.layouts.Tabs Figure, 
     'piano_rolls': bokeh.models.layouts.Tabs Figure,
     'captions_audios': tuple(str, numpy.ndarray), 
}
```

> **Note** The above keys are only available if the `need_heatmap`, `need_global_features`, `need_audio`, 
> `need_piano_roll` parameters are set to True during initialization.

The `velocity_heatmaps`, `global_feature_pdfs`, `piano_rolls` are all tabulated [bokeh](https://docs.bokeh.org/en/latest/#) plots 
and the `captions_audios` are tuples where the first element is the caption and the second element is the audio array.

Here is an example to display the plots or save the audio files
```python
# Show bokeh figures
from bokeh.io import show
show(_predicted_logging_data['velocity_heatmaps'])
show(_predicted_logging_data['global_feature_pdfs'])
show(_predicted_logging_data['piano_rolls'])


# Save audio files
import os
import soundfile as sf
def save_wav_file(filename, data, sample_rate):
    # make directories if filename has directories
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    # save file
    sf.write(filename, data, sample_rate)

sample_audio_tuple = _predicted_logging_data['captions_audios'][0]
fname = sample_audio_tuple[0]
data = sample_audio_tuple[1]
save_wav_file(os.path.join("misc", fname), data, 44100)
```


##### 2.4.4.2 Results for `WandB` <a name="2_iv_d_ii"></a>
Same as above, but the results are in a format that can be directly logged to `WandB` using the `wandb.log` method.

# # TODO: HERE


