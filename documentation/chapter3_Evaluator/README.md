# Chapter 3 - Evaluation Tools

-----

# Table of Contents
1. [Introduction](#1)
2. [GrooveEvaluator Basics](https://github.com/behzadhaki/VariationalMonotonicGrooveTransformer/blob/main/documentation/chapter3_Evaluator/1_basics.md#2-grooveevaluator-basics-)
   1. [Prepapre the data used for Evaluation](https://github.com/behzadhaki/VariationalMonotonicGrooveTransformer/blob/main/documentation/chapter3_Evaluator/1_basics.md#21-prepapre-the-data-used-for-evaluation-)
   2. [Initialization](https://github.com/behzadhaki/VariationalMonotonicGrooveTransformer/blob/main/documentation/chapter3_Evaluator/1_basics.md#22-initialization-)
   3. [Evaluating Predictions](https://github.com/behzadhaki/VariationalMonotonicGrooveTransformer/blob/main/documentation/chapter3_Evaluator/1_basics.md#23-preparing-predictions-)
      1. [Get Ground Truth Samples](https://github.com/behzadhaki/VariationalMonotonicGrooveTransformer/blob/main/documentation/chapter3_Evaluator/1_basics.md#231-get-ground-truth-samples--)
      2. [Pass Samples to Model](https://github.com/behzadhaki/VariationalMonotonicGrooveTransformer/blob/main/documentation/chapter3_Evaluator/1_basics.md#232-pass-samples-to-model-)
      3. [Add Predictions to Evaluator](https://github.com/behzadhaki/VariationalMonotonicGrooveTransformer/blob/main/documentation/chapter3_Evaluator/1_basics.md#233-add-predictions-to-evaluator-)
   4. [Saving and Loading](https://github.com/behzadhaki/VariationalMonotonicGrooveTransformer/blob/main/documentation/chapter3_Evaluator/1_basics.md#24-saving-and-loading-) 
3. [Accessing Evaluation Results](#3)
   1. [Results for general inspection](#3_i)
   2. [Get Evaluation Results for `WandB`](#3_ii)
4. [Ready-to-use Evaluator Templates](#4)       # TODO - not implemented yet
   
## 1. Introduction <a name="1"></a>

----

- There are a number of evaluation tools available in the `evaluator` directory. 
   These tools are used to evaluate the performance of the model and better inspect the generated output.
   - Here is a summary of these tools:
     - [GrooveEvaluator](#2): 
       - Allows for automatically extracting the `global features` already implemented in the HVO_Sequence class 
       ([example](https://wandb.ai/mmil_tap2drum/transformer_groove_tap2drum/reports/global_feature_pdfs-Test_Set_Predictions-22-10-14-18-15-10---VmlldzoyNzk2Mzg5)).  
       - Allows for generating `velocity heatmaps` ([example](https://wandb.ai/mmil_tap2drum/transformer_groove_tap2drum/reports/velocity_heatmaps-Test_Set_Predictions-22-10-14-18-16-18---VmlldzoyNzk2Mzkz)).
       - Allows for automatically `synthesizing` generated patterns into audio ([example](https://wandb.ai/mmil_tap2drum/transformer_groove_tap2drum/reports/audios-Validation_Set_Predictions-22-10-14-18-18-30---VmlldzoyNzk2NDAw))
       - Allows for displaying the generations as `piano rolls` ([example](https://wandb.ai/mmil_tap2drum/transformer_groove_tap2drum/reports/piano_roll_html-Validation_Set_Predictions-22-10-14-18-17-13---VmlldzoyNzk2Mzk1)).
       - This tool is designed to smoothly work within the training pipeline (using `W&B`), as well as, outside of the training process.
       - All the analysis done in this section `can be conducted on different subset groups` of the dataset. 
       For example, the dataset can be split into different genres so as to analyze the performance of the model on each genre separately. 


## 3. Accessing Evaluation Results <a name="3"></a>
---

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


