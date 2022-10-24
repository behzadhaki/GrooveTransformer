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
3. [Accessing Evaluation Results](https://github.com/behzadhaki/VariationalMonotonicGrooveTransformer/blob/main/documentation/chapter3_Evaluator/2_standalone_usage.md#3-accessing-evaluation-results-)
   1. [Results as Dictionaries or Pandas.DataFrame](https://github.com/behzadhaki/VariationalMonotonicGrooveTransformer/blob/main/documentation/chapter3_Evaluator/2_standalone_usage.md#31-results-as-dictionaries-or-pandasdataframe-)
   2. [Rendering Results as Bokeh Plots](https://github.com/behzadhaki/VariationalMonotonicGrooveTransformer/blob/main/documentation/chapter3_Evaluator/2_standalone_usage.md#32-rendering-results-as-bokeh-plots-)
   3. [Rendering Piano Rolls/Audio/Midi](https://github.com/behzadhaki/VariationalMonotonicGrooveTransformer/blob/main/documentation/chapter3_Evaluator/2_standalone_usage.md#33-rendering-piano-rollsaudiomidi-)
4. [Ready-to-use Evaluator Templates](#4)       # TODO - not implemented yet
   
## 1. Introduction <a name="1"></a>

----

- There are a number of evaluation tools available in the `eval` directory. 
   These tools are used to evaluate the performance of the model and better inspect the generated outputs.
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

