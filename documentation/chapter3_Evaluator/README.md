# Chapter 3 - Evaluation Tools

-----

# Table of Contents
1. [Introduction](#1)
2. [GrooveEvaluator Basics](https://github.com/behzadhaki/VariationalMonotonicGrooveTransformer/blob/main/documentation/chapter3_Evaluator/1_grooveevalbasics.md#2-grooveevaluator-basics-)
   1. [Prepapre the data used for Evaluation](https://github.com/behzadhaki/VariationalMonotonicGrooveTransformer/blob/main/documentation/chapter3_Evaluator/1_grooveevalbasics.md#21-prepapre-the-data-used-for-evaluation-)
   2. [Initialization](https://github.com/behzadhaki/VariationalMonotonicGrooveTransformer/blob/main/documentation/chapter3_Evaluator/1_grooveevalbasics.md#22-initialization-)
   3. [Evaluating Predictions](https://github.com/behzadhaki/VariationalMonotonicGrooveTransformer/blob/main/documentation/chapter3_Evaluator/1_grooveevalbasics.md#23-preparing-predictions-)
      1. [Get Ground Truth Samples](https://github.com/behzadhaki/VariationalMonotonicGrooveTransformer/blob/main/documentation/chapter3_Evaluator/1_grooveevalbasics.md#231-get-ground-truth-samples--)
      2. [Pass Samples to Model](https://github.com/behzadhaki/VariationalMonotonicGrooveTransformer/blob/main/documentation/chapter3_Evaluator/1_grooveevalbasics.md#232-pass-samples-to-model-)
      3. [Add Predictions to Evaluator](https://github.com/behzadhaki/VariationalMonotonicGrooveTransformer/blob/main/documentation/chapter3_Evaluator/1_grooveevalbasics.md#233-add-predictions-to-evaluator-)
   4. [Saving and Loading](https://github.com/behzadhaki/VariationalMonotonicGrooveTransformer/blob/main/documentation/chapter3_Evaluator/1_grooveevalbasics.md#24-saving-and-loading-)  
3. [Accessing Evaluation Results](https://github.com/behzadhaki/VariationalMonotonicGrooveTransformer/blob/main/documentation/chapter3_Evaluator/2_grooveeval_standalone_usage.md#3-accessing-evaluation-results-)
   1. [Results as Dictionaries or Pandas.DataFrame](https://github.com/behzadhaki/VariationalMonotonicGrooveTransformer/blob/main/documentation/chapter3_Evaluator/2_grooveeval_standalone_usage.md#31-results-as-dictionaries-or-pandasdataframe-)
   2. [Rendering Results as Bokeh Plots](https://github.com/behzadhaki/VariationalMonotonicGrooveTransformer/blob/main/documentation/chapter3_Evaluator/2_grooveeval_standalone_usage.md#32-rendering-results-as-bokeh-plots-)
   3. [Rendering Piano Rolls/Audio/Midi](https://github.com/behzadhaki/VariationalMonotonicGrooveTransformer/blob/main/documentation/chapter3_Evaluator/2_grooveeval_standalone_usage.md#33-rendering-piano-rollsaudiomidi-)
4. [Compiling Plots for Logging](https://github.com/behzadhaki/VariationalMonotonicGrooveTransformer/blob/main/documentation/chapter3_Evaluator/2_grooveeval_standalone_usage.md#4-compiling-logging-media-)
5. [Ready-to-use Evaluator Templates](#5)       # TODO - not implemented yet
   
## 1. Introduction <a name="1"></a>

----

- There are a number of evaluation tools available in the `eval` directory. 
   These tools are used to evaluate the performance of the model and better inspect the generated outputs.
   - Here is a summary of these tools:
     - [GrooveEvaluator](#2): 
         - This is the main evaluation tool. It is used to evaluate the performance of the model and better inspect the generated outputs.
         - Using this tool, we can extract features, cross compare generations with ground truth, and synthesize/visualize the results.
 
     - [MultiSetEvaluator](#3): 
       - Allows for comparing the performance of different models on the same dataset.
       - Allows for comparing the performance of the same model on different datasets.
       - Allows for comparing the performance of different models on different datasets.
       - This tool is mostly intended to work outside of the training process.
