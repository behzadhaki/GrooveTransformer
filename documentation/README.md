
# List of Contents
### Guides for using the repository can be found here

---

[Chapter 1 - Data Representation & Datasets](https://github.com/behzadhaki/VariationalMonotonicGrooveTransformer/tree/main/documentation/chapter1_Data/README.md) 
---
1. [Introduction](https://github.com/behzadhaki/VariationalMonotonicGrooveTransformer/blob/main/documentation/chapter1_Data/README.md#1-introduction-)
2. [Data Representation](https://github.com/behzadhaki/VariationalMonotonicGrooveTransformer/blob/main/documentation/chapter1_Data/README.md#2-data-representation-)
   1. [HVO_Sequence](https://github.com/behzadhaki/VariationalMonotonicGrooveTransformer/blob/main/documentation/chapter1_Data/README.md#21-hvo_sequence-)
   2. [Example Code](https://github.com/behzadhaki/VariationalMonotonicGrooveTransformer/blob/main/documentation/chapter1_Data/README.md#22-example-code-)
3. [Datasets](https://github.com/behzadhaki/VariationalMonotonicGrooveTransformer/blob/main/documentation/chapter1_Data/README.md#3-datasets-)
   1. [Groove Midi Dataset](https://github.com/behzadhaki/VariationalMonotonicGrooveTransformer/blob/main/documentation/chapter1_Data/README.md#31-groove-midi-dataset-)
      1. [Load dataset as a dictionary](https://github.com/behzadhaki/VariationalMonotonicGrooveTransformer/blob/main/documentation/chapter1_Data/README.md#311-load-dataset-as-a-dictionary-)
      2. [Extract `HVO_Sequence` objects from dataset dictionaries](https://github.com/behzadhaki/VariationalMonotonicGrooveTransformer/blob/main/documentation/chapter1_Data/README.md#312-extract-hvo_sequence-objects-from-dataset-dictionaries-above--)
      3. [**_Load GMD Dataset in `HVO_Sequence` format using a single command !!!_**](https://github.com/behzadhaki/VariationalMonotonicGrooveTransformer/blob/main/documentation/chapter1_Data/README.md#313-load-gmd-dataset-in-hvo_sequence-format-using-a-single-command---)


[Chapter 2 - Model](https://github.com/behzadhaki/VariationalMonotonicGrooveTransformer/tree/main/documentation/chapter2_Model/README.md)
----

1. [Introduction](https://github.com/behzadhaki/VariationalMonotonicGrooveTransformer/blob/main/documentation/chapter2_Model/README.md#1-introduction-)
2. [Instantiating a Model](https://github.com/behzadhaki/VariationalMonotonicGrooveTransformer/blob/main/documentation/chapter2_Model/README.md#2-instantiating-a-model-)
   1. [BasicGrooveTransformer.GrooveTransformer](https://github.com/behzadhaki/VariationalMonotonicGrooveTransformer/blob/main/documentation/chapter2_Model/README.md#2i-basicgroovetransformergroovetransformer--)
   2. [BasicGrooveTransformer.GrooveTransformerEncoder](https://github.com/behzadhaki/VariationalMonotonicGrooveTransformer/blob/main/documentation/chapter2_Model/README.md#2ii-basicgroovetransformergroovetransformerencoder-)
3. [Storing a Model](https://github.com/behzadhaki/VariationalMonotonicGrooveTransformer/blob/main/documentation/chapter2_Model/README.md#3-storing-a-model-)
4. [Loading a Stored Model](https://github.com/behzadhaki/VariationalMonotonicGrooveTransformer/blob/main/documentation/chapter2_Model/README.md#4-loading-a-stored-model-)
5. [Generation using a Model](https://github.com/behzadhaki/VariationalMonotonicGrooveTransformer/blob/main/documentation/chapter2_Model/README.md#5-generation-using-a-model-)


[Chapter 3 - Evaluation Tools](https://github.com/behzadhaki/VariationalMonotonicGrooveTransformer/tree/main/documentation/chapter3_Evaluator/README.md)
----
1. [Introduction](https://github.com/behzadhaki/VariationalMonotonicGrooveTransformer/tree/main/documentation/chapter3_Evaluator#1-introduction-)
2. **Part A.** [`GrooveEvaluator`](https://github.com/behzadhaki/VariationalMonotonicGrooveTransformer/blob/main/documentation/chapter3_Evaluator/1_grooveevalbasics.md#2-grooveevaluator-basics-)
   + **Part A1**
   1. [Prepapre the data used for Evaluation](https://github.com/behzadhaki/VariationalMonotonicGrooveTransformer/blob/main/documentation/chapter3_Evaluator/1_grooveevalbasics.md#21-prepapre-the-data-used-for-evaluation-)
   2. [Initialization](https://github.com/behzadhaki/VariationalMonotonicGrooveTransformer/blob/main/documentation/chapter3_Evaluator/1_grooveevalbasics.md#22-initialization-)
   3. [Evaluating Predictions](https://github.com/behzadhaki/VariationalMonotonicGrooveTransformer/blob/main/documentation/chapter3_Evaluator/1_grooveevalbasics.md#23-preparing-predictions-)
      1. [Get Ground Truth Samples](https://github.com/behzadhaki/VariationalMonotonicGrooveTransformer/blob/main/documentation/chapter3_Evaluator/1_grooveevalbasics.md#231-get-ground-truth-samples--)
      2. [Pass Samples to Model](https://github.com/behzadhaki/VariationalMonotonicGrooveTransformer/blob/main/documentation/chapter3_Evaluator/1_grooveevalbasics.md#232-pass-samples-to-model-)
      3. [Add Predictions to Evaluator](https://github.com/behzadhaki/VariationalMonotonicGrooveTransformer/blob/main/documentation/chapter3_Evaluator/1_grooveevalbasics.md#233-add-predictions-to-evaluator-)
   4. [Saving and Loading](https://github.com/behzadhaki/VariationalMonotonicGrooveTransformer/blob/main/documentation/chapter3_Evaluator/1_grooveevalbasics.md#24-saving-and-loading-) 
   + **Part A2**
   5. [Accessing Evaluation Results](https://github.com/behzadhaki/VariationalMonotonicGrooveTransformer/blob/main/documentation/chapter3_Evaluator/2_grooveeval_standalone_usage.md#3-accessing-evaluation-results-)
      1. [Results as Dictionaries or Pandas.DataFrame](https://github.com/behzadhaki/VariationalMonotonicGrooveTransformer/blob/main/documentation/chapter3_Evaluator/2_grooveeval_standalone_usage.md#31-results-as-dictionaries-or-pandasdataframe-)
      2. [Rendering Results as Bokeh Plots](https://github.com/behzadhaki/VariationalMonotonicGrooveTransformer/blob/main/documentation/chapter3_Evaluator/2_grooveeval_standalone_usage.md#32-rendering-results-as-bokeh-plots-)
      3. [Rendering Piano Rolls/Audio/Midi](https://github.com/behzadhaki/VariationalMonotonicGrooveTransformer/blob/main/documentation/chapter3_Evaluator/2_grooveeval_standalone_usage.md#33-rendering-piano-rollsaudiomidi-)
   6. [Compiling Plots for Logging](https://github.com/behzadhaki/VariationalMonotonicGrooveTransformer/blob/main/documentation/chapter3_Evaluator/2_grooveeval_standalone_usage.md#4-compiling-logging-media-)
3. **Part B.** [`MultiSetEvaluator`](https://github.com/behzadhaki/VariationalMonotonicGrooveTransformer/blob/main/documentation/chapter3_Evaluator/3_multiseteval_demo.md)
   1. [Prepapre the sets used for cross comparison](https://github.com/behzadhaki/VariationalMonotonicGrooveTransformer/blob/main/documentation/chapter3_Evaluator/3_multiseteval_demo.md#1-prepare-the-sets-used-for-cross-comparison-)
   2. [Initialization](https://github.com/behzadhaki/VariationalMonotonicGrooveTransformer/blob/main/documentation/chapter3_Evaluator/3_multiseteval_demo.md#2-initialization-)
   3. [Saving and Loading](https://github.com/behzadhaki/VariationalMonotonicGrooveTransformer/blob/main/documentation/chapter3_Evaluator/3_multiseteval_demo.md#3-saving-and-loading-)
   4. [Available Analyzers](https://github.com/behzadhaki/VariationalMonotonicGrooveTransformer/blob/main/documentation/chapter3_Evaluator/3_multiseteval_demo.md#4-available-analyzers-)
      1. [Inter-Intra Analysis (raw statistics, distribution plots and KL/OA Plots)](https://github.com/behzadhaki/VariationalMonotonicGrooveTransformer/blob/main/documentation/chapter3_Evaluator/3_multiseteval_demo.md#41-accessing-evaluation-results-)
      2. [Hit, Velocity, Offset Analysis](https://github.com/behzadhaki/VariationalMonotonicGrooveTransformer/blob/main/documentation/chapter3_Evaluator/3_multiseteval_demo.md#42-hit-velocity-offset-analysis-)
   5. [Compiling Results](#2_vi)

[Chapter 4 - WANDB](https://github.com/behzadhaki/VariationalMonotonicGrooveTransformer/tree/main/documentation/chapter4_WANDB/README.md)
----

[Chapter 5 - Training](https://github.com/behzadhaki/VariationalMonotonicGrooveTransformer/tree/main/documentation/chapter5_Training/README.md)
----

