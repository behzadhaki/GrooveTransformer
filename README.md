
# List of Contents
### Guides for using the repository can be found here

# [Chapter 0]() 
### [A. ENVIRONMENT SETUP](https://github.com/behzadhaki/GrooveTransformer/blob/main/documentation/Chapter%200_PyEnvAndCluster/README.md)
1. [Required Packages](https://github.com/behzadhaki/GrooveTransformer/blob/main/documentation/Chapter%200_PyEnvAndCluster/README.md#1-required-packages-)
2. [CPU Installation](https://github.com/behzadhaki/GrooveTransformer/blob/main/documentation/Chapter%200_PyEnvAndCluster/README.md#2-cpu-installation--)
   1. [venv Installation](https://github.com/behzadhaki/GrooveTransformer/blob/main/documentation/Chapter%200_PyEnvAndCluster/README.md#21--venv-installation-)
   2. [Anaconda Installation](https://github.com/behzadhaki/GrooveTransformer/blob/main/documentation/Chapter%200_PyEnvAndCluster/README.md#22-anaconda-installation-)
3. [GPU Installation](https://github.com/behzadhaki/GrooveTransformer/blob/main/documentation/Chapter%200_PyEnvAndCluster/README.md#3-gpu-installation-)
   1. [Local GPU Installation](https://github.com/behzadhaki/GrooveTransformer/blob/main/documentation/Chapter%200_PyEnvAndCluster/README.md#31-gpu-installation-on-local-machines-)
   2. [HPC Cluster GPU Installation](https://github.com/behzadhaki/GrooveTransformer/blob/main/documentation/Chapter%200_PyEnvAndCluster/README.md#32-installation-on-hpc-clusters-)
      1. [Anaconda Installation](https://github.com/behzadhaki/GrooveTransformer/blob/main/documentation/Chapter%200_PyEnvAndCluster/README.md#321-anaconda)
 
### [B. HPC Cluster Guide](https://github.com/behzadhaki/GrooveTransformer/blob/main/documentation/Chapter%200_PyEnvAndCluster/HPC_Cluster_Guide.md)
1. [Getting Started](https://github.com/behzadhaki/GrooveTransformer/blob/main/documentation/Chapter%200_PyEnvAndCluster/HPC_Cluster_Guide.md#1-getting-started-)
   1. [Accounts](https://github.com/behzadhaki/GrooveTransformer/blob/main/documentation/Chapter%200_PyEnvAndCluster/HPC_Cluster_Guide.md#11-accounts-)
   2. [Accessing the Clusters](https://github.com/behzadhaki/GrooveTransformer/blob/main/documentation/Chapter%200_PyEnvAndCluster/HPC_Cluster_Guide.md#12-accessing-the-clusters-)
2. [Using the Clusters](https://github.com/behzadhaki/GrooveTransformer/blob/main/documentation/Chapter%200_PyEnvAndCluster/HPC_Cluster_Guide.md#2-using-the-clusters-)
    1. [Resources](https://github.com/behzadhaki/GrooveTransformer/blob/main/documentation/Chapter%200_PyEnvAndCluster/HPC_Cluster_Guide.md#21-resources-)
    2. [Interactive Sessions](https://github.com/behzadhaki/GrooveTransformer/blob/main/documentation/Chapter%200_PyEnvAndCluster/HPC_Cluster_Guide.md#22-interactive-sessions-)
    3. [Submitting Jobs](https://github.com/behzadhaki/GrooveTransformer/blob/main/documentation/Chapter%200_PyEnvAndCluster/HPC_Cluster_Guide.md#23-submitting-jobs-)
    4. [Available Software](https://github.com/behzadhaki/GrooveTransformer/blob/main/documentation/Chapter%200_PyEnvAndCluster/HPC_Cluster_Guide.md#24-available-software-)
    5. [Monitoring or Cancelling Jobs](https://github.com/behzadhaki/GrooveTransformer/blob/main/documentation/Chapter%200_PyEnvAndCluster/HPC_Cluster_Guide.md#25-monitoring-or-canceling-jobs-)

# [Chapter 1]() 
### [A. Data Representation & Datasets](https://github.com/behzadhaki/GrooveTransformer/tree/main/documentation/chapter1A_Data/README.md) 

1. [Introduction](https://github.com/behzadhaki/GrooveTransformer/blob/main/documentation/chapter1A_Data/README.md#1-introduction-)
2. [Data Representation](https://github.com/behzadhaki/GrooveTransformer/blob/main/documentation/chapter1A_Data/README.md#2-data-representation-)
   1. [`HVO_Sequence`](https://github.com/behzadhaki/GrooveTransformer/blob/main/documentation/chapter1A_Data/README.md#21-hvo_sequence-)
   2. [Example Code](https://github.com/behzadhaki/GrooveTransformer/blob/main/documentation/chapter1A_Data/README.md#22-example-code-)
3. [Datasets](https://github.com/behzadhaki/GrooveTransformer/blob/main/documentation/chapter1A_Data/README.md#3-datasets-)
   1. [Groove Midi Dataset](https://github.com/behzadhaki/GrooveTransformer/blob/main/documentation/chapter1A_Data/README.md#31-groove-midi-dataset-)
      1. [Load dataset as a dictionary](https://github.com/behzadhaki/GrooveTransformer/blob/main/documentation/chapter1A_Data/README.md#311-load-dataset-as-a-dictionary-)
      2. [Extract `HVO_Sequence` objects from dataset dictionaries](https://github.com/behzadhaki/GrooveTransformer/blob/main/documentation/chapter1A_Data/README.md#312-extract-hvo_sequence-objects-from-dataset-dictionaries-above--)
      3. [**_Load GMD Dataset in `HVO_Sequence` format using a single command !!!_**](https://github.com/behzadhaki/GrooveTransformer/blob/main/documentation/chapter1A_Data/README.md#313-load-gmd-dataset-in-hvo_sequence-format-using-a-single-command---)


### [B. HVO Sequence](https://github.com/behzadhaki/GrooveTransformer/tree/main/documentation/Chapter1B_HVO_Sequence#chapter-1b---hvo-sequence-a-grid-relative-piano-roll-representation-for-drum-sequences)


1. [Basic Attributes](https://github.com/behzadhaki/GrooveTransformer/tree/main/documentation/Chapter1B_HVO_Sequence#1-basic-attributes-)
   1. [Beat Division Factors](https://github.com/behzadhaki/GrooveTransformer/tree/main/documentation/Chapter1B_HVO_Sequence#11-beat-division-factors-)
   2. [Drum Mapping](https://github.com/behzadhaki/GrooveTransformer/tree/main/documentation/Chapter1B_HVO_Sequence#12-drum-mapping-)
   3. [Grid Attributes](https://github.com/behzadhaki/GrooveTransformer/tree/main/documentation/Chapter1B_HVO_Sequence#13-grid-attributes-)
   4. [Metadata](https://github.com/behzadhaki/GrooveTransformer/tree/main/documentation/Chapter1B_HVO_Sequence#14-metadata-)
   5. [HVO: Piano-roll Score](https://github.com/behzadhaki/GrooveTransformer/tree/main/documentation/Chapter1B_HVO_Sequence#15-hvo_sequence-)
2. [Simple Usage](https://github.com/behzadhaki/GrooveTransformer/tree/main/documentation/Chapter1B_HVO_Sequence#2-simple-usage-)
3. [Built-in Tools](https://github.com/behzadhaki/GrooveTransformer/tree/main/documentation/Chapter1B_HVO_Sequence#32-built-in-tools-)
4. [Multi-Segment Scores](https://github.com/behzadhaki/GrooveTransformer/tree/main/documentation/Chapter1B_HVO_Sequence#4-multi-segment-hvo-sequences-)

[Chapter 2 - Models](https://github.com/behzadhaki/GrooveTransformer/tree/main/documentation/chapter2_Model/A_MonotonicGrooveTransformer/README.md)
----
### [A. Groove Transformer](https://github.com/behzadhaki/GrooveTransformer/tree/main/documentation/chapter2_Model/A_MonotonicGrooveTransformer)
1. [Introduction](https://github.com/behzadhaki/GrooveTransformer/blob/main/documentation/chapter2_Model/A_MonotonicGrooveTransformer/README.md#1-introduction-)
2. [Instantiating a Model](https://github.com/behzadhaki/GrooveTransformer/blob/main/documentation/chapter2_Model/A_MonotonicGrooveTransformer/README.md#2-instantiating-a-model-)
   1. [BasicGrooveTransformer.GrooveTransformer](https://github.com/behzadhaki/GrooveTransformer/blob/main/documentation/chapter2_Model/A_MonotonicGrooveTransformer/README.md#2i-basicgroovetransformergroovetransformer--)
   2. [BasicGrooveTransformer.GrooveTransformerEncoder](https://github.com/behzadhaki/GrooveTransformer/blob/main/documentation/chapter2_Model/A_MonotonicGrooveTransformer/README.md#2ii-basicgroovetransformergroovetransformerencoder-)
3. [Storing a Model](https://github.com/behzadhaki/GrooveTransformer/blob/main/documentation/chapter2_Model/A_MonotonicGrooveTransformer/README.md#3-storing-a-model-)
4. [Loading a Stored Model](https://github.com/behzadhaki/GrooveTransformer/blob/main/documentation/chapter2_Model/A_MonotonicGrooveTransformer/README.md#4-loading-a-stored-model-)
5. [Generation using a Model](https://github.com/behzadhaki/GrooveTransformer/blob/main/documentation/chapter2_Model/A_MonotonicGrooveTransformer/README.md#5-generation-using-a-model-)

### [B. Variational Groove transformer](https://github.com/behzadhaki/GrooveTransformer/tree/main/documentation/chapter2_Model/B_VariationalMonotonicGrooveTransformer)
1. [Introduction](https://github.com/behzadhaki/GrooveTransformer/tree/main/documentation/chapter2_Model/B_VariationalMonotonicGrooveTransformer#1-introduction-) 
2. [Model Description](https://github.com/behzadhaki/GrooveTransformer/tree/main/documentation/chapter2_Model/B_VariationalMonotonicGrooveTransformer#2-model-description-)
   1. [Network Architecture](https://github.com/behzadhaki/GrooveTransformer/tree/main/documentation/chapter2_Model/B_VariationalMonotonicGrooveTransformer#2i-network-architecture-)
   2. [loss functions](https://github.com/behzadhaki/GrooveTransformer/tree/main/documentation/chapter2_Model/B_VariationalMonotonicGrooveTransformer#2ii-loss-functions-)
   2. [Training Parameters](https://github.com/behzadhaki/GrooveTransformer/tree/main/documentation/chapter2_Model/B_VariationalMonotonicGrooveTransformer#2iii-training-parameters-)
3. [MonotonicGrooveVAE.GrooveTransformerEncoderVAE](https://github.com/behzadhaki/GrooveTransformer/tree/main/documentation/chapter2_Model/B_VariationalMonotonicGrooveTransformer#3i-instantiation-)
   1. [Instantiation](https://github.com/behzadhaki/GrooveTransformer/tree/main/documentation/chapter2_Model/B_VariationalMonotonicGrooveTransformer#3i-instantiation-)
   2. [Storing](https://github.com/behzadhaki/GrooveTransformer/tree/main/documentation/chapter2_Model/B_VariationalMonotonicGrooveTransformer#3ii-storing-)
   3. [Loading](https://github.com/behzadhaki/GrooveTransformer/tree/main/documentation/chapter2_Model/B_VariationalMonotonicGrooveTransformer#3iii-loading-)
   4. [Pretrained Versions](https://github.com/behzadhaki/GrooveTransformer/tree/main/documentation/chapter2_Model/B_VariationalMonotonicGrooveTransformer#3iv-pretrained-versions-)
   5. [Generation](https://github.com/behzadhaki/GrooveTransformer/tree/main/documentation/chapter2_Model/B_VariationalMonotonicGrooveTransformer#3v-generation-)

[Chapter 3 - Evaluation Tools](https://github.com/behzadhaki/GrooveTransformer/tree/main/documentation/chapter3_Evaluator/README.md)
----

### [A. `GrooveEvaluator`](https://github.com/behzadhaki/GrooveTransformer/blob/main/documentation/chapter3_Evaluator/1_grooveevalbasics.md#2-grooveevaluator-basics-)
   #### **Part A1**
   1. [Prepapre the data used for Evaluation](https://github.com/behzadhaki/GrooveTransformer/blob/main/documentation/chapter3_Evaluator/1_grooveevalbasics.md#21-prepapre-the-data-used-for-evaluation-)
   2. [Initialization](https://github.com/behzadhaki/GrooveTransformer/blob/main/documentation/chapter3_Evaluator/1_grooveevalbasics.md#22-initialization-)
   3. [Preparing Predictions](https://github.com/behzadhaki/GrooveTransformer/blob/main/documentation/chapter3_Evaluator/1_grooveevalbasics.md#23-preparing-predictions-)
      1. [Get Ground Truth Samples](https://github.com/behzadhaki/GrooveTransformer/blob/main/documentation/chapter3_Evaluator/1_grooveevalbasics.md#231-get-ground-truth-samples--)
      2. [Pass Samples to Model](https://github.com/behzadhaki/GrooveTransformer/blob/main/documentation/chapter3_Evaluator/1_grooveevalbasics.md#232-pass-samples-to-model-)
      3. [Add Predictions to Evaluator](https://github.com/behzadhaki/GrooveTransformer/blob/main/documentation/chapter3_Evaluator/1_grooveevalbasics.md#233-add-predictions-to-evaluator-)
   4. [Saving and Loading](https://github.com/behzadhaki/GrooveTransformer/blob/main/documentation/chapter3_Evaluator/1_grooveevalbasics.md#24-saving-and-loading-) 
   #### **Part A2**
   5. [Accessing Evaluation Results](https://github.com/behzadhaki/GrooveTransformer/blob/main/documentation/chapter3_Evaluator/2_grooveeval_standalone_usage.md#3-accessing-evaluation-results-)
      1. [Results as Dictionaries or Pandas.DataFrame](https://github.com/behzadhaki/GrooveTransformer/blob/main/documentation/chapter3_Evaluator/2_grooveeval_standalone_usage.md#3_i)
      2. [Rendering Results as Bokeh Plots](https://github.com/behzadhaki/GrooveTransformer/blob/main/documentation/chapter3_Evaluator/2_grooveeval_standalone_usage.md#52-rendering-results-as-bokeh-plots-)
      3. [Rendering Piano Rolls/Audio/Midi](https://github.com/behzadhaki/GrooveTransformer/blob/main/documentation/chapter3_Evaluator/2_grooveeval_standalone_usage.md#53-rendering-piano-rollsaudiomidi-)
   6. [Compiling Plots for Logging](https://github.com/behzadhaki/GrooveTransformer/blob/main/documentation/chapter3_Evaluator/2_grooveeval_standalone_usage.md#6-compiling-logging-media-)
### [B. `MultiSetEvaluator`](https://github.com/behzadhaki/GrooveTransformer/blob/main/documentation/chapter3_Evaluator/3_multiseteval_demo.md)
   1. [Prepapre the sets used for cross comparison](https://github.com/behzadhaki/GrooveTransformer/blob/main/documentation/chapter3_Evaluator/3_multiseteval_demo.md#1-prepare-the-sets-used-for-cross-comparison-)
   2. [Initialization](https://github.com/behzadhaki/GrooveTransformer/blob/main/documentation/chapter3_Evaluator/3_multiseteval_demo.md#2-initialization-)
   3. [Saving and Loading](https://github.com/behzadhaki/GrooveTransformer/blob/main/documentation/chapter3_Evaluator/3_multiseteval_demo.md#3-saving-and-loading-)
   4. [Available Analyzers](https://github.com/behzadhaki/GrooveTransformer/blob/main/documentation/chapter3_Evaluator/3_multiseteval_demo.md#4-available-analyzers-)
      1. [Inter-Intra Analysis (raw statistics, distribution plots and KL/OA Plots)](https://github.com/behzadhaki/GrooveTransformer/blob/main/documentation/chapter3_Evaluator/3_multiseteval_demo.md#41-accessing-evaluation-results-)
      2. [Hit, Velocity, Offset Analysis](https://github.com/behzadhaki/GrooveTransformer/blob/main/documentation/chapter3_Evaluator/3_multiseteval_demo.md#42-hit-velocity-offset-analysis-)
   5. [Compiling Results](https://github.com/behzadhaki/GrooveTransformer/blob/main/documentation/chapter3_Evaluator/3_multiseteval_demo.md#5-compiling-results-)

[Chapter 4 - WANDB](https://github.com/behzadhaki/GrooveTransformer/tree/main/documentation/chapter4_WANDB/README.md)
----

[Chapter 5 - Training](https://github.com/behzadhaki/GrooveTransformer/tree/main/documentation/chapter5_Training/README.md)
----

