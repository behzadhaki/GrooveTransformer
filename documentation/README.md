
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


[Chapter 3 - Evaluator](https://github.com/behzadhaki/VariationalMonotonicGrooveTransformer/tree/main/documentation/chapter3_Evaluator/README.md)
----
1. [Introduction](#1)
2. [GrooveEvaluator Basics](https://github.com/behzadhaki/VariationalMonotonicGrooveTransformer/blob/main/documentation/chapter3_Evaluator/1_basics.md#2-grooveevaluator-basics-)
   1. [Prepapre the data used for Evaluation](https://github.com/behzadhaki/VariationalMonotonicGrooveTransformer/blob/main/documentation/chapter3_Evaluator/1_basics.md#21-prepapre-the-data-used-for-evaluation-)
   2. [Initialization](https://github.com/behzadhaki/VariationalMonotonicGrooveTransformer/blob/main/documentation/chapter3_Evaluator/1_basics.md#22-initialization-)
   3. [Evaluating Predictions](https://github.com/behzadhaki/VariationalMonotonicGrooveTransformer/blob/main/documentation/chapter3_Evaluator/1_basics.md#23-preparing-predictions-)
      1. [Get Ground Truth Samples](https://github.com/behzadhaki/VariationalMonotonicGrooveTransformer/blob/main/documentation/chapter3_Evaluator/1_basics.md#231-get-ground-truth-samples--)
      2. [Pass Samples to Model](https://github.com/behzadhaki/VariationalMonotonicGrooveTransformer/blob/main/documentation/chapter3_Evaluator/1_basics.md#232-pass-samples-to-model-)
      3. [Add Predictions to Evaluator](https://github.com/behzadhaki/VariationalMonotonicGrooveTransformer/blob/main/documentation/chapter3_Evaluator/1_basics.md#233-add-predictions-to-evaluator-)
   4. [Saving and Loading](https://github.com/behzadhaki/VariationalMonotonicGrooveTransformer/blob/main/documentation/chapter3_Evaluator/1_basics.md#24-saving-and-loading-) 

[Chapter 4 - WANDB](https://github.com/behzadhaki/VariationalMonotonicGrooveTransformer/tree/main/documentation/chapter4_WANDB/README.md)
----

[Chapter 5 - Training](https://github.com/behzadhaki/VariationalMonotonicGrooveTransformer/tree/main/documentation/chapter5_Training/README.md)
----

#### Test 1

<embed type="text/html" src="https://github.com/behzadhaki/VariationalMonotonicGrooveTransformer/blob/main/testers/evaluator/misc/offset_plots.html" width="600" height="400"></embed>

#### Test 2

<embed type="text/html" src="offset_plots.html" width="600" height="400"></embed>


#### Test 3


<embed type="text/html" src="../../testers/evaluator/misc/offset_plots.html" width="600" height="400"></embed>


<div id="code-element">
<iframe src="offset_plots.html"
    sandbox="allow-same-origin allow-scripts"
    width="100%"
    height="500"
    scrolling="no"
    seamless="seamless"
    frameborder="0">
</iframe></div>

<iframe src="offset_plots.html"
    sandbox="allow-same-origin allow-scripts"
    width="100%"
    height="500"
    scrolling="no"
    seamless="seamless"
    frameborder="0">
</iframe>