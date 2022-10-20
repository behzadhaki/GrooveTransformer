# Chapter 3 - Evaluation Tools (Part B - Accessing Evaluation Results)

-----

# Table of Contents
3. [Accessing Evaluation Results](#3)
   1. [Results as Dictionaries or Pandas.DataFrame](#3_i)
   2. [Rendering Results as Bokeh Plots](#3_ii)
   3. [Rendering Piano Rolls/Audio/Midi](#3_ii)
   

   
## 3. Accessing Evaluation Results <a name="3"></a>

---

The evaluation results can be accesssed in multiple ways. 
The results can be accessed as a dictionary or a Pandas.DataFrame (Section [3.1](#3_i)), 
or they can be rendered as Bokeh plots (Section [3.2](#3_ii)). 
Also, the ground truth samples and the generations can be rendered to piano roll, audio or midi files (Section [3.3](#3_iii)). 

### 3.1. Results as Dictionaries or Pandas.DataFrame <a name="3_i"></a>

The evaluation results can be accessed as a dictionary or a Pandas.DataFrame. 
The following numerical results are automatically computed and compiled into a dictionary or pandas dataframe:


   1. [Quality of Hits](a): Hit counts for gt and pred samples, as well as, cross comparison of hits (accuracy, PPV, ...)
   2. [Quality of Velocities](b): Velocity distributions for gt and pred samples
   3. [Quality of Offsets](c): Offset distributions for gt and pred samples
   4. [Rhythmic Distances](d): Distance of pred samples from gt samples using the rhythm distance metrics implemented in HVO_Sequence (l1, l2, cosine, hamming, ...)
   5. [Global features](e): Global features of the gt and pred samples, these features are the features implemented in HVO_Sequence (NoI, Midness, ...)

These results can be accessed as raw data, that is, per sample values extracted from the evaluation, 
or as aggregated statistics, that is, the mean and standard deviation of the per sample values. 

### Quality of Hits <a name="a"></a>

### Quality of Velocities <a name="b"></a>

### Quality of Offsets <a name="c"></a>

### Rhythmic Distances <a name="d"></a>

### Global features <a name="e"></a>





### 3.2 Rendering Results as Bokeh Plots <a name="3_ii"></a>

The results in section [3.1](#3_i) can also be automatically rendered as Bokeh plots. 
These plots are violin plots, super-imposed with boxplots and the raw scatter data. The plots are separated by Tabs 
for each set of analysis results.

### Quality of Hits <a name="a2"></a>

### Quality of Velocities <a name="b2"></a>

### Quality of Offsets <a name="c2"></a>

### Rhythmic Distances <a name="d2"></a>

### Global features <a name="e2"></a>


### 3.3 Rendering Piano Rolls/Audio/Midi <a name="3_iii"></a>


### Piano Rolls <a name="a3"></a>


### Audio <a name="b3"></a>


### Midi <a name="c3"></a>


