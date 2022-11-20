## Chapter 5 - Weight and Biases (W&B) for Experiment Tracking  



# Table of Contents
1. [Introduction](#1)
2. [Basic Usage](#2)
   1. [Logging in](#2_1)
   2. [Hyperparameter Tracking](#2_2)
   3. [Metric Tracking](#2_3)
   4. [Model Tracking](#2_4)
   5. [Code Tracking](#2_5)
   6. [Artifacts Tracking](#2_6)
3. [Hyperparameter Tuning Using Sweeps](#3)
   1. [Formatting the training script](#3_1)
   2. [Sweep Configuration](#3_2)
   3. [Sweep Execution](#3_3)
   4. [Sweep Results](#3_4)


## 1. Introduction <a name="1"></a>

[Weight and Biases](https://wandb.ai/site) is a tool for experiment tracking and visualization. 
It is a great tool for tracking your experiments and comparing them. 
It is also a great tool for hyperparameter tuning. It is a free tool for open-source projects. 
For more information, please visit the [official website](https://wandb.ai/site).

## 2. Basic Usage <a name="2"></a>

Below is a simple example of how to use W&B for experiment tracking.


```python
import wandb
import torch
# Rest of the imports

hparams = {
    "lr": 0.001,
    "batch_size": 32,
    "epochs": 10,
    "optimizer": "Adam",
    "loss": "CrossEntropyLoss"
    # ...
}

# Select the device to train on 
hparams["device"] = "cuda" if torch.cuda.is_available() else "cpu",

if __name__ == "__main__":
    # Initialize W&B
    wandb.init(
        config=hparams,                         # either from config file or CLI specified hyperparameters
        project=hparams["wandb_project"],          # name of the project
        anonymous="allow",
        entity="mmil_vae_g2d",                          # 
        settings=wandb.Settings(code_dir="train.py")    # for code saving
    )
    
    config = wandb.config           # VERY IMPORTANT: from now on, use config instead of hparams.
                                    #                 This is because we may be using wandb sweep.
                                    #                 SEE SECTION 3 FOR MORE DETAILS.

    # Load Training and Test Data

    # Create Model
    model = model_class(config)

    
```