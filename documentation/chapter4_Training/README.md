## Chapter 4 - A Guide to Training  


# Table of Contents
1. [Introduction](#1)
2. [Hyper-parameters](#2)
3. [Command Line Interface (CLI)](#3)
4. [Training Structure](#4)

## 1. Introduction <a name="1"></a>

Any model that you develop consists of two sets of parameters:

1. The model parameters, which are the weights of the model. 
    These are the parameters that are updated during training using the gradients.
2. The hyper-parameters, which are the parameters that are not updated during training. 
    These are the parameters that are used to control the training process and the model architecture.

In addition to the above model related parameters, there are also some settings that are used to 
control the training process. For example, the number of epochs, the dataset to use, whether to save
the model, etc.

In this chapter, we will provide a template for training a model.


## 2. Hyper-parameters <a name="2"></a>

Hyperparameters are the parameters that are not updated using the gradients. 
This means that they are not learned during training. These parameters are used to control (1) the training process
(the learning rate, the batch size, the optimizer, the loss function, etc.) and (2) the model architecture 
(the number of layers, model dimensionality, etc.). To find the best set of hyperparameters, 
we can use a hyperparameter tuning tool or we can use parameters that are known to work well.

We will define these hyperparameters in a dictionary. For example:

```python
hparams = {
    "latent_dim": 128,
    "n_layers": 3,
    "lr": 0.001,
    "batch_size": 32,
    "epochs": 10,
    "optimizer": "Adam",
    "loss": "CrossEntropyLoss"
    # ...
}
```

## 3. Command Line Interface (CLI) <a name="3"></a>

You should develop your code such that it can be run from the command line. This means that you should
avoid hard-coding any parameters, settings, paths, etc. Instead, 
you should use the `argparse` module to parse the command line arguments.

For example, you can use the following code to parse the command line arguments:

```python
import argparse

# Create the parser
parser = argparse.ArgumentParser()

# Add arguments
parser.add_argument("--d_model_enc", help="Dimension of the encoder model", default=32)

# Parse the arguments
args = parser.parse_args()


# compile into a hparams dictionary
hparams = dict(
        d_model_enc=args.d_model_enc,
        d_model_dec=args.d_model_dec,
        # ...
        dropout=args.dropout,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        optimizer=args.optimizer,
        # ...
        dataset_path=args.dataset_path,
        dataset_json_fname=args.dataset_json_fname,
    )
```

> **Warning** One parameter that you would set dynamically within the script is the `device`. This is to ensure that 
> the model is trainable both on the CPU and/or GPU if it is available. You can use the following code to set the device:
> ```python
> hparams.keys( device="cuda" if torch.cuda.is_available() else "cpu")
> ```


## 4. Training Structure <a name="4"></a>

During training, you want to perform the following steps:
    
1. Load the dataset
2. Create the model
3. Create the optimizer
4. Create the loss function
5. For each epoch:
    1. For each batch of training data:
       1. Move the batch in/out data to the device if necessary
       2. Forward pass the input
       3. Compute the loss 
       4. Backward pass 
       5. Update the model parameters
       6. Track metrics (loss, accuracy, etc.)
    2. Compute the average of each metric and log it 
    3. Evaluate model on the validation set if necessary (evaluate also in batches if using GPU)
    5. Save the model if necessary


```python
from torch.utils.data import DataLoader

if __name__ == "__main__":
    # ----------------------------------------------------------------------------------------------------------
    # Load Training and Testing Datasets and Wrap them in torch.utils.data.Dataloader
    # ----------------------------------------------------------------------------------------------------------
    train_dataset = DatasetClass(...)
    train_dataloader = DataLoader(train_dataset, batch_size=hparams["batch_size"], shuffle=True)
    tests_dataset = DatasetClass(...)
    tests_dataloader = DataLoader(tests_dataset, batch_size=hparams["batch_size"], shuffle=True)
    
    # ----------------------------------------------------------------------------------------------------------
    # Initialize the model
    # ----------------------------------------------------------------------------------------------------------
    model_instance = ModelClass(hparams)
    
    # ----------------------------------------------------------------------------------------------------------
    # Instantiate the loss Criterion and Optimizer
    # ----------------------------------------------------------------------------------------------------------
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model_instance.parameters(), lr=hparams["lr"])
    
    # ------------------------------------------------------------------------------------------------------------
    # Iterate over epochs
    # ------------------------------------------------------------------------------------------------------------
    metrics = dict()
    
    for epoch in range(hparams["configs"]):
        
        # Run the training loop (trains per batch internally)
        model_instance.train()
        
        # Prepare the metric trackers for the new epoch
        train_loss = []
        train_accuracy = []
        
        # ------------------------------------------------------------------------------------------
        # Iterate over batches
        # ------------------------------------------------------------------------------------------
        for batch_count, (inputs, outputs_tg) in enumerate(train_dataloader):

            # Move data to GPU if available
            inputs = inputs.to(device)
            outputs = outputs.to(device)
            
            # Forward pass
            outputs_pred = model_instance.forward(inputs)
            
            # Compute the loss
            loss = criterion(outputs_pred, outputs)
            
            # Backward pass using only the gradients  correponding to current 
            optimizer.zero_grad()
            loss.backward()
            
            # Update the model parameters
            optimizer.step()
            
            # Track metrics
            train_loss.append(loss.item())
            train_accuracy.append(calculate_accuracy(outputs_pred, outputs))
            
        # Compute the average of the metrics
        metrics["train/loss"] = np.mean(train_loss)
        metrics["train/accuracy"] = np.mean(train_accuracy)
        
        # ------------------------------------------------------------------------------------------
        # Evaluate the model on the test set
        # ------------------------------------------------------------------------------------------

        # Put the model in evaluation mode
        model_instance.eval()
        with torch.no_grad():

            # Prepare the metric trackers for the new epoch
            test_loss = []
            test_accuracy = []
            
            # Iterate over batches
            for batch_count, (inputs, outputs_tgt, indices) in enumerate(tests_dataloader):
                
                # Move data to GPU if available
                # ---------------------------------------------------------------------------------------
                inputs = inputs.to(device)
                outputs = outputs.to(device)
                
                # Forward pass
                # ---------------------------------------------------------------------------------------
                outputs_pred = model_instance.forward(inputs)
                
                # Compute the loss
                # ---------------------------------------------------------------------------------------
                loss = criterion(outputs_pred, outputs)
                
                # Track metrics
                # ---------------------------------------------------------------------------------------
                test_loss.append(loss.item())
                test_accuracy.append(calculate_accuracy(outputs_pred, outputs))

            # ------------------------------------------------------------------------------------------------                
            # Compute the average of the metrics
            # ---------------------------------------------------------------------------------------------
            metrics["test/loss"] = np.mean(test_loss)
            metrics["test/accuracy"] = np.mean(test_accuracy)
        
        # ----------------------------------------------------------------------------------------------------------
        # Log/Print/Store the metrics
        # ----------------------------------------------------------------------------------------------------------
        print(f"Epoch {epoch}: {metrics}")

        # ----------------------------------------------------------------------------------------------------------
        # Save the model and the parameters in a json file
        # ----------------------------------------------------------------------------------------------------------
        torch.save(model_instance.state_dict(), f"model_{epoch}.pt")
        with open(f"model_parameters_{epoch}.json", "w") as f:
            json.dump(hparams, f)
```


The above code is a simple example of how to train a model. You can use it as a template for your own training script.
That being said, you can make the code more modular by creating a `batch_looper` function that takes as input the
dataloader, the model, the loss function, and the device, the optimizer (optionally, if training). 
This function will then iterate over the batches and perform the forward and optionally backward passes. 
The `batch_looper` function will return the average of the metrics. You can
then use wrap function in two functions: `train` and `test` that will call the `batch_looper` function and return the
average of the metrics. You can then use these functions in the main training loop.

```python
def batch_looper(dataloader_, model_, loss_criterion_, device_, optimizer=None):
    # Prepare the metric trackers for the new epoch
        train_loss = []
        train_accuracy = []
        
        # ------------------------------------------------------------------------------------------
        # Iterate over batches
        # ------------------------------------------------------------------------------------------
        for batch_count, (inputs, outputs_tg) in enumerate(dataloader_):

            # Move data to GPU if available
            inputs = inputs.to(device_)
            outputs = outputs.to(device_)
            
            # Forward pass
            outputs_pred = model_.forward(inputs)
            
            # Compute the loss
            loss = loss_criterion_(outputs_pred, outputs)
            
            if optimizer is not None:
                # Backward pass using only the gradients  correponding to current 
                optimizer.zero_grad()
                loss.backward()
                
                # Update the model parameters
                optimizer.step()
            
            # Track metrics
            train_loss.append(loss.item())
            train_accuracy.append(calculate_accuracy(outputs_pred, outputs))
            
        # Compute the average of the metrics
        metrics["loss"] = np.mean(train_loss)
        metrics["accuracy"] = np.mean(train_accuracy)
    
    return metrics

def train_batch_looper(dataloader_, model, loss_fn_, optimizer):
    model.train()
    return batch_looper(dataloader_, model, loss_fn_, optimizer)

def test_batch_looper(dataloader_, model, loss_fn_):
    model.eval()
    with torch.no_grad():
        return batch_looper(dataloader_, model, loss_fn_, optimizer=None)
```

With this, you can simplify the main training script as follows:

```python

from torch.utils.data import DataLoader

if __name__ == "__main__":
    # ----------------------------------------------------------------------------------------------------------
    # Load Training and Testing Datasets and Wrap them in torch.utils.data.Dataloader
    # ----------------------------------------------------------------------------------------------------------
    train_dataset = DatasetClass(...)
    train_dataloader = DataLoader(train_dataset, batch_size=hparams["batch_size"], shuffle=True)
    tests_dataset = DatasetClass(...)
    tests_dataloader = DataLoader(tests_dataset, batch_size=hparams["batch_size"], shuffle=True)
    
    # ----------------------------------------------------------------------------------------------------------
    # Initialize the model
    # ----------------------------------------------------------------------------------------------------------
    model_instance = ModelClass(hparams)
    
    # ----------------------------------------------------------------------------------------------------------
    # Instantiate the loss Criterion and Optimizer
    # ----------------------------------------------------------------------------------------------------------
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model_instance.parameters(), lr=hparams["lr"])
    
    # ------------------------------------------------------------------------------------------------------------
    # Iterate over epochs
    # ------------------------------------------------------------------------------------------------------------
    metrics = dict()
    
    for epoch in range(hparams["configs"]):
        
        # Run the training loop (trains per batch internally)
        metrics_train = train_batch_looper(train_dataloader, model_instance, criterion, optimizer)
        metric.update({"train/" + k: v for k, v in metrics_train.items()})
        
        # ------------------------------------------------------------------------------------------
        # Evaluate the model on the test set
        # ------------------------------------------------------------------------------------------
        metrics_test = test_batch_looper(test_dataloader, model_instance, criterion)
        metrics.update({"test/" + k: v for k, v in metrics_test.items()})
        
        # ----------------------------------------------------------------------------------------------------------
        # Log/Print/Store the metrics
        # ----------------------------------------------------------------------------------------------------------
        print(f"Epoch {epoch}: {metrics}")

        # ----------------------------------------------------------------------------------------------------------
        # Save the model and the parameters in a json file
        # ----------------------------------------------------------------------------------------------------------
        torch.save(model_instance.state_dict(), f"model_{epoch}.pt")
        with open(f"model_parameters_{epoch}.json", "w") as f:
            json.dump(hparams, f)
```



