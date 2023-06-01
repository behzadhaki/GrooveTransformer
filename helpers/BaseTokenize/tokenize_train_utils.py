
import os
import torch
#from torchmetrics import Accuracy
import wandb
import re
import numpy as np
from model.Base.BasicGrooveTransformer import GrooveTransformerEncoder, GrooveTransformer

from logging import getLogger
logger = getLogger("VAE_LOSS_CALCULATOR")
logger.setLevel("DEBUG")

def calculate_token_loss(token_logits, token_targets):

    loss_fn = torch.nn.CrossEntropyLoss()
    token_loss = loss_fn(token_logits.view(-1, token_logits.shape[-1]), token_targets.view(-1).long())
    return token_loss

def calculate_hit_loss(hit_logits, hit_targets):
    """
    Calculate the hit loss for the given hit logits and hit targets.
    The loss is calculated either using BCE or Dice loss function.
    :param hit_logits:  (torch.Tensor)  predicted output of the model (**Pre-ACTIVATION**)
    :param hit_targets:     (torch.Tensor)  target output of the model
    :param hit_loss_function:     (torch.nn.BCEWithLogitsLoss)
    :return:    hit_loss (batch, time_steps, n_voices)  the hit loss value per each step and voice (unreduced)
    """
    loss_fn = torch.nn.BCEWithLogitsLoss()
    loss_h = loss_fn(hit_logits, hit_targets)           # batch, time steps, voices
    return loss_h       # batch_size,  time_steps, n_voices


def calculate_velocity_loss(vel_logits, vel_targets):
    """
    Calculate the velocity loss for the velocity targets and the **Pre-Activation** output of the model.
    The loss is calculated using either MSE or BCE loss function.
    :param vel_logits:  (torch.Tensor)  predicted output of the model (**Pre-ACTIVATION**)
    :param vel_targets:     (torch.Tensor)  target output of the model
    :param vel_loss_function:     (str)  either "mse" or "bce"
    :return:    vel_loss (batch_size, time_steps, n_voices)  the velocity loss value per each step and voice (unreduced)
    """
    loss_fn = torch.nn.BCEWithLogitsLoss()
    loss_v = loss_fn(vel_logits, vel_targets)
    return loss_v       # batch_size,  time_steps, n_voices

def calculate_average_hits_per_batch(token_logits, hit_value):
    tokens = torch.argmax(token_logits, dim=2)
    total_hits = tokens.eq(hit_value).sum(dim=1)
    avg = total_hits.float().mean().cpu()

    return avg

def batch_loop(dataloader_, model, device, optimizer=None, starting_step=None, hit_value=0):
    """
    This function iteratively loops over the given dataloader and calculates the loss for each batch. If an optimizer is
    provided, it will also perform the backward pass and update the model parameters. The loss values are accumulated
    and returned at the end of the loop.

    **Can be used for both training and testing. In testing however, backpropagation will not be performed**


    :param dataloader_:     (torch.utils.data.DataLoader)  dataloader for the dataset
    :param model:  (GrooveTransformerVAE)  the model
    :param hit_loss_fn:     (str)  either "dice" or "bce"
    :param velocity_loss_fn:    (str)  either "mse" or "bce"
    :param offset_loss_fn:  (str)  either "mse" or "bce"
    :param device:  (torch.device)  the device to use for the model
    :param optimizer:   (torch.optim.Optimizer)  the optimizer to use for the model
    :param starting_step:   (int)  the starting step for the optimizer
    :param kl_beta: (float)  the beta value for the KLD loss
    :return:    (dict)  a dictionary containing the loss values for the current batch

                metrics = {
                    "loss_total": np.mean(loss_total),
                    "loss_h": np.mean(loss_h),
                    "loss_v": np.mean(loss_v),
                    "loss_o": np.mean(loss_o),
                    "loss_KL": np.mean(loss_KL)}

                (int)  the current step of the optimizer (if provided)
    """
    # Prepare the metric trackers for the new epoch
    # ------------------------------------------------------------------------------------------
    loss_total, loss_token, loss_h, loss_v, avg_hits = [], [], [], [], []

    # Iterate over batches
    # ------------------------------------------------------------------------------------------
    for batch_count, (indices, in_tokens_, in_hv_, out_tokens_, out_hv_, masks_) in enumerate(dataloader_):
        # Move data to GPU if available
        # ---------------------------------------------------------------------------------------
        in_tokens = in_tokens_.to(device) if in_tokens_.device.type != device else in_tokens_
        in_hv = in_hv_.to(device) if in_hv_.device.type != device else in_hv_
        out_tokens = out_tokens_.to(device) if out_tokens_.device.type!= device else out_tokens_
        out_hv = out_hv_.to(device) if out_hv_.device.type != device else out_hv_
        masks = masks_.to(device) if masks_.device.type != device else masks_

        print(f"tokens shape: {in_tokens.shape}")
        print(f"hv shape: {in_hv.shape}")


        # Forward pass
        # ---------------------------------------------------------------------------------------

        token_logits, hit_logits, vel_logits = model.forward(in_tokens, in_hv, mask=masks)

        # Separate hit and velocity targets
        hit_targets, vel_targets = torch.split(out_hv, int(out_hv.shape[2] / 2), 2)


        # Compute losses
        # ---------------------------------------------------------------------------------------
        batch_loss_tokens = calculate_token_loss(token_logits=token_logits, token_targets=out_tokens)
        batch_loss_h = calculate_hit_loss(hit_logits = hit_logits, hit_targets=hit_targets)
        batch_loss_v = calculate_velocity_loss(vel_logits=vel_logits, vel_targets=vel_targets)
        batch_avg_hits = calculate_average_hits_per_batch(token_logits=token_logits, hit_value=hit_value)

        batch_loss_total = (batch_loss_tokens + batch_loss_h + batch_loss_v)

        # Backpropagation and optimization step (if training)
        # ---------------------------------------------------------------------------------------
        if optimizer is not None:
            optimizer.zero_grad()
            batch_loss_total.backward()
            optimizer.step()

        # Todo: Use raw logits here, and then pass through sigmoid and threshold
        # Todo: Obtain total number of hits, and return as list of total_hits

        # Update the per batch loss trackers
        # -----------------------------------------------------------------
        loss_token.append(batch_loss_tokens.item())
        loss_h.append(batch_loss_h.item())
        loss_v.append(batch_loss_v.item())
        avg_hits.append(batch_avg_hits)
        loss_total.append(batch_loss_total.item())
        # Add an average # of hits (total) - only on outputs


        # Increment the step counter
        # ---------------------------------------------------------------------------------------
        if starting_step is not None:
            starting_step += 1

    # empty gpu cache if cuda
    if device != 'cpu':
        torch.cuda.empty_cache()

    metrics = {
        "loss_total": np.mean(loss_total),
        "loss_token": np.mean(loss_token),
        "loss_h": np.mean(loss_h),
        "loss_v": np.mean(loss_v),
        "hit_totals": np.mean(avg_hits)
    }

    if starting_step is not None:
        return metrics, starting_step
    else:
        return metrics


def train_loop(train_dataloader, model, optimizer, device, starting_step, hit_value=0):
    """
    This function performs the training loop for the given model and dataloader. It will iterate over the dataloader
    and perform the forward and backward pass for each batch. The loss values are accumulated and the average is
    returned at the end of the loop.

    :param train_dataloader:    (torch.utils.data.DataLoader)  dataloader for the training dataset
    :param model:  (GrooveTransformerVAE)  the model
    :param optimizer:  (torch.optim.Optimizer)  the optimizer to use for the model (sgd or adam)
    :param hit_loss_fn:     ("dice" or torch.nn.BCEWithLogitsLoss)
    :param velocity_loss_fn:  (torch.nn.MSELoss or torch.nn.BCEWithLogitsLoss)
    :param offset_loss_fn:      (torch.nn.MSELoss or torch.nn.BCEWithLogitsLoss)
    :param loss_hit_penalty_multiplier:  (float)  the hit loss penalty multiplier
    :param device:  (str)  the device to use for the model
    :param starting_step:   (int)  the starting step for the optimizer
    :param kl_beta: (float)  the beta value for the KL loss

    :return:    (dict)  a dictionary containing the loss values for the current batch

            metrics = {
                    "train/loss_total": np.mean(loss_total),
                    "train/loss_h": np.mean(loss_h),
                    "train/loss_v": np.mean(loss_v),
                    "train/loss_o": np.mean(loss_o),
                    "train/loss_KL": np.mean(loss_KL)}
    """
    # ensure model is in training mode
    if model.training is False:
        logger.warning("Model is not in training mode. Setting to training mode.")
        model.train()

    # Run the batch loop
    metrics, starting_step = batch_loop(
        dataloader_=train_dataloader,
        model=model,
        device=device,
        optimizer=optimizer,
        starting_step=starting_step,
        hit_value=hit_value)

    metrics = {f"train/{key}": value for key, value in metrics.items()}
    return metrics, starting_step


def test_loop(test_dataloader, model, device, hit_value=0):
    """
    This function performs the test loop for the given model and dataloader. It will iterate over the dataloader
    and perform the forward pass for each batch. The loss values are accumulated and the average is returned at the end
    of the loop.

    :param test_dataloader:   (torch.utils.data.DataLoader)  dataloader for the test dataset
    :param model:  (GrooveTransformerVAE)  the model
    :param hit_loss_fn:     ("dice" or torch.nn.BCEWithLogitsLoss)
    :param velocity_loss_fn:    (torch.nn.MSELoss or torch.nn.BCEWithLogitsLoss)
    :param offset_loss_fn:    (torch.nn.MSELoss or torch.nn.BCEWithLogitsLoss)
    :param loss_hit_penalty_multiplier:     (float)  the hit loss penalty multiplier
    :param device:  (str)  the device to use for the model
    :return:   (dict)  a dictionary containing the loss values for the current batch

            metrics = {
                    "test/loss_total": np.mean(loss_total),
                    "test/loss_h": np.mean(loss_h),
                    "test/loss_v": np.mean(loss_v),
                    "test/loss_o": np.mean(loss_o),
                    "test/loss_KL": np.mean(loss_KL)}
    """
    # ensure model is in eval mode
    if model.training is True:
        logger.warning("Model is not in eval mode. Setting to eval mode.")
        model.eval()

    with torch.no_grad():
        # Run the batch loop
        metrics = batch_loop(
            dataloader_=test_dataloader,
            model=model,
            device=device,
            optimizer=None,
            hit_value=hit_value)

    metrics = {f"test/{key}": value for key, value in metrics.items()}
    return metrics


