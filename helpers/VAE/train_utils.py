
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


def dice_fn(hit_logits, hit_targets):
    """
    Dice loss function for binary segmentation problems. This function is used to calculate the loss for the hits.

    **This code was taken from https://gist.github.com/weiliu620/52d140b22685cf9552da4899e2160183**

    :param pred: Predicted output of the model
    :param target: Target output of the model
    :return: Dice loss value (1 - dice coefficient) where dice coefficient is calculated as 2*TP/(2*TP + FP + FN)

    :param pred:    (torch.Tensor)  predicted output of the model -->
    :param target:  (torch.Tensor)  target output of the model
    :return:
    """
    hit_probs = torch.sigmoid(hit_logits)

    smooth = 1.
    # have to use contiguous since they may from a torch.view op
    flat_probs = hit_probs.contiguous().view(-1)
    flat_targets = hit_targets.contiguous().view(-1)
    intersection = (flat_probs * flat_targets).sum()

    A_sum = torch.sum(flat_probs * flat_probs)
    B_sum = torch.sum(flat_targets * flat_targets)

    return 1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth))


def calculate_hit_loss(hit_logits, hit_targets, hit_loss_function, hit_loss_penalty_tensor=None):
    """
    Calculate the hit loss for the given hit logits and hit targets.
    The loss is calculated either using BCE or Dice loss function.
    :param hit_logits:  (torch.Tensor)  predicted output of the model (**Pre-ACTIVATION**)
    :param hit_targets:     (torch.Tensor)  target output of the model
    :param hit_loss_function:     (str)  either "bce" or "dice"
    :param hit_loss_penalty_tensor: (default None)
                                    (torch.Tensor)  tensor of shape (batch_size, seq_len, num_voices),
                                    containing the hit loss penalty values per each location in the sequences.
                                    **Only used when hit_loss_function is "bce"**
    :return:    hit_loss (float)  the hit loss value
    """
    if hit_loss_penalty_tensor is None:
        hit_loss_penalty_tensor = torch.ones_like(hit_targets)

    if hit_loss_function == "dice":
        assert hit_loss_function in ['dice']
        logger.warning(f"the hit_loss_penalty value is ignored for {hit_loss_function} loss function")
        hit_loss = dice_fn(hit_logits, hit_targets)
    else:
        assert isinstance(hit_loss_function, torch.nn.BCEWithLogitsLoss)
        loss_h = hit_loss_function(hit_logits, hit_targets) * hit_loss_penalty_tensor  # batch, time steps, voices
        bce_h_sum_voices = torch.sum(loss_h, dim=2)  # batch, time_steps
        hit_loss = bce_h_sum_voices.mean()

    return hit_loss


def calculate_velocity_loss(vel_logits, vel_targets, vel_loss_function, hit_loss_penalty_mat):
    """
    Calculate the velocity loss for the velocity targets and the **Pre-Activation** output of the model.
    The loss is calculated using either MSE or BCE loss function.
    :param vel_logits:  (torch.Tensor)  predicted output of the model (**Pre-ACTIVATION**)
    :param vel_targets:     (torch.Tensor)  target output of the model
    :param vel_loss_function:     (str)  either "mse" or "bce"
    :param hit_loss_penalty_mat: (torch.Tensor)  tensor of shape (batch_size, seq_len, num_voices),
                                    containing the hit loss penalty values per each location in the sequences.
                                    **Used regardless of the vel_loss_function**
    :return:    vel_loss (float)  the velocity loss value
    """
    if isinstance(vel_loss_function, torch.nn.MSELoss):
        loss_v = vel_loss_function(torch.sigmoid(vel_logits), vel_targets) * hit_loss_penalty_mat
    elif isinstance(vel_loss_function, torch.nn.BCEWithLogitsLoss):
        loss_v = vel_loss_function(vel_logits, vel_targets) * hit_loss_penalty_mat
    else:
        raise NotImplementedError(f"the vel_loss_function {vel_loss_function} is not implemented")

    return torch.sum(loss_v, dim=2).mean()


def calculate_offset_loss(offset_logits, offset_targets, offset_loss_function, hit_loss_penalty_mat):
    """
    Calculate the offset loss for the offset targets and the **Pre-Activation** output of the model.
    The loss is calculated using either MSE or BCE loss function.

    **For MSE, the offset_logit is first mapped to -0.5 to 0.5 using a tanh function. Alternatively, for BCE,
    it is assumed that the offset_logit will be activated using a sigmoid function.**

    :param offset_logits:  (torch.Tensor)  predicted output of the model (**Pre-ACTIVATION**)
    :param offset_targets:     (torch.Tensor)  target output of the model
    :param offset_loss_function:     (str)  either "mse" or "bce"
    :param hit_loss_penalty_mat: (torch.Tensor)  tensor of shape (batch_size, seq_len, num_voices),
                                    containing the hit loss penalty values per each location in the sequences.
                                    **Used regardless of the offset_loss_function**
    :return:    offset_loss (float)  the offset loss value

    """

    if isinstance(offset_loss_function, torch.nn.MSELoss):
        loss_o = offset_loss_function(torch.tanh(offset_logits), offset_targets) * hit_loss_penalty_mat
    elif isinstance(offset_loss_function, torch.nn.BCEWithLogitsLoss):
        # here the offsets MUST be in the range of [0, 1]. Our existing targets are from [-0.5, 0.5].
        # So we need to shift them to [0, 1] range by adding 0.5
        loss_o = offset_loss_function(offset_logits, offset_targets+0.5) * hit_loss_penalty_mat
    else:
        raise NotImplementedError(f"the offset_loss_function {offset_loss_function} is not implemented")

    return torch.sum(loss_o, dim=2).mean()


def calculate_kld_loss(mu, log_var):
    """calculate the KLD loss for the given mu and log_var values against a standard normal distribution"""
    return torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2. - log_var.exp(), dim=1), dim=0)


def batch_loop(dataloader_, groove_transformer_vae, hit_loss_fn, velocity_loss_fn,
               offset_loss_fn, loss_hit_penalty_multiplier, device, optimizer=None):
    """
    This function iteratively loops over the given dataloader and calculate the loss for each batch. If an optimizer is
    provided, it will also perform the backward pass and update the model parameters. The loss values are accumulated
    and returned at the end of the loop.

    **Can be used for both training and testing. In testing however, backpropagation will not be performed**


    :param dataloader_:     (torch.utils.data.DataLoader)  dataloader for the dataset
    :param groove_transformer_vae:  (GrooveTransformerVAE)  the model
    :param hit_loss_fn:     (str)  either "dice" or "bce"
    :param velocity_loss_fn:    (str)  either "mse" or "bce"
    :param offset_loss_fn:  (str)  either "mse" or "bce"
    :param loss_hit_penalty_multiplier:  (float)  the hit loss penalty multiplier
    :param device:  (torch.device)  the device to use for the model
    :param optimizer:   (torch.optim.Optimizer)  the optimizer to use for the model
    :return:    (dict)  a dictionary containing the loss values for the current batch

                metrics = {
                    "loss_total": np.mean(loss_total),
                    "loss_h": np.mean(loss_h),
                    "loss_v": np.mean(loss_v),
                    "loss_o": np.mean(loss_o),
                    "loss_KL": np.mean(loss_KL)}
    """
    # Prepare the metric trackers for the new epoch
    # ------------------------------------------------------------------------------------------
    loss_total, loss_h, loss_v, loss_o, loss_KL = [], [], [], [], []

    # Iterate over batches
    # ------------------------------------------------------------------------------------------
    for batch_count, (inputs, outputs, indices) in enumerate(dataloader_):
        # Move data to GPU if available
        # ---------------------------------------------------------------------------------------
        inputs = inputs.to(device)
        outputs = outputs.to(device)

        # Forward pass
        # ---------------------------------------------------------------------------------------
        (h_logits, v_logits, o_logits), mu, log_var, latent_z = groove_transformer_vae.forward(inputs)

        # Prepare targets for loss calculation
        h_targets, v_targets, o_targets = torch.split(outputs, int(outputs.shape[2] / 3), 2)
        hit_loss_penalty_mat = torch.where(h_targets == 1, float(1), float(loss_hit_penalty_multiplier))

        # Compute losses
        # ---------------------------------------------------------------------------------------
        batch_loss_h = calculate_hit_loss(
            hit_logits=h_logits, hit_targets=h_targets,
            hit_loss_function=hit_loss_fn, hit_loss_penalty_tensor=hit_loss_penalty_mat)

        batch_loss_v = calculate_velocity_loss(
            vel_logits=v_logits, vel_targets=v_targets,
            vel_loss_function=velocity_loss_fn, hit_loss_penalty_mat=hit_loss_penalty_mat)

        batch_loss_o = calculate_offset_loss(
            offset_logits=o_logits, offset_targets=o_targets,
            offset_loss_function=offset_loss_fn, hit_loss_penalty_mat=hit_loss_penalty_mat)

        batch_loss_KL = calculate_kld_loss(mu, log_var)

        batch_loss_total = batch_loss_h + batch_loss_v + batch_loss_o + batch_loss_KL

        # Backward pass
        # ---------------------------------------------------------------------------------------
        if optimizer is not None:
            optimizer.zero_grad()
            batch_loss_total.backward()
            optimizer.step()

        # Update the per batch loss trackers
        # ---------------------------------------------------------------------------------------
        loss_h.append(batch_loss_h.item())
        loss_v.append(batch_loss_v.item())
        loss_o.append(batch_loss_o.item())
        loss_KL.append(batch_loss_KL.item())
        loss_total.append(batch_loss_total.item())

    # empty gpu cache if cuda
    if device != 'cpu':
        torch.cuda.empty_cache()

    metrics = {
        "loss_total": np.mean(loss_total),
        "loss_h": np.mean(loss_h),
        "loss_v": np.mean(loss_v),
        "loss_o": np.mean(loss_o),
        "loss_KL": np.mean(loss_KL)}

    return metrics


def train_loop(train_dataloader, groove_transformer_vae, optimizer, hit_loss_fn, velocity_loss_fn,
               offset_loss_fn, loss_hit_penalty_multiplier, device):
    """
    This function performs the training loop for the given model and dataloader. It will iterate over the dataloader
    and perform the forward and backward pass for each batch. The loss values are accumulated and the average is
    returned at the end of the loop.

    :param train_dataloader:    (torch.utils.data.DataLoader)  dataloader for the training dataset
    :param groove_transformer_vae:  (GrooveTransformerVAE)  the model
    :param optimizer:  (torch.optim.Optimizer)  the optimizer to use for the model (sgd or adam)
    :param hit_loss_fn:     ("dice" or torch.nn.BCEWithLogitsLoss)
    :param velocity_loss_fn:  (torch.nn.MSELoss or torch.nn.BCEWithLogitsLoss)
    :param offset_loss_fn:      (torch.nn.MSELoss or torch.nn.BCEWithLogitsLoss)
    :param loss_hit_penalty_multiplier:  (float)  the hit loss penalty multiplier
    :param device:  (str)  the device to use for the model

    :return:    (dict)  a dictionary containing the loss values for the current batch

            metrics = {
                    "train/loss_total": np.mean(loss_total),
                    "train/loss_h": np.mean(loss_h),
                    "train/loss_v": np.mean(loss_v),
                    "train/loss_o": np.mean(loss_o),
                    "train/loss_KL": np.mean(loss_KL)}
    """
    # ensure model is in training mode
    if groove_transformer_vae.training is False:
        logger.warning("Model is not in training mode. Setting to training mode.")
        groove_transformer_vae.train()

    # Run the batch loop
    metrics = batch_loop(
        dataloader_=train_dataloader,
        groove_transformer_vae=groove_transformer_vae,
        hit_loss_fn=hit_loss_fn,
        velocity_loss_fn=velocity_loss_fn,
        offset_loss_fn=offset_loss_fn,
        loss_hit_penalty_multiplier=loss_hit_penalty_multiplier,
        device=device,
        optimizer=optimizer)

    metrics = {f"train/{key}": value for key, value in metrics.items()}
    return metrics


def test_loop(test_dataloader, groove_transformer_vae, hit_loss_fn, velocity_loss_fn,
               offset_loss_fn, loss_hit_penalty_multiplier, device):
    """
    This function performs the test loop for the given model and dataloader. It will iterate over the dataloader
    and perform the forward pass for each batch. The loss values are accumulated and the average is returned at the end
    of the loop.

    :param test_dataloader:   (torch.utils.data.DataLoader)  dataloader for the test dataset
    :param groove_transformer_vae:  (GrooveTransformerVAE)  the model
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
    if groove_transformer_vae.training is True:
        logger.warning("Model is not in eval mode. Setting to eval mode.")
        groove_transformer_vae.eval()

    with torch.no_grad():
        # Run the batch loop
        metrics = batch_loop(
            dataloader_=test_dataloader,
            groove_transformer_vae=groove_transformer_vae,
            hit_loss_fn=hit_loss_fn,
            velocity_loss_fn=velocity_loss_fn,
            offset_loss_fn=offset_loss_fn,
            loss_hit_penalty_multiplier=loss_hit_penalty_multiplier,
            device=device,
            optimizer=None)

    metrics = {f"test/{key}": value for key, value in metrics.items()}
    return metrics