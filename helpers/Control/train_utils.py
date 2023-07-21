
import numpy as np
import torch
from helpers.Control.loss_functions import *



def batch_loop(dataloader_, vae_model, adversarial_models, hit_loss_fn, velocity_loss_fn,
               offset_loss_fn, device, vae_optimizer=None, starting_step=None,
               kl_beta=1.0, adversarial_loss_modifier=1.0, control_encoding_loss_modifier=0.2,
               reduce_by_sum=False, balance_vo=True):
    """
    This function iteratively loops over the given dataloader and calculates the loss for each batch. If an optimizer is
    provided, it will also perform the backward pass and update the model parameters. The loss values are accumulated
    and returned at the end of the loop.

    **Can be used for both training and testing. In testing however, backpropagation will not be performed**


    :param dataloader_:     (torch.utils.data.DataLoader)  dataloader for the dataset
    :param vae_model:  (GrooveTransformerVAE)  the model
    :param adversarial_models: (dict) the models used for adversarial training in format {active, model, optimizer}
    :param hit_loss_fn:     (str)  either "dice" or "bce"
    :param velocity_loss_fn:    (str)  either "mse" or "bce"
    :param offset_loss_fn:  (str)  either "mse" or "bce"
    :param device:  (torch.device)  the device to use for the model
    :param vae_optimizer:   (torch.optim.Optimizer)  the optimizer to use for the model
    :param starting_step:   (int)  the starting step for the optimizer
    :param kl_beta: (float)  the beta value for the KLD loss
    :param reduce_by_sum:   (bool)  whether to reduce the loss by sum or mean
    :return:    (dict)  a dictionary containing the loss values for the current batch
    """
    # Prepare the metric trackers for the new epoch
    # ------------------------------------------------------------------------------------------
    vae_loss_total = []
    loss_recon, loss_h, loss_v, loss_o = [], [], [], []
    loss_kl, loss_kl_beta_scaled = [], []

    # For each control parameter we have 3 loss functions to track:
    # 1. Gradient Reversal Layer (grl), 2. Adversarial Loss 3. Encoding Loss
    loss_adversarial_trackers = {"density_grl": [],
                                 "density": [],
                                 "density_encoding": [],
                                 "sync_grl": [],
                                 "sync": [],
                                 "sync_encoding": [],
                                 "genre_grl": [],
                                 "genre": [],
                                 "genre_encoding": []}

    loss_adversarial_total, loss_encoder_total = [], []

    # Iterate over batches
    # ------------------------------------------------------------------------------------------
    for batch_count, (inputs_, outputs_, densities_, syncopations_, genres_,
                      hit_balancing_weights_per_sample_, genre_balancing_weights_per_sample_,
                      indices) in enumerate(dataloader_):

        # Move data to GPU if available
        # ---------------------------------------------------------------------------------------
        data_list = [inputs_, outputs_, densities_, syncopations_, genres_,
                     hit_balancing_weights_per_sample_, genre_balancing_weights_per_sample_]

        inputs, outputs, densities, syncopations, genres, \
            hit_balancing_weights_per_sample, genre_balancing_weights_per_sample = \
            [tensor.to(device) if tensor.device.type != device else tensor for tensor in data_list]
        # inputs = inputs_.to(device) if inputs_.device.type!= device else inputs_
        # outputs = outputs_.to(device) if outputs_.device.type!= device else outputs_
        # densities = densities_.to(device) if densities_.device.type!= device else densities_
        # syncopations = syncopations_.to(device) if syncopations_.device.type!= device else syncopations_
        # genres = genres_.to(device) if genres_.device.type!= device else genres_
        # hit_balancing_weights_per_sample = hit_balancing_weights_per_sample_.to(device) \
        #     if hit_balancing_weights_per_sample_.device.type!= device else hit_balancing_weights_per_sample_
        # genre_balancing_weights_per_sample = genre_balancing_weights_per_sample_.to(device) \
        #     if genre_balancing_weights_per_sample_.device.type!= device else genre_balancing_weights_per_sample_


        # Forward pass of VAE
        # ---------------------------------------------------------------------------------------
        (h_logits, v_logits, o_logits), mu, log_var, latent_z = vae_model.forward(inputs, densities)
        h_targets, v_targets, o_targets = torch.split(outputs, int(outputs.shape[2] / 3), 2)

        # Compute VAE losses
        # ---------------------------------------------------------------------------------------
        # Reconstruction
        batch_loss_h = calculate_hit_loss(
            hit_logits=h_logits, hit_targets=h_targets, hit_loss_function=hit_loss_fn)
        batch_loss_h = (batch_loss_h * hit_balancing_weights_per_sample * genre_balancing_weights_per_sample)
        batch_loss_h = batch_loss_h.sum() if reduce_by_sum else batch_loss_h.mean()

        if balance_vo:
            mask = h_logits.detach().clone()
            v_logits = v_logits * mask
            o_logits = o_logits * mask

        batch_loss_v = calculate_velocity_loss(
            vel_logits=v_logits, vel_targets=v_targets, vel_loss_function=velocity_loss_fn)
        batch_loss_v = (batch_loss_v * hit_balancing_weights_per_sample * genre_balancing_weights_per_sample)
        batch_loss_v = batch_loss_v.sum() if reduce_by_sum else batch_loss_v.mean()

        batch_loss_o = calculate_offset_loss(
            offset_logits=o_logits, offset_targets=o_targets, offset_loss_function=offset_loss_fn)
        batch_loss_o = (batch_loss_o * hit_balancing_weights_per_sample * genre_balancing_weights_per_sample)
        batch_loss_o = batch_loss_o.sum() if reduce_by_sum else batch_loss_o.mean()

        batch_loss_recon = batch_loss_h + batch_loss_v + batch_loss_o

        # KL
        batch_loss_KL = calculate_kld_loss(mu, log_var)
        batch_loss_KL_Beta_Scaled = (batch_loss_KL * genre_balancing_weights_per_sample[:, 0, 0].view(-1, 1)) * kl_beta
        batch_loss_KL_Beta_Scaled = batch_loss_KL_Beta_Scaled.sum() if \
            reduce_by_sum else batch_loss_KL_Beta_Scaled.mean()
        batch_loss_KL = batch_loss_KL.sum() if reduce_by_sum else batch_loss_KL.mean()

        # Adversarial Latent
        params = {"densities": densities}
        adversarial_loss, control_encoding_loss = \
            calculate_adversarial_losses(adversarial_models, latent_z, params, loss_adversarial_trackers)
        adversarial_loss *= adversarial_loss_modifier
        control_encoding_loss *= control_encoding_loss_modifier

        batch_vae_loss_total = batch_loss_recon + batch_loss_KL_Beta_Scaled + control_encoding_loss - adversarial_loss

        # Backpropagation and optimization step (if training)
        # ---------------------------------------------------------------------------------------
        if vae_optimizer is not None:
            vae_optimizer.zero_grad()
            batch_vae_loss_total.backward()
            vae_optimizer.step()

            # Train the adversarial networks to estimate control values from z_star
            train_adversarial_models(adversarial_models, latent_z, params, loss_adversarial_trackers)

        # Update the per batch loss trackers
        # -----------------------------------------------------------------
        loss_h.append(batch_loss_h.item())
        loss_v.append(batch_loss_v.item())
        loss_o.append(batch_loss_o.item())
        loss_recon.append(batch_loss_recon.item())
        loss_kl.append(batch_loss_KL.item())
        loss_kl_beta_scaled.append(batch_loss_KL_Beta_Scaled.item())
        loss_adversarial_total.append(adversarial_loss.item())
        vae_loss_total.append(batch_vae_loss_total.item())


        # Increment the step counter
        # ---------------------------------------------------------------------------------------
        if starting_step is not None:
            starting_step += 1

    # empty gpu cache if cuda
    if device != 'cpu':
        torch.cuda.empty_cache()
    metrics = {
        "vae_loss_total": np.mean(vae_loss_total),

        "loss_recon": np.mean(loss_recon),
        "loss_h": np.mean(loss_h),
        "loss_v": np.mean(loss_v),
        "loss_o": np.mean(loss_o),

        "loss_kl": np.mean(loss_kl),
        "loss_kl_beta_scaled": np.mean(loss_kl_beta_scaled),

        "loss_adversarial_total": np.mean(loss_adversarial_total),

        "loss_adv_density_grl": np.mean(loss_adversarial_trackers["density_grl"]),
        "loss_adv_density": np.mean(loss_adversarial_trackers["density"]),
        "loss_enc_density": np.mean(loss_adversarial_trackers["density_encoding"]),

        "loss_adv_syncopation_grl": np.mean(loss_adversarial_trackers["sync_grl"]),
        "loss_adv_syncopation": np.mean(loss_adversarial_trackers["sync"]),
        "loss_enc_syncopation": np.mean(loss_adversarial_trackers["sync_encoding"]),

        "loss_adv_genre_grl": np.mean(loss_adversarial_trackers["genre_grl"]),
        "loss_adv_genre": np.mean(loss_adversarial_trackers["genre"]),
        "loss_enc_genre": np.mean(loss_adversarial_trackers["genre_encoding"]),
    }

    if starting_step is not None:
        return metrics, starting_step
    else:
        return metrics


def train_loop(train_dataloader, model, adversarial_models, vae_optimizer,  hit_loss_fn, velocity_loss_fn,
               offset_loss_fn, device, starting_step, kl_beta=1, reduce_by_sum=False, balance_vo=True):
    """
    This function performs the training loop for the given model and dataloader. It will iterate over the dataloader
    and perform the forward and backward pass for each batch. The loss values are accumulated and the average is
    returned at the end of the loop.

    :param train_dataloader:    (torch.utils.data.DataLoader)  dataloader for the training dataset
    :param model:  (GrooveTransformerVAE)  the model
    :param vae_optimizer:  (torch.optim.Optimizer)  the optimizer to use for the model (sgd or adam)
    :param hit_loss_fn:     ("dice" or torch.nn.BCEWithLogitsLoss)
    :param velocity_loss_fn:  (torch.nn.MSELoss or torch.nn.BCEWithLogitsLoss)
    :param offset_loss_fn:      (torch.nn.MSELoss or torch.nn.BCEWithLogitsLoss)
    :param loss_hit_penalty_multiplier:  (float)  the hit loss penalty multiplier
    :param device:  (str)  the device to use for the model
    :param starting_step:   (int)  the starting step for the optimizer
    :param kl_beta: (float)  the beta value for the KL loss
    :param reduce_by_sum:   (bool)  if True, the loss values are reduced by sum instead of mean

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
        model.train()

    # Run the batch loop
    metrics, starting_step = batch_loop(
        dataloader_=train_dataloader,
        vae_model=model,
        adversarial_models=adversarial_models,
        hit_loss_fn=hit_loss_fn,
        velocity_loss_fn=velocity_loss_fn,
        offset_loss_fn=offset_loss_fn,
        device=device,
        vae_optimizer=vae_optimizer,
        starting_step=starting_step,
        kl_beta=kl_beta,
        reduce_by_sum=reduce_by_sum,
        balance_vo=balance_vo)

    metrics = {f"train/{key}": value for key, value in metrics.items()}
    return metrics, starting_step


def test_loop(test_dataloader, vae_model, adversarial_models, hit_loss_fn, velocity_loss_fn,
              offset_loss_fn, device, kl_beta=1, reduce_by_sum=False, balance_vo=True):
    """
    This function performs the test loop for the given model and dataloader. It will iterate over the dataloader
    and perform the forward pass for each batch. The loss values are accumulated and the average is returned at the end
    of the loop.

    :param test_dataloader:   (torch.utils.data.DataLoader)  dataloader for the test dataset
    :param vae_model:  (GrooveTransformerVAE)  the model
    :param hit_loss_fn:     ("dice" or torch.nn.BCEWithLogitsLoss)
    :param velocity_loss_fn:    (torch.nn.MSELoss or torch.nn.BCEWithLogitsLoss)
    :param offset_loss_fn:    (torch.nn.MSELoss or torch.nn.BCEWithLogitsLoss)
    :param loss_hit_penalty_multiplier:     (float)  the hit loss penalty multiplier
    :param device:  (str)  the device to use for the model
    :param kl_beta: (float)  the beta value for the KL loss
    :param reduce_by_sum:   (bool)  if True, the loss values are reduced by sum instead of mean
    :return:   (dict)  a dictionary containing the loss values for the current batch

            metrics = {
                    "test/loss_total": np.mean(loss_total),
                    "test/loss_h": np.mean(loss_h),
                    "test/loss_v": np.mean(loss_v),
                    "test/loss_o": np.mean(loss_o),
                    "test/loss_KL": np.mean(loss_KL)}
    """
    # ensure model is in eval mode
    if vae_model.training is True:
        vae_model.eval()

    with torch.no_grad():
        # Run the batch loop
        metrics = batch_loop(
            dataloader_=test_dataloader,
            vae_model=vae_model,
            adversarial_models=adversarial_models,
            hit_loss_fn=hit_loss_fn,
            velocity_loss_fn=velocity_loss_fn,
            offset_loss_fn=offset_loss_fn,
            device=device,
            vae_optimizer=None,
            kl_beta=kl_beta,
            reduce_by_sum=reduce_by_sum,
            balance_vo=balance_vo)

    metrics = {f"test/{key}": value for key, value in metrics.items()}
    return metrics