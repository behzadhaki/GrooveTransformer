
import numpy as np
import torch
from loss_functions import *



def batch_loop(dataloader_, vae_model, adversarial_models, hit_loss_fn, velocity_loss_fn,
               offset_loss_fn, device, vae_optimizer=None, starting_step=None,
               kl_beta=1.0, adversarial_loss_modifier=1.0, encoder_loss_modifier=1.0,
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
    loss_total, loss_recon, loss_h, loss_v, loss_o, loss_KL, loss_KL_beta_scaled, \
        loss_adv_density, loss_density, loss_adv_sync, loss_sync, loss_adv_genre, loss_genre = [], [], [], [], [], [], [], [], [], [], [], [], []

    # Iterate over batches
    # ------------------------------------------------------------------------------------------
    for batch_count, (inputs_, outputs_, densities_, syncopations_, genres_,
                      hit_balancing_weights_per_sample_, genre_balancing_weights_per_sample_,
                      indices) in enumerate(dataloader_):

        # Move data to GPU if available
        # ---------------------------------------------------------------------------------------
        inputs = inputs_.to(device) if inputs_.device.type!= device else inputs_
        outputs = outputs_.to(device) if outputs_.device.type!= device else outputs_
        densities = densities_.to(device) if densities_.device.type!= device else densities_
        syncopations = syncopations_.to(device) if syncopations_.device.type!= device else syncopations_
        genres = genres_.to(device) if genres_.device.type!= device else genres_
        hit_balancing_weights_per_sample = hit_balancing_weights_per_sample_.to(device) \
            if hit_balancing_weights_per_sample_.device.type!= device else hit_balancing_weights_per_sample_
        genre_balancing_weights_per_sample = genre_balancing_weights_per_sample_.to(device) \
            if genre_balancing_weights_per_sample_.device.type!= device else genre_balancing_weights_per_sample_


        # Forward pass of VAE
        # ---------------------------------------------------------------------------------------
        (h_logits, v_logits, o_logits), mu, log_var, latent_z = vae_model.forward(inputs, densities)

        # Prepare data for loss calculation
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

        batch_loss_recon = (batch_loss_h + batch_loss_v + batch_loss_o)

        # KL
        batch_loss_KL = calculate_kld_loss(mu, log_var)
        batch_loss_KL_Beta_Scaled = (batch_loss_KL * genre_balancing_weights_per_sample[:, 0, 0].view(-1, 1)) * kl_beta
        batch_loss_KL_Beta_Scaled = batch_loss_KL_Beta_Scaled.sum() if \
            reduce_by_sum else batch_loss_KL_Beta_Scaled.mean()
        batch_loss_KL = batch_loss_KL.sum() if reduce_by_sum else batch_loss_KL.mean()

        # Adversarial Latent
        adversarial_loss = 0
        encoding_loss = 0

        if adversarial_models["density"]["active"]:
            z_star = remove_elements(latent_z, 0)
            adversarial_density_pred = adversarial_models["density"]["model"].forward(z_star)
            adversarial_density_loss = calculate_regressor_loss(adversarial_density_pred, densities)
            adversarial_loss += adversarial_density_loss
            density_loss = calculate_regressor_loss(latent_z[:, 0], densities)
            encoding_loss += density_loss
            # Record the values
            loss_adv_density.append(adversarial_density_loss.item())
            loss_density.append(density_loss.item())


        if adversarial_models["syncopation"]["active"]:
            z_star = remove_elements(latent_z, 1)
            adversarial_syncopation_pred = adversarial_models["syncopation"]["model"].forward(z_star)
            adversarial_syncopation_loss = calculate_regressor_loss(adversarial_syncopation_pred, syncopations)
            adversarial_loss += adversarial_syncopation_loss
            syncopation_loss = calculate_regressor_loss(latent_z[:, 1], syncopations)
            encoding_loss += syncopation_loss
            # Record the values
            loss_adv_sync.append(adversarial_syncopation_loss)
            loss_sync.append(syncopation_loss.item())

        if adversarial_models["genre"]["active"]:
            z_star = remove_elements(latent_z, start=2, end=18)
            adversarial_genre_pred = adversarial_models["genre"]["model"].forward(z_star)
            adversarial_genre_loss = calculate_classifier_loss(adversarial_genre_pred, genres)
            adversarial_loss += adversarial_genre_loss
            genre_loss = calculate_classifier_loss(latent_z[:, 2:18], genres)
            encoding_loss += genre_loss
            # Record the values
            loss_adv_genre.append(adversarial_genre_loss)
            loss_genre.append(genre_loss.item())

        # Scale losses according to hyperparams
        adversarial_loss *= adversarial_loss_modifier
        encoding_loss *= encoder_loss_modifier

        batch_loss_total = batch_loss_recon + batch_loss_KL_Beta_Scaled - adversarial_loss + encoding_loss

        # Backpropagation and optimization step (if training)
        # ---------------------------------------------------------------------------------------
        if vae_optimizer is not None:
            vae_optimizer.zero_grad()
            batch_loss_total.backward()
            vae_optimizer.step()

        # Train the adversarial networks
        # ---------------------------------------------------------------------------------------
        if adversarial_models["density"]["active"]:
            adversarial_models["density"]["optimizer"].zero_grad()
            z_star = remove_elements(latent_z, 0)
            predictions = adversarial_models["density"]["model"].forward(z_star.detach())
            loss = calculate_regressor_loss(predictions, densities)
            loss.backward()
            adversarial_models["density"]["optimizer"].step()

        if adversarial_models["syncopation"]["active"]:
            adversarial_models["syncopation"]["optimizer"].zero_grad()
            z_star = remove_elements(latent_z, 0)
            predictions = adversarial_models["syncopation"]["model"].forward(z_star.detach())
            loss = calculate_regressor_loss(predictions, densities)
            loss.backward()
            adversarial_models["syncopation"]["optimizer"].step()

        if adversarial_models["genre"]["active"]:
            adversarial_models["genre"]["optimizer"].zero_grad()
            z_star = remove_elements(latent_z, 0)
            predictions = adversarial_models["genre"]["model"].forward(z_star.detach())
            loss = calculate_classifier_loss(predictions, densities)
            loss.backward()
            adversarial_models["genre"]["optimizer"].step()

        # Update the per batch loss trackers
        # -----------------------------------------------------------------
        loss_h.append(batch_loss_h.item())
        loss_v.append(batch_loss_v.item())
        loss_o.append(batch_loss_o.item())
        loss_total.append(batch_loss_total.item())
        loss_recon.append(batch_loss_recon.item())
        loss_KL.append(batch_loss_KL.item())
        loss_KL_beta_scaled.append(batch_loss_KL_Beta_Scaled.item())


        # Increment the step counter
        # ---------------------------------------------------------------------------------------
        if starting_step is not None:
            starting_step += 1

    # empty gpu cache if cuda
    if device != 'cpu':
        torch.cuda.empty_cache()

    metrics = {
        "loss_total": np.mean(loss_total),
        "loss_h": np.mean(loss_h),
        "loss_v": np.mean(loss_v),
        "loss_o": np.mean(loss_o),
        "loss_KL": np.mean(loss_KL),
        "loss_KL_beta_scaled": np.mean(loss_KL_beta_scaled),
        "loss_recon": np.mean(loss_recon),
        "loss_adv_density": np.mean(loss_adv_density),
        "loss_enc_density": np.mean(loss_density),
        "loss_adv_syncopation": np.mean(loss_adv_sync),
        "loss_enc_syncopation": np.mean(loss_sync),
        "loss_adv_genre": np.mean(loss_adv_genre),
        "loss_enc_genre": np.mean(loss_genre)
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
        balance_vo=balance_vo,
        train_density=train_genre,
        train_syncopation=train_syncopation,
        train_genre=train_genre)

    metrics = {f"train/{key}": value for key, value in metrics.items()}
    return metrics, starting_step


def test_loop(test_dataloader, model, hit_loss_fn, velocity_loss_fn,
              offset_loss_fn, device, kl_beta=1, reduce_by_sum=False, balance_vo=True):
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
    if model.training is True:
        model.eval()

    with torch.no_grad():
        # Run the batch loop
        metrics = batch_loop(
            dataloader_=test_dataloader,
            vae_model=model,
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