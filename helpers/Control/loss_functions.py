import torch
import numpy as np
from copy import deepcopy

def calculate_hit_loss(hit_logits, hit_targets, hit_loss_function):
    """
    Calculate the hit loss for the given hit logits and hit targets.
    The loss is calculated either using BCE or Dice loss function.
    :param hit_logits:  (torch.Tensor)  predicted output of the model (**Pre-ACTIVATION**)
    :param hit_targets:     (torch.Tensor)  target output of the model
    :param hit_loss_function:     (torch.nn.BCEWithLogitsLoss)
    :return:    hit_loss (batch, time_steps, n_voices)  the hit loss value per each step and voice (unreduced)
    """
    assert isinstance(hit_loss_function, torch.nn.BCEWithLogitsLoss)
    loss_h = hit_loss_function(hit_logits, hit_targets)           # batch, time steps, voices
    return loss_h       # batch_size,  time_steps, n_voices


def calculate_velocity_loss(vel_logits, vel_targets, vel_loss_function):
    """
    Calculate the velocity loss for the velocity targets and the **Pre-Activation** output of the model.
    The loss is calculated using either MSE or BCE loss function.
    :param vel_logits:  (torch.Tensor)  predicted output of the model (**Pre-ACTIVATION**)
    :param vel_targets:     (torch.Tensor)  target output of the model
    :param vel_loss_function:     (str)  either "mse" or "bce"
    :return:    vel_loss (batch_size, time_steps, n_voices)  the velocity loss value per each step and voice (unreduced)
    """
    if isinstance(vel_loss_function, torch.nn.MSELoss):
        loss_v = vel_loss_function(torch.sigmoid(vel_logits), vel_targets)
    elif isinstance(vel_loss_function, torch.nn.BCEWithLogitsLoss):
        loss_v = vel_loss_function(vel_logits, vel_targets)
    else:
        raise NotImplementedError(f"the vel_loss_function {vel_loss_function} is not implemented")

    return loss_v       # batch_size,  time_steps, n_voices


def calculate_offset_loss(offset_logits, offset_targets, offset_loss_function):
    """
    Calculate the offset loss for the offset targets and the **Pre-Activation** output of the model.
    The loss is calculated using either MSE or BCE loss function.

    **For MSE, the offset_logit is first mapped to -0.5 to 0.5 using a tanh function. Alternatively, for BCE,
    it is assumed that the offset_logit will be activated using a sigmoid function.**

    :param offset_logits:  (torch.Tensor)  predicted output of the model (**Pre-ACTIVATION**)
    :param offset_targets:     (torch.Tensor)  target output of the model
    :param offset_loss_function:     (str)  either "mse" or "bce"
    :return:    offset_loss (batch_size, time_steps, n_voices)  the offset loss value per each step
                    and voice (unreduced)

    """

    if isinstance(offset_loss_function, torch.nn.MSELoss):
        # the offset logits after the tanh activation are in the range of -1 to 1 . Therefore, we need to
        # scale the offset targets to the same range. This is done by multiplying the offset values after
        # the tanh activation by 0.5
        loss_o = offset_loss_function(torch.tanh(offset_logits)*0.5, offset_targets)
    elif isinstance(offset_loss_function, torch.nn.BCEWithLogitsLoss):
        # here the offsets MUST be in the range of [0, 1]. Our existing targets are from [-0.5, 0.5].
        # So we need to shift them to [0, 1] range by adding 0.5
        loss_o = offset_loss_function(offset_logits, offset_targets+0.5)
    else:
        raise NotImplementedError(f"the offset_loss_function {offset_loss_function} is not implemented")

    return loss_o           # batch_size,  time_steps, n_voices


def calculate_kld_loss(mu, log_var):
    """ calculate the KLD loss for the given mu and log_var values against a standard normal distribution
    :param mu:  (torch.Tensor)  the mean values of the latent space
    :param log_var: (torch.Tensor)  the log variance values of the latent space
    :return:    kld_loss (torch.Tensor)  the KLD loss value (unreduced) shape: (batch_size,  time_steps, n_voices)

    """
    mu = mu.view(mu.shape[0], -1)
    log_var = log_var.view(log_var.shape[0], -1)
    kld_loss = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())

    return kld_loss     # batch_size,  time_steps, n_voices\


def calculate_adversarial_losses(adversarial_models, latent_z, params, trackers):
    """
    Calculate the adversarial + encoding losses for each parameter.
    :param adversarial_models: (dictionary) in format {active, model, optimizer}
    :param latent_z: (torch.Tensor) the latent space vector
    :param params: (dictionary) each element is a tensor of ground truth parameters from current batch
    :param trackers: (dictionary) lists of each loss value to be tracked. Append and take mean at end of epoch
    """

    adversarial_loss = 0.0
    control_encoding_loss = 0.0

    if adversarial_models["density"]["active"]:
        densities_gt = params["densities"]
        densities_onehot_gt = convert_continuous_values_to_onehot_vectors(densities_gt, adversarial=False)

        z_star, ctrl_encoding = separate_control_elements(latent_z, 0, squeeze_result=True)

        # Get predictions from the classifier
        density_preds = adversarial_models["density"]["model"].forward(z_star)

        adversarial_density_loss = calculate_classifier_loss(density_preds, densities_onehot_gt)
        adversarial_loss += adversarial_density_loss

        density_loss = calculate_regressor_loss(ctrl_encoding, densities_gt)
        control_encoding_loss += density_loss

        trackers["density_grl"].append(adversarial_density_loss.item())
        trackers["density_encoding"].append(density_loss.item())

    if adversarial_models["intensity"]["active"]:
        z_star = separate_control_elements(latent_z, 1)
        adversarial_intensity_preds = adversarial_models["intensity"]["model"].forward(z_star)
        adversarial_intensity_loss = calculate_regressor_loss(adversarial_intensity_preds, params["intensities"])
        adversarial_loss += adversarial_intensity_loss

        intensity_loss = calculate_regressor_loss(latent_z[:, 1], params["intensities"])
        control_encoding_loss += intensity_loss

        trackers["intensity_grl"].append(adversarial_intensity_loss.item())
        trackers["intensity_encoding"].append(intensity_loss.item())

    if adversarial_models["genre"]["active"]:
        z_star = separate_control_elements(latent_z, start=2, end=18)
        adversarial_genre_pred = adversarial_models["genre"]["model"].forward(z_star)
        adversarial_genre_loss = calculate_classifier_loss(adversarial_genre_pred, params["genres"])
        adversarial_loss += adversarial_genre_loss

        genre_loss = calculate_classifier_loss(latent_z[:, 2:18], params["genres"])
        control_encoding_loss += genre_loss

        trackers["genre_grl"].append(adversarial_genre_loss.item())
        trackers["genre_encoding"].append(genre_loss.item())

    return adversarial_loss, control_encoding_loss


def train_classifier_models(adversarial_models, latent_z, params, trackers, backprop, loss_scale=1.0):
    """
    Train the adversarial classifier models to accurately determine the control value(s) from
    the non-control elements of the latent space.
    """
    classifier_loss_total = 0.0

    if adversarial_models["density"]["active"]:
        z_star, _ = separate_control_elements(latent_z, 0)
        predictions = adversarial_models["density"]["model"].forward(z_star.detach())
        densities_gt_onehot = convert_continuous_values_to_onehot_vectors(params["densities"])

        loss = calculate_classifier_loss(predictions, densities_gt_onehot) * loss_scale

        if backprop:
            adversarial_models["density"]["optimizer"].zero_grad()
            loss.backward()
            adversarial_models["density"]["optimizer"].step()

        trackers["density_classifier"].append(loss.item())
        classifier_loss_total += loss.item()

    if adversarial_models["intensity"]["active"]:
        adversarial_models["intensity"]["optimizer"].zero_grad()
        z_star = separate_control_elements(latent_z, 0)
        predictions = adversarial_models["intensity"]["model"].forward(z_star.detach())
        loss = calculate_regressor_loss(predictions, params["intensities"])
        loss.backward()
        adversarial_models["intensity"]["optimizer"].step()
        trackers["intensity"].append(loss.item())

    if adversarial_models["genre"]["active"]:
        adversarial_models["genre"]["optimizer"].zero_grad()
        z_star = separate_control_elements(latent_z, 0)
        predictions = adversarial_models["genre"]["model"].forward(z_star.detach())
        loss = calculate_classifier_loss(predictions, params["genres"])
        loss.backward()
        adversarial_models["genre"]["optimizer"].step()
        trackers["genre"].append(loss.item())

    return classifier_loss_total

def calculate_regressor_loss(prediction, target):
    loss_fn = torch.nn.MSELoss()
    return loss_fn(prediction, target)


def calculate_classifier_loss(prediction, target):
    loss_fn = torch.nn.BCELoss()
    loss = loss_fn(prediction, target)
    return loss


def generate_beta_curve(n_epochs, period_epochs, rise_ratio, start_first_rise_at_epoch=0):
    """
    Generate a beta curve for the given parameters

    Args:
        n_epochs:            The number of epochs to generate the curve for
        period_epochs:       The period of the curve in epochs (for multiple cycles)
        rise_ratio:         The ratio of the period to be used for the rising part of the curve
        start_first_rise_at_epoch:  The epoch to start the first rise at (useful for warmup)

    Returns:

    """
    def f(x, K):
        if x == 0:
            return 0
        elif x == K:
            return 1
        else:
            return 1 / (1 + np.exp(-10 * (x - K / 2) / K))

    def generate_rising_curve(K):
        curve = []
        for i in range(K):
            curve.append(f(i, K - 1))
        return np.array(curve)

    def generate_single_beta_cycle(period, rise_ratio):
        cycle = np.ones(period)

        curve_steps_in_epochs = int(period * rise_ratio)

        rising_curve = generate_rising_curve(curve_steps_in_epochs)

        cycle[:rising_curve.shape[0]] = rising_curve[:cycle.shape[0]]

        return cycle

    beta_curve = np.zeros((start_first_rise_at_epoch))
    effective_epochs = n_epochs - start_first_rise_at_epoch
    n_cycles = np.ceil(effective_epochs / period_epochs)

    single_cycle = generate_single_beta_cycle(period_epochs, rise_ratio)

    for c in np.arange(n_cycles):
        beta_curve = np.append(beta_curve, single_cycle)

    return beta_curve[:n_epochs]


def separate_control_elements(tensor, start, end=None, squeeze_result=False):
    """Removes elements from a tensor, either a single element or a slice.

    Args:
    tensor (torch.Tensor): Input tensor.
    start (int): Start index of the slice or index of a single element to be removed.
    end (int, optional): End index of the slice to be removed. If None, removes a single element.

    Returns:
    torch.Tensor: Tensor with elements removed.
    """
    if end is None:
        z_star = torch.cat((tensor[:, :start], tensor[:, start+1:]), dim=1)
        element = tensor[:, start]
    else:
        z_star = torch.cat((tensor[:, :start], tensor[:, end+1:]), dim=1)
        element = tensor[:, start:end]

    if squeeze_result:
        element = element.squeeze()

    return z_star, element

    # if end is None:  # If no end index is given, remove a single element
    #     return torch.cat((tensor[:, :start], tensor[:, start+1:]), dim=1)
    # else:  # If an end index is given, remove a slice
    #     return torch.cat((tensor[:, :start], tensor[:, end+1:]), dim=1)


def convert_continuous_values_to_onehot_vectors(values, adversarial=False, num_classes=10):
    n = values.shape[0]
    # if adversarial:
    #     onehot = torch.ones(n, num_classes)
    # else:
    #     onehot = torch.zeros(n, num_classes)

    indices = (values * num_classes).round().long().unsqueeze(1)
    indices = torch.clamp(indices, max=9)
    # for i in range(n):
    if adversarial:
        onehot = torch.ones(n, num_classes).scatter_(1, indices, 0)
    else:
        onehot = torch.zeros(n, num_classes).scatter_(1, indices, 1)

    return onehot
