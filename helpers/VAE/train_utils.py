
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
    if isinstance(vel_loss_function, torch.nn.MSELoss):
        loss_v = vel_loss_function(torch.sigmoid(vel_logits), vel_targets) * hit_loss_penalty_mat
    elif isinstance(vel_loss_function, torch.nn.BCEWithLogitsLoss):
        loss_v = vel_loss_function(vel_logits, vel_targets) * hit_loss_penalty_mat
    else:
        raise NotImplementedError(f"the vel_loss_function {vel_loss_function} is not implemented")

    return torch.sum(loss_v, dim=2).mean()


def calculate_kld_loss(mu, log_var):
    return torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2. - log_var.exp(), dim=1), dim=0)


def calculate_offset_loss(offset_logits, offset_targets, offset_loss_function, hit_loss_penalty_mat):
    if isinstance(offset_loss_function, torch.nn.MSELoss):
        loss_o = offset_loss_function(torch.tanh(offset_logits), offset_targets) * hit_loss_penalty_mat
    elif isinstance(offset_loss_function, torch.nn.BCEWithLogitsLoss):
        # here the offsets MUST be in the range of [0, 1]. Our existing targets are from [-0.5, 0.5].
        # So we need to shift them to [0, 1] range by adding 0.5
        loss_o = offset_loss_function(offset_logits, offset_targets+0.5) * hit_loss_penalty_mat
    else:
        raise NotImplementedError(f"the offset_loss_function {offset_loss_function} is not implemented")

    return torch.sum(loss_o, dim=2).mean()



def hits_accuracy(hits_predicted, hits_target):

    y_true = np.reshape(hits_target, (-1,))
    y_pred = np.reshape(hits_predicted, (-1,))

    acc = np.sum(np.equal(y_true.astype(int), y_pred.astype(int))) / len(y_true)
    return acc




def initialize_model(params):
    model_params = params["model"]
    training_params = params["training"]
    load_model = params["load_model"]

    if model_params['encoder_only']:
        groove_transformer = GrooveTransformerEncoder(model_params['d_model'], model_params['embedding_size_src'],
                                                      model_params['embedding_size_tgt'], model_params['n_heads'],
                                                      model_params['dim_feedforward'], model_params['dropout'],
                                                      model_params['num_encoder_layers'],
                                                      model_params['max_len'], model_params['device'])
    else:
        groove_transformer = GrooveTransformer(model_params['d_model'],
                                               model_params['embedding_size_src'],
                                               model_params['embedding_size_tgt'], model_params['n_heads'],
                                               model_params['dim_feedforward'], model_params['dropout'],
                                               model_params['num_encoder_layers'],
                                               model_params['num_decoder_layers'],
                                               model_params['max_len'], model_params['device'])

    groove_transformer.to(model_params['device'])
    optimizer = torch.optim.Adam(groove_transformer.parameters(), lr=training_params['learning_rate']) if \
        model_params['optimizer'] == 'adam' else torch.optim.SGD(groove_transformer.parameters(),
                                                                 lr=training_params['learning_rate'])
    epoch = 0

    if load_model is not None:

        # If model was saved locally
        if load_model['location'] == 'local':

            last_checkpoint = 0
            # From the file pattern, get the file extension of the saved model (in case there are other files in dir)
            file_extension_pattern = re.compile(r'\w+')
            file_ext = file_extension_pattern.findall(load_model['file_pattern'])[-1]

            # Search for all continuous digits in the file name
            ckpt_pattern = re.compile(r'\d+')
            ckpt_filename = ""

            # Iterate through files in directory, find last checkpoint
            for root, dirs, files in os.walk(load_model['dir']):
                for name in files:
                    if name.endswith(file_ext):
                        checkpoint_epoch = int(ckpt_pattern.findall(name)[-1])
                        if checkpoint_epoch > last_checkpoint:
                            last_checkpoint = checkpoint_epoch
                            ckpt_filename = name

            # Load latest checkpoint found
            if last_checkpoint > 0:
                path = os.path.join(load_model['dir'], ckpt_filename)
                checkpoint = torch.load(path)

        # If restoring from wandb
        elif load_model['location'] == 'wandb':
            model_file = wandb.restore(load_model['file_pattern'].format(load_model['run'],
                                                                         load_model['epoch']),
                                       run_path=load_model['dir'])
            checkpoint = torch.load(model_file.name)

        groove_transformer.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']

    return groove_transformer, optimizer, epoch


def batch_loop(dataloader_, groove_transformer_vae, hit_loss_fn, velocity_loss_fn,
               offset_loss_fn, loss_hit_penalty_multiplier, device, optimizer=None):
    # DONT PASS OPTIMIZER FOR EVALUATION



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