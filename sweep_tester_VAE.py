import sys
import wandb
import math
import random
import torch, torchvision
from model.src.BasicGrooveTransformer_VAE import *
from helpers.BasicGrooveTransformer_train_VAE import *
# Load dataset as torch.utils.data.Dataset
from data.src.dataLoaders import MonotonicGrooveDataset
from torch.utils.data import DataLoader
import os

os.environ["WANDB_NOTEBOOK_NAME"] = "testers/evaluator/VAE_testers/sweep_tester_VAE.py"
sweep_config = {
            'method': 'random',
            'metric': {
                'name': 'loss',
                'goal': 'minimize'
            }}

parameters_dict = {
            'nhead_enc': {
                        'values': [1, 2, 4, 8, 16]},
            'nhead_dec': {
                        'values': [1, 2, 4, 8, 16]},
            'd_model_enc': {
                        'values': [16, 32, 64, 128, 256, 512]},
            'd_model_dec': {
                        'values': [16, 32, 64, 128, 256, 512]},
            'embedding_size_src': {
                        'value': 27},
            'embedding_size_tgt': {
                        'value': 27},
            'dim_feedforward': {
                        'values': [16, 32, 64, 128, 256, 512]},
            'dropout': {
                 'distribution': 'uniform',
                 'min': 0.1,
                 'max': 0.3
            },
            'loss_hit_penalty_multiplier': {
                 'distribution': 'uniform',
                 'min': 0,
                 'max': 1
            },
            'num_encoder_layers': {
                        'values': [6, 8, 10, 12]},
            'num_decoder_layers': {
                        'values': [6, 8, 10, 12]},
            'max_len': {
                    'value': 32},
            'device': {
                    'value': 0},
            'latent_dim': {
                    'value': int((32 * 27) / 4)},
            "epochs": {
                    'value': 100},
            "batch_size": {
                    'values': [16, 32]},
            "lr": {
                    'values': [1e-3, 1e-4]},
            "bce": {
                'values': [True , False]},
            "dice": {
                'values': [True , False]},
            }

sweep_config['parameters'] = parameters_dict

# wandb_run = wandb.init(config=sweep_config, project="VAE_sweep1")

sweep_id = wandb.sweep(sweep_config, project="VAE_sweep1")
# wandb.init(
#         project="VAE_sweep1",
#         config=sweep_config )


# load dataset as torch.utils.data.Dataset
training_dataset = MonotonicGrooveDataset(
    dataset_setting_json_path="data/dataset_json_settings/4_4_Beats_gmd.json",
    subset_tag="train",
    max_len=32,
    tapped_voice_idx=2,
    collapse_tapped_sequence=False)
test_dataset = MonotonicGrooveDataset(
    dataset_setting_json_path="data/dataset_json_settings/4_4_Beats_gmd.json",
    subset_tag="test",
    max_len=32,
    tapped_voice_idx=2,
    collapse_tapped_sequence=False)

## train function

def train(config=None):
    # Initialize a new wandb run
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    with wandb.init(config=config):
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config
        # BCE used for hit loss
        bce_fn = torch.nn.BCEWithLogitsLoss(reduction='none')
        # MSE used for velocities and offsets losses
        mse_fn = torch.nn.MSELoss(reduction='none')
        hit_loss_penalty = config.loss_hit_penalty_multiplier

        groove_transformer = GrooveTransformerEncoderVAE(config.d_model_enc, config.d_model_dec,
                                                         config.embedding_size_src,
                                                         config.embedding_size_tgt, config.nhead_enc, config.nhead_dec,
                                                         config.dim_feedforward, config.dropout,
                                                         config.num_encoder_layers,
                                                         config.latent_dim, config.num_decoder_layers, config.max_len,
                                                         config.device,
                                                         config.bce)
        groove_transformer.to(device)
        optimizer = torch.optim.Adam(groove_transformer.parameters(), lr=1e-4)
        batch_size = config.batch_size
        train_dataloader = DataLoader(training_dataset, batch_size=config.batch_size, shuffle=True)

        for epoch in range(config.epochs):
            groove_transformer.train()  # train mode
            for batch_count, (inputs, outputs, indices) in enumerate(train_dataloader):
                print(f"Epoch {epoch} - Batch #{batch_count} - inputs.shape {inputs.shape} - "
                      f"outputs.shape {outputs.shape} - indices.shape {indices.shape} ")

                inputs = torch.tenosr(inputs.float()).to(device)
                outputs = torch.tenosr(outputs.float()).to(device)
                # run one epoch


                # forward + backward + optimize
                output_net = groove_transformer(inputs)
                # loss = calculate_loss_VAE(outputs, labels)

                loss, losses = calculate_loss_VAE(output_net, outputs, bce_fn, mse_fn,
                                                  hit_loss_penalty, config.bce, config.dice)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                metrics = {"train/train_loss": loss.detach().numpy(),
                           "train/epoch": epoch,
                           "train/h_loss": losses['loss_h'],
                           "train/v_loss": losses['loss_v'],
                           "train/o_loss": losses['loss_o'],
                           "train/KL_loss": losses['KL_loss'],
                           }

            groove_transformer.eval()
            output_test = test_dataset.outputs.to(device)
            inputs_test = test_dataset.inputs.to(device)

            output_net_test = groove_transformer(inputs_test)
            val_loss, val_losses = calculate_loss_VAE(output_net_test, output_test, bce_fn, mse_fn,
                                              hit_loss_penalty, config.bce, config.dice)
            val_metrics = {"val/train_loss": loss.detach().numpy(),
                           "val/h_loss": losses['loss_h'],
                           "val/v_loss": losses['loss_v'],
                           "val/o_loss": losses['loss_o'],
                           "val/KL_loss": losses['KL_loss']
                           }


            wandb.log({**metrics, **val_metrics })






# run agent


wandb.agent(sweep_id, train, count=5)