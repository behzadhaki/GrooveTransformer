import sys
sys.path.insert(1, "/")
sys.path.insert(1, "../")
import os
import wandb
import math
import random
import torch, torchvision
from model.src.BasicGrooveTransformer_VAE import *
from helpers.BasicGrooveTransformer_train_VAE import *
# Load dataset as torch.utils.data.Dataset
from data.src.dataLoaders import MonotonicGrooveDataset
from torch.utils.data import DataLoader


hyperparameter_defaults = dict(
    nhead_enc=2,
    nhead_dec=2,
    d_model_enc=16,
    d_model_dec=16,
    embedding_size_src=27,
    embedding_size_tgt=27,
    dim_feedforward=32,
    dropout=0.3,
    loss_hit_penalty_multiplier=0.5,
    num_encoder_layers=2,
    num_decoder_layers=2,
    max_len=32,
    device=0,
    latent_dim=32,
    epochs=1,
    batch_size=16,
    lr=1e-3,
    bce=True,
    dice=True)

# sweep_config['parameters'] = parameters_dict
wandb_run = wandb.init(config=hyperparameter_defaults, project="transformerVAE1", anonymous="allow")
# this config will be set by Sweep Controller
config = wandb.config




if __name__ == "__main__":
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    # If called by wandb.agent, as below,

    # BCE used for hit loss
    bce_fn = torch.nn.BCEWithLogitsLoss(reduction='none')
    # MSE used for velocities and offsets losses
    mse_fn = torch.nn.MSELoss(reduction='none')
    hit_loss_penalty = config.loss_hit_penalty_multiplier

    groove_transformer_cpu = GrooveTransformerEncoderVAE(config.d_model_enc, config.d_model_dec,
                                                     config.embedding_size_src,
                                                     config.embedding_size_tgt, config.nhead_enc,
                                                     config.nhead_dec,
                                                     config.dim_feedforward, config.dropout,
                                                     config.num_encoder_layers,
                                                     config.latent_dim, config.num_decoder_layers,
                                                     config.max_len,
                                                     config.device,
                                                     config.bce)
    groove_transformer = groove_transformer_cpu.cuda() if torch.cuda.is_available() else groove_transformer_cpu
    optimizer = torch.optim.Adam(groove_transformer.parameters(), lr=1e-4)
    batch_size = config.batch_size
    train_dataloader = DataLoader(training_dataset, batch_size=config.batch_size, shuffle=True)

    for epoch in range(config.epochs):
        groove_transformer.train()  # train mode
        for batch_count, (inputs, outputs, indices) in enumerate(train_dataloader):
            print(f"Epoch {epoch} - Batch #{batch_count} - inputs.shape {inputs.shape} - "
                  f"outputs.shape {outputs.shape} - indices.shape {indices.shape} ")

            # inputs = inputs.clone().detach()#torch.tensor(inputs.float())
            inputs.to(device)
            # outputs = torch.tensor(outputs.float())
            outputs.to(device)
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
        output_test = test_dataset.outputs
        output_test.to(device)
        inputs_test = test_dataset.inputs
        inputs_test.to(device)
        output_net_test = groove_transformer(inputs_test)
        val_loss, val_losses = calculate_loss_VAE(output_net_test, output_test, bce_fn, mse_fn,
                                                  hit_loss_penalty, config.bce, config.dice)
        val_metrics = {"val/train_loss": loss.detach().numpy(),
                       "val/h_loss": losses['loss_h'],
                       "val/v_loss": losses['loss_v'],
                       "val/o_loss": losses['loss_o'],
                       "val/KL_loss": losses['KL_loss']
                       }

        wandb.log({**metrics, **val_metrics})
            #
            # finally:
            # print("\nDone!")
            # wandb.finish()

