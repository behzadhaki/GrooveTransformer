import wandb
import math
import random
import torch, torchvision
from model.Base.BasicGrooveTransformer_VAE import *
from helpers.BasicGrooveTransformer_train_VAE import *
# Load dataset as torch.utils.data.Dataset
from data.dataLoaders import MonotonicGrooveDataset
from torch.utils.data import DataLoader

wandb.init(project="VAE_tester1")
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



# Launch 5 experiments, trying different dropout rates

# PYTORCH LOSS FUNCTIONS
# BCE used for hit loss
bce_fn = torch.nn.BCEWithLogitsLoss(reduction='none')
# MSE used for velocities and offsets losses
mse_fn = torch.nn.MSELoss(reduction='none')
hit_loss_penalty = 0.49

for drop in [0.4, 0.3, 0.1]:
    # 🐝 initialise a wandb run
    wandb.init(
        project="VAE_tester1",
        config={
            'nhead_enc': 3,
            'nhead_dec' : 3,
            'd_model_enc': 27,
            'd_model_dec' : 30,
            'embedding_size_src': 27,
            'embedding_size_tgt': 27,
            'dim_feedforward': 2048,
            'dropout': drop,
            'num_encoder_layers': 2,
            'num_decoder_layers': 2,
            'max_len': 32,
            'device': 0,
            'latent_dim': int((32 * 27) / 4),
            "epochs": 10,
            "batch_size": 32,
            "lr": 1e-3,
            })

    config = wandb.config

    ### call VAE encoder

    groove_transformer = GrooveTransformerEncoderVAE(config.d_model, config.d_model_dec, config.embedding_size,
                                                     config.embedding_size_tgt, config.n_head, config.nhead_dec,
                                                     config.dim_feedforward, config.dropout, config.num_encoder_layers,
                                                     config.latent_dim, config.num_decoder_layers, config.max_len,
                                                     config.device)
    optimizer = torch.optim.Adam(groove_transformer.parameters(), lr=1e-4)
    batch_size = config.batch_size
    train_dataloader = DataLoader(training_dataset, batch_size= config.batch_size, shuffle=True)

    for epoch in range(config.epochs):
        # in each epoch we iterate over the entire dataset
        for batch_count, (inputs, outputs, indices) in enumerate(train_dataloader):

            print(f"Epoch {epoch} - Batch #{batch_count} - inputs.shape {inputs.shape} - "
                  f"outputs.shape {outputs.shape} - indices.shape {indices.shape} ")

            inputs = inputs.float()
            # run one epoch
            groove_transformer.train()  # train mode

            # forward + backward + optimize
            output_net = groove_transformer(inputs)
            # loss = calculate_loss_VAE(outputs, labels)

            loss, losses = calculate_loss_VAE(output_net, inputs, bce_fn, mse_fn, hit_loss_penalty)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            metrics = {"train/train_loss": loss.detach().numpy(),
                       "train/epoch": epoch,
                       "train/bce_h_loss": losses['bce_h'],
                       "train/mse_v_loss": losses['mse_v'],
                       "train/KL_loss": losses['KL_loss'],
                       }
        #test_dataset =
        wandb.log({**metrics })
    wandb.finish()
