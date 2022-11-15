import torch

torch.backends.cudnn.enabled = False
from model.Base.BasicGrooveTransformer_VAE import *
from helpers.BasicGrooveTransformer_train_VAE import *

### encoder params
nhead_enc = 1
nhead_dec = 1
d_model_enc = 2
d_model_dec = 2
embedding_size_src = 27
embedding_size_tgt = 27
dim_feedforward = 2048
dropout = 0.4
num_encoder_layers = 1
num_decoder_layers = 1
max_len = 32
device = 0

latent_dim = 16#int((max_len * d_model_enc)/4)

groove_transformer = GrooveTransformerEncoderVAE(d_model_enc, d_model_dec, embedding_size_src, embedding_size_tgt,
                 nhead_enc, nhead_dec, dim_feedforward, dropout, num_encoder_layers, latent_dim,
                 num_decoder_layers, max_len, device, bce = True)


# =================================================================================================
# Load dataset as torch.utils.data.Dataset
from data.src.dataLoaders import MonotonicGrooveDataset

# load dataset as torch.utils.data.Dataset
training_dataset = MonotonicGrooveDataset(
    dataset_setting_json_path="data/dataset_json_settings/4_4_Beats_gmd.json",
    subset_tag="train",
    max_len=32,
    tapped_voice_idx=2,
    collapse_tapped_sequence=False)


# use the above dataset in the training pipeline, you need to use torch.utils.data.DataLoader
from torch.utils.data import DataLoader
train_dataloader = DataLoader(training_dataset, batch_size=10, shuffle=True) #int(32/4)

# =================== optimezer and loss ============


optimizer = torch.optim.Adam(groove_transformer.parameters(), lr=1e-4)
#inputs = torch.rand(20, max_len, embedding_size_src)
# PYTORCH LOSS FUNCTIONS
# BCE used for hit loss
bce_fn = torch.nn.BCEWithLogitsLoss(reduction='none')
# MSE used for velocities and offsets losses
mse_fn = torch.nn.MSELoss(reduction='none')
hit_loss_penalty = 0.1

LOSS = []

epochs = 1
groove_transformer.train()  # train mode

for epoch in range(epochs):
    # in each epoch we iterate over the entire dataset
    for batch_count, (inputs, outputs, indices) in enumerate(train_dataloader):
        print(f"Epoch {epoch} - Batch #{batch_count} - inputs.shape {inputs.shape} - "
              f"outputs.shape {outputs.shape} - indices.shape {indices.shape} ")

        inputs = inputs.float()
        optimizer.zero_grad()

        # forward + backward + optimize
        output_net = groove_transformer(inputs)
        # loss = calculate_loss_VAE(outputs, labels)

        loss, losses = calculate_loss_VAE(output_net, inputs, bce_fn, mse_fn, hit_loss_penalty,
                                          dice = True, bce = True)
        loss.backward()
        optimizer.step()

        LOSS.append(loss.detach().numpy())
    #test eval
