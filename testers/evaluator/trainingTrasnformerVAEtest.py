import torch
from model.src.BasicGrooveTransformer_VAE import *
from helpers.BasicGrooveTransformer_train_VAE import *

### encoder params
nhead_enc = 3
nhead_dec = 3
d_model_enc = 27
d_model_dec = 30
embedding_size_src = 27
embedding_size_tgt = 27
dim_feedforward = 2048
dropout = 0.4
num_encoder_layers = 2
num_decoder_layers = 2
max_len = 32
device = 0

latent_dim = int((max_len * d_model_enc)/4)

### call VAE encoder

groove_transformer = GrooveTransformerEncoderVAE(d_model_enc, d_model_dec, embedding_size_src, embedding_size_tgt,
                 nhead_enc, nhead_dec, dim_feedforward, dropout, num_encoder_layers, latent_dim,
                 num_decoder_layers, max_len, device)

optimizer = torch.optim.Adam(groove_transformer.parameters(), lr=1e-4)
inputs = torch.rand(20, max_len, embedding_size_src)

# PYTORCH LOSS FUNCTIONS
# BCE used for hit loss
bce_fn = torch.nn.BCEWithLogitsLoss(reduction='none')
# MSE used for velocities and offsets losses
mse_fn = torch.nn.MSELoss(reduction='none')
hit_loss_penalty = 0.1

# run one epoch
groove_transformer.train()  # train mode
loss = 0

optimizer.zero_grad()
# forward + backward + optimize
outputs = groove_transformer(inputs)
#loss = calculate_loss_VAE(outputs, labels)

loss, training_accuracy, training_perplexity, bce_h, mse_v, mse_o = calculate_loss_VAE(outputs, inputs, bce_fn, mse_fn,
                                                                                    hit_loss_penalty)

loss.backward()
optimizer.step()