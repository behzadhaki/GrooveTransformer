import torch
from model.src.shared_model_components_VAE import *
from model.src.BasicGrooveTransformer_VAE import *


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

src = torch.rand(20, max_len, embedding_size_src)

### call VAE encoder

encoVae = GrooveTransformerEncoderVAE(d_model_enc, d_model_dec, embedding_size_src, embedding_size_tgt,
                                      nhead_enc, nhead_dec, dim_feedforward, dropout, num_encoder_layers, latent_dim,
                                      num_decoder_layers, max_len, device)

out = encoVae(src)

print(len(out), out[1].shape)


## training