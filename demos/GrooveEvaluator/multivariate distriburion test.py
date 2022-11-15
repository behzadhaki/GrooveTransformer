import torch

from model.Base.shared_model_components_VAE import *

# import math

encoder_layer = torch.nn.TransformerEncoderLayer(d_model=100, nhead=10, batch_first=True)
src = torch.rand(20, 32, 100)
# out = encoder_layer(Base)
print(src.shape)

encodermodule = Encoder(d_model=100, nhead=10, dim_feedforward=2048, dropout=0.4, num_encoder_layers=2)
vaeEncode = reparameterize( max_len=32, d_model= 100, latent_dim=  (32*100))
deco_in = deco_imput(max_len=32, d_model= 100, latent_dim=  (32*100))
srcenco = encodermodule(src)

## vae encode
# N = 10
# d_model = 100
# latent_dim = (10 * 100)
# fc_mu = torch.nn.Linear(int(N * d_model), latent_dim)
# fc_var = torch.nn.Linear(int(N * d_model), latent_dim)
#
# result = torch.flatten(srcenco, start_dim=1)
# # Split the result into mu and var components
# # of the latent Gaussian distribution
# mu = fc_mu(result)
# log_var = fc_var(result)
#
# std = torch.exp(0.5 * log_var)
# eps = torch.randn_like(std)
#
_,_,out = vaeEncode(srcenco)#eps * std + mu
print(out.shape)




## fix imput decoder
#
outreshape = deco_in(out) #out.view(-1, N, d_model)
print( outreshape.shape)