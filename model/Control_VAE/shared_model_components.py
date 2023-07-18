import torch
from copy import deepcopy

def create_layers(module, N):
    # User to stack multiple Encoder layers for In-Attention mechanism
    return torch.nn.ModuleList([deepcopy(module) for i in range(N)])

class InAttentionEncoder(torch.nn.Module):
    """
    This function is a modified version of the In-Attention Transformer from Musemorphose paper
    https://arxiv.org/abs/2105.04090
    We create a stack of num_encoder_layers transformer encoders. Separately, we expect control parameters to be
    a vector of num_params length. This vector is expanded to d_model with a learnable Linear layer.
    Then, at each Layer, it is summed with the source (which initially comes from sampling VAE latent space)
    and fed through the next iteration of multi-head self-attention.
    """
    def __init__(self, d_model, n_head, dim_feedforward, dropout, num_encoder_layers, num_params):
        super(InAttentionEncoder, self).__init__()
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model, n_head, dim_feedforward, dropout)
        self.layers = create_layers(encoder_layer, num_encoder_layers)
        self.parameter_projection = torch.nn.Linear(num_params, d_model)

    def forward(self, x, parameters):
        parameters = self.parameter_projection(parameters)
        for mod in self.layers:
            x += parameters
            x = mod(x)
        return x