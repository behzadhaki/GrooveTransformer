import torch
import numpy as np
import typing
from helpers.Control.density_model_Loader import load_density_model

class full_2D_model(torch.nn.Module):

    def __init__(self, input_layer, encoder, latent, decoder):
        super(full_2D_model, self).__init__()
        self.input_layer = input_layer
        self.encoder = encoder
        self.latent = latent
        self.decoder = decoder

    def forward(self, hvo, density, threshold):
        num_patterns = hvo.shape[0]
        densities = density.repeat(num_patterns)
        threshold = float(threshold.item())
        encoded_input = self.input_layer(hvo, densities)
        memory = self.encoder(encoded_input)
        mu, log_var, _ = self.latent(memory)
        z = self.reparametrize(mu, log_var)
        h_logits, v_logits, o_logits = self.decoder(z)
        h = self.get_hits_activation(h_logits, threshold)
        v = torch.sigmoid(v_logits)
        o = torch.tanh(o_logits) * 0.5
        hvo = torch.cat([h, v, o], dim=-1)
        return hvo

    def reparametrize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = eps * std + mu
        return z

    def get_hits_activation(self, _h, threshold: float=0.5):
        _h = torch.sigmoid(_h)
        h = torch.where(_h > threshold, 1, 0)
        return h

if __name__ == "__main__":
    model_name = "scripted_summer_13_with_params"

    # Load individual components
    input_layer_encoder = torch.jit.load("InputLayerEncoder.pt")
    encoder = torch.jit.load("Encoder.pt")
    latent = torch.jit.load("LatentEncoder.pt")
    decoder = torch.jit.load("Decoder.pt")

    full_model = full_2D_model(input_layer_encoder, encoder, latent, decoder)

    params = sum(p.numel() for p in full_model.parameters())
    print(params)
    # Script + save
    # scripted_model = torch.jit.script(full_model)
    # scripted_model.save((model_name + ".pt"))
