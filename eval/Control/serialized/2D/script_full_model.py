import torch
import typing

class full_2D_model(torch.nn.Module):

    def __init__(self, input_layer, encoder, latent, decoder):
        super(full_2D_model, self).__init__()
        self.input_layer = input_layer
        self.encoder = encoder
        self.latent = latent
        self.decoder = decoder

    def forward(self, hvo):
        num_patterns = hvo.shape[0]
        fake_densities = torch.full((num_patterns,), 0.5)
        encoded_input = self.input_layer(hvo, fake_densities)
        memory = self.encoder(encoded_input)
        mu, log_var, _ = self.latent(memory)
        z = self.reparametrize(mu, log_var)
        h_logits, v_logits, o_logits = self.decoder(z)
        h = self.get_hits_activation(h_logits)
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
    model_name = "2D_summer_sweep_13"

    # Load individual components
    input_layer_encoder = torch.jit.load("InputLayerEncoder.pt")
    encoder = torch.jit.load("Encoder.pt")
    latent = torch.jit.load("LatentEncoder.pt")
    decoder = torch.jit.load("Decoder.pt")
    print(input_layer_encoder)

    full_model = full_2D_model(input_layer_encoder, encoder, latent, decoder)

    # Test with some inputs
    # input_hvo = torch.rand((3, 32, 27))
    # input_density = torch.tensor([0.25, 0.7, 0.4592])
    # hvo = full_model.forward(input_hvo)
    # print(hvo.shape)

    # Script + save
    scripted_model = torch.jit.script(full_model)
    scripted_model.save((model_name + ".pt"))
