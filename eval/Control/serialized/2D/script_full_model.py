import os
import torch
import wandb
from model import Density2D


def load_density_2d_model(model_path, params_dict=None, is_evaluating=True, device=None):
    try:
        if device is not None:
            loaded_dict = torch.load(model_path, map_location=device)
        else:
            loaded_dict = torch.load(model_path)
    except:
        loaded_dict = torch.load(model_path, map_location=torch.device('cpu'))

    if params_dict is None:
        if 'params' in loaded_dict:
            params_dict = loaded_dict['params']
        else:
            raise Exception(f"Could not instantiate model as params_dict is not found. "
                            f"Please provide a params_dict either as a json path or as a dictionary")

    if isinstance(params_dict, str):
        import json
        with open(params_dict, 'r') as f:
            params_dict = json.load(f)

    params_dict['n_params'] = 1
    params_dict['add_params'] = False
    model = Density2D(params_dict)
    model.load_state_dict(loaded_dict["model_state_dict"])
    if is_evaluating:
        model.eval()

    return model


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

EPOCH = 240
VERSION = 135
RUN_NAME = "summer_sweep_13"
ARTIFACT_PATH = f"mmil_julian/Control 1D/model_epoch_{EPOCH}:v{VERSION}"
MODEL_NAME = "serialized_summer_sweep"

if __name__ == "__main__":

    run = wandb.init()


    EPOCH = ARTIFACT_PATH.split("model_epoch_")[-1].split(":")[0]

    artifact = run.use_artifact(ARTIFACT_PATH, type='model')
    artifact_dir = artifact.download()
    model = load_density_2d_model(os.path.join(artifact_dir, f"{EPOCH}.pth"))

    model.serialize(os.getcwd())


    # Load individual components
    input_layer_encoder = torch.jit.load("InputLayerEncoder.pt")
    encoder = torch.jit.load("Encoder.pt")
    latent = torch.jit.load("LatentEncoder.pt")
    decoder = torch.jit.load("Decoder.pt")

    full_model = full_2D_model(input_layer_encoder, encoder, latent, decoder)
    params = sum(p.numel() for p in full_model.parameters())
    print(params)

    # Script + save
    scripted_model = torch.jit.script(full_model)
    scripted_model.save((MODEL_NAME + ".pt"))
