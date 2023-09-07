import torch
import json
from model import GrooveControl_VAE


def load_vaeder_model(model_path, params_dict=None, is_evaluating=True, device=None,
                      genre_json_path=None):
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

            if isinstance(genre_json_path, str):
                with open(genre_json_path, 'r') as f:
                    genre_dict = json.load(f)
                    params_dict['genre_dict'] = genre_dict

        else:
            raise Exception(f"Could not instantiate model as params_dict is not found. "
                            f"Please provide a params_dict either as a json path or as a dictionary")

    if isinstance(params_dict, str):
        with open(params_dict, 'r') as f:
            params_dict = json.load(f)

    model = GrooveControl_VAE(params_dict)
    model.load_state_dict(loaded_dict["model_state_dict"])
    if is_evaluating:
        model.eval()

    return model

