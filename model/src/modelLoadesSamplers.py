import torch
from model.src.BasicGrooveTransformer import GrooveTransformerEncoder


# --------------------------------------------------------------------------------
# ------------             Model Loaders                     ---------------------
# --------------------------------------------------------------------------------

def load_groove_transformer_encoder_model(model_path, params_dict=None, eval=True, device=None):
    ''' Load a GrooveTransformerEncoder model from a given path

    Args:
        model_path (str): path to the model
        params_dict (None, dict, or json path): dictionary containing the parameters of the model
        eval (bool): if True, the model is set to eval mode
        device (None or torch.device): device to load the model to (if cpu, the model is loaded to cpu)

    Returns:
        model (GrooveTransformerEncoder): the loaded model
    '''

    try:
        if device is not None:
            loaded_dict = torch.load(model_path, map_location=device)
        else:
            loaded_dict = torch.load(model_path)
    except:
        loaded_dict = torch.load(model_path, map_location=torch.device('cpu'))
        print(f"Model was loaded to cpu!!!")

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


    model = GrooveTransformerEncoder(**params_dict)
    model.load_state_dict(loaded_dict["model_state_dict"])
    if eval:
        model.eval()

    return model



# --------------------------------------------------------------------------------
# ------------             Model SAMPLING                     ---------------------
# --------------------------------------------------------------------------------

def get_prediction(trained_model, input_tensor, voice_thresholds, voice_max_count_allowed, return_concatenated = False, sampling_mode=0):
    trained_model.eval()
    with torch.no_grad():

        if isinstance(trained_model, GrooveTransformerEncoder):
            _h, v, o = trained_model.forward(input_tensor)  # Nx32xembedding_size_src/3,Nx32xembedding_size_src/3,Nx32xembedding_size_src/3

            _h = torch.sigmoid(_h)
            h = torch.zeros_like(_h)


            for ix, (thres, max_count) in enumerate(zip(voice_thresholds, voice_max_count_allowed)):
                max_indices = torch.topk(_h[:, :, ix], max_count).indices[0]
                h[:, max_indices, ix]  = _h[:, max_indices, ix]
                h[:, :, ix] = torch.where(h[:, :, ix] > thres, 1, 0)

            if return_concatenated:
                return torch.concat((h, v, o), -1)
            else:
                return h, v, 0

        else:
            print(f"NO SAMPLERS IMPLEMENTED FOR THE GIVEN MODEL OF TYPE {type(trained_model)}")
            return None

