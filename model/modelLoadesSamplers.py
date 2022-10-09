import torch
from model.src.BasicGrooveTransformer import GrooveTransformerEncoder


# --------------------------------------------------------------------------------
# ------------             Model Loaders                     ---------------------
# --------------------------------------------------------------------------------

def load_groove_transformer_encoder_model(model_path, params_dict):
    """ Loads a GrooveTransformerEncoder stored at a specific path

    :param model_path: path of a trained model ending in .model or .bz2model
    :param params_dict: a dictionary of parameters including the following necessary
    fields: {d_model, embedding_sz, n_heads, dim_ff, dropout, n_layers, max_len, device}
    :return: the loaded GrooveTransformerEncoder instance
    """

    # load checkpoint
    checkpoint = torch.load(model_path, map_location=params_dict['device'])

    # Initialize model
    groove_transformer = GrooveTransformerEncoder(params_dict['d_model'],
                                                  params_dict['embedding_sz'],
                                                  params_dict['embedding_sz'],
                                                  params_dict['n_heads'],
                                                  params_dict['dim_ff'],
                                                  params_dict['dropout'],
                                                  params_dict['n_layers'],
                                                  params_dict['max_len'],
                                                  params_dict['device'])

    # Load model and put in evaluation mode
    groove_transformer.load_state_dict(checkpoint['model_state_dict'])
    groove_transformer.eval()

    return groove_transformer


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

