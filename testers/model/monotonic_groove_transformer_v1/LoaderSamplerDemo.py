from model import load_groove_transformer_encoder_model
from model.saved.monotonic_groove_transformer_v1.old.params import model_params
import torch
import numpy as np

# Model path and model_param dictionary
model_name = "colorful_sweep_41"

# 4. LOAD MODEL
# load state_dict and params_dict from different locations
# exp1. model state_dict stored as .model and params_dict as a dictionary
model_path = f"model/saved/monotonic_groove_transformer_v1/old/{model_name}.model"
params_dict = model_params[model_name]
GrooveTransformer = load_groove_transformer_encoder_model(model_path, params_dict)

# exp2. model state_dict stored as .bz2model and params_dict in a json file
model_path = f"model/saved/monotonic_groove_transformer_v1/latest/{model_name}.pth"
params_dict = f"model/saved/monotonic_groove_transformer_v1/latest/{model_name}.json"
GrooveTransformer = load_groove_transformer_encoder_model(model_path, params_dict)

# exp3. loading a self contained model
model_path = f"model/saved/monotonic_groove_transformer_v1/latest/{model_name}.pth"
GrooveTransformer = load_groove_transformer_encoder_model(model_path)


# 5. Sampling from the loaded model
#   5.1 Grab a random groove from the test set
from data import load_gmd_hvo_sequences
test_set = load_gmd_hvo_sequences(
    "data/gmd/resources/storedDicts/groove_2bar-midionly.bz2pickle", "gmd", "data/dataset_json_settings/4_4_Beats_gmd.json", [4],
    "ROLAND_REDUCED_MAPPING", "train")
input_hvo_seq = test_set[np.random.randint(0, len(test_set))]
input_groove_hvo = torch.tensor(input_hvo_seq.flatten_voices(), dtype=torch.float32)

#   5.2 Pass groove to model and sample
from model import get_prediction
voice_thresholds = [0.5] * 9
voice_max_count_allowed = [32] * 9
output_hvo = get_prediction(GrooveTransformer, input_groove_hvo, voice_thresholds,
                            voice_max_count_allowed, return_concatenated=True)


#  5. 3. Plot input/groove/output piano rolls
from hvo_sequence.hvo_seq import zero_like
input = input_hvo_seq
groove = zero_like(input_hvo_seq)                        # create template for groove hvo_sequence object
groove.hvo = input_groove_hvo.cpu().detach().numpy()                     # add score
output = zero_like(input_hvo_seq)                        # create template for output hvo_sequence object
output.hvo = output_hvo[0, :, :].cpu().detach().numpy()                    # add score

input.to_html_plot("in.html", show_figure=True)
groove.to_html_plot("groove.html", show_figure=True)
output.to_html_plot("output.html", show_figure=True)

