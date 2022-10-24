from model.modelLoadesSamplers import load_groove_transformer_encoder_model
from model.saved.monotonic_groove_transformer_v1.params import model_params
import torch
import numpy as np

# Model path and model_param dictionary
model_name = "colorful_sweep_41"
model_path = f"model/saved/monotonic_groove_transformer_v1/{model_name}.model"
model_param = model_params[model_name]

# 4. LOAD MODEL
GrooveTransformer = load_groove_transformer_encoder_model(model_path, model_param)
checkpoint = torch.load(model_path, map_location=model_param['device'])

# 5. Sampling from the loaded model
#   5.1 Grab a random groove from the test set
from data.dataLoaders import load_gmd_hvo_sequences
test_set = load_gmd_hvo_sequences(
    "data/gmd/resources/storedDicts/groove_2bar-midionly.bz2pickle", "gmd", "data/dataset_json_settings/4_4_Beats_gmd.json", [4],
    "ROLAND_REDUCED_MAPPING", "train")
input_hvo_seq = test_set[np.random.randint(0, len(test_set))]
input_groove_hvo = torch.tensor(input_hvo_seq.flatten_voices(), dtype=torch.float32)

#   5.2 Pass groove to model and sample
from model.modelLoadesSamplers import get_prediction
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

