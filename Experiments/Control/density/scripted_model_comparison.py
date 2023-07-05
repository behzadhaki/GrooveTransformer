import torch
from data import GrooveDataSet_Density
from model import Density2D
from copy import deepcopy
import os
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

if __name__ == "__main__":

    os.chdir("../../../")

    original_model_fp = "eval/Control/serialized/2D/original_summer_epoch290.pth"
    scripted_model_fp = "eval/Control/serialized/2D/scripted_summer__13.pt"

    original_model = load_density_2d_model(original_model_fp)
    scripted_model = torch.jit.load(scripted_model_fp)

    idx = 10
    dataset = GrooveDataSet_Density(
        dataset_setting_json_path="data/dataset_json_settings/4_4_BeatsAndFills_gmd.json",
        subset_tag="test",
        max_len=32)

    os.chdir("experiments/Control/density/midi_2d/comparison")
    # Get inputs and HVO sequences
    gt_sequence = dataset.get_hvo_sequences_at(idx)
    tapped_hvo_seq = deepcopy(gt_sequence)
    model_original_seq = deepcopy(gt_sequence)
    model_scripted_seq = deepcopy(gt_sequence)

    tapped_sequence, _, d, _, _, _ = dataset[idx]
    input = torch.unsqueeze(tapped_sequence, dim=0)
    density = torch.unsqueeze(torch.tensor(0.6), dim=0)

    gt_sequence.save_hvo_to_midi(filename="original.mid")

    tapped_sequence = tapped_sequence.numpy()
    tapped_hvo_seq.hvo = tapped_sequence
    tapped_hvo_seq.save_hvo_to_midi(filename="tapped_input.mid")

    original_output, _, _, _ = original_model.predict(input, density, return_concatenated=True)
    original_hvo = torch.squeeze(original_output, dim=0).detach().numpy()
    model_original_seq.hvo = original_hvo
    model_original_seq.save_hvo_to_midi(filename="model_original.mid")

    scripted_output = scripted_model.forward(input)
    scripted_hvo = torch.squeeze(scripted_output, dim=0).detach().numpy()
    model_scripted_seq.hvo = scripted_hvo
    model_scripted_seq.save_hvo_to_midi(filename="model_scripted.mid")




