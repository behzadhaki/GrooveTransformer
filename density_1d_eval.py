import torch
import numpy as np
import wandb
from hvo_sequence import HVO_Sequence
from hvo_sequence import ROLAND_REDUCED_MAPPING
from data.src.dataLoaders import GrooveDataSet_Density
from model import GrooVAEDensity1D
import os
import logging
root_logger = logging.getLogger()
root_logger.setLevel(logging.WARNING)



# subset = load_gmd_hvo_sequences(dataset_setting_json_path="data/dataset_json_settings/4_4_BeatsAndFills_gmd_96.json",
#                                         subset_tag="test",
#                                         force_regenerate=False)

def load_density_1d_model(model_path, params_dict=None, is_evaluating=True, device=None):
    """ Load a GrooveTransformerEncoder model from a given path

    Args:
        model_path (str): path to the model
        params_dict (None, dict, or json path): dictionary containing the parameters of the model
        is_evaluating (bool): if True, the model is set to eval mode
        device (None or torch.device): device to load the model to (if cpu, the model is loaded to cpu)

    Returns:
        model (GrooveTransformerEncoder): the loaded model
    """

    try:
        if device is not None:
            loaded_dict = torch.load(model_path, map_location=device)
        else:
            loaded_dict = torch.load(model_path)
    except:
        loaded_dict = torch.load(model_path, map_location=torch.device('cpu'))
        #logger.info(f"Model was loaded to cpu!!!")

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
    params_dict['add_params'] = True
    model = GrooVAEDensity1D(params_dict)
    model.load_state_dict(loaded_dict["model_state_dict"])
    if is_evaluating:
        model.eval()

    return model

densities = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

if __name__ == "__main__":

        os.chdir("/Users/jlenz/Desktop/Thesis/GrooveTransformer")
        print(os.getcwd())
        test_dataset = GrooveDataSet_Density(
                dataset_setting_json_path="/Users/jlenz/Desktop/Thesis/GrooveTransformer/data/dataset_json_settings/4_4_BeatsAndFills_gmd.json",
                subset_tag="test",
                max_len=32)

        # download model from wandb and load it
        run = wandb.init()

        epoch = 150
        version = 74
        run_name = "fine-sweep-56" #"rosy-sweep-51"       # f"apricot-sweep-56_ep{epoch}"

        artifact_path = f"mmil_julian/Control 1D/model_epoch_{epoch}:v{version}"
        epoch = artifact_path.split("model_epoch_")[-1].split(":")[0]

        artifact = run.use_artifact(artifact_path, type='model')
        artifact_dir = artifact.download()
        model = load_density_1d_model(os.path.join(artifact_dir, f"{epoch}.pth"))

        os.chdir("/Users/jlenz/Desktop/Thesis/GrooveTransformer/Experiments/Control/density/midi_1d")

        for i in range(10):
            hvo_sequence = test_dataset.get_hvo_sequences_at(i)
            hvo_sequence.save_hvo_to_midi(filename=f"{i}_original.mid")
            input, _, _, _, _, _ = test_dataset[i]
            input = torch.unsqueeze(input, dim=0)
            for density in densities:
                density_tensor = torch.unsqueeze(torch.tensor(density), dim=0)
                hvo, _, _, _ = model.predict(input, density_tensor, return_concatenated=True)
                hvo = torch.squeeze(hvo, dim=0).numpy()
                hvo_sequence.hvo = hvo
                hvo_sequence.save_hvo_to_midi(filename=f"{i}_density.{density}.mid")



