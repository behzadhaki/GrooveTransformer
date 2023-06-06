import torch
import numpy as np
import wandb
from hvo_sequence import HVO_Sequence
from hvo_sequence import ROLAND_REDUCED_MAPPING
from data.src.dataLoaders import load_gmd_hvo_sequences, GrooveDataSet_Density
#from helpers.Control.density_1D_modelLoader import load_1d_density_model
import os
print(os.getcwd())
# subset = load_gmd_hvo_sequences(dataset_setting_json_path="data/dataset_json_settings/4_4_BeatsAndFills_gmd_96.json",
#                                         subset_tag="test",
#                                         force_regenerate=False)

test_dataset = GrooveDataSet_Density(
        dataset_setting_json_path="/Users/jlenz/Desktop/Thesis/GrooveTransformer/data/dataset_json_settings/4_4_Beats_gmd.json",
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
model = load_1d_density_model(os.path.join(artifact_dir, f"{epoch}.pth"))

# for idx in range(len(hvo_seqs)):
#     in_hv, out_hv = evaluator[idx]
#     in_hv.save_hvo_to_midi(filename=f"misc/tokenize/midi/in_{idx}.mid")
#     out_hv.save_hvo_to_midi(filename=f"misc/tokenize/midi/out_{idx}.mid")