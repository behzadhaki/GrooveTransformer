
import logging
wandb_logger = logging.getLogger('wandb')
wandb_logger.setLevel(logging.WARNING)

import matplotlib
import matplotlib.pyplot as plt
import torch
import os
import wandb
import numpy as np
import json
import random
from copy import deepcopy
import shutil
import os

from data.src.dataLoaders import GrooveDataSet_Control
from eval_vaeder_tools import *



DATASET_JSON_PATH = "../../data/dataset_json_settings/4_4_BeatsAndFills_gmd.json"
WANDB_PATH = "mmil_julian/ControlAdversarial/"

"""
Revived 3: High offset dropout
Jumping 22: Highest DICE and TP hit scores (most accurate overall) and highest density/intensity
Rose 43: Highest KL value (0.48)
classic 31: Lowest hit score loss, high intensity values

"""


MODELS_DICT = {"revived_3": "490:v2",
               "jumping_22": "130:v57",
               "rose_43": "190:v37",
               "classic_31": "160:v30"
               }


GENRE_JSON_PATH = "../../data/control/gmd_genre_dict.json"
GEN_UMAPS = True
GEN_MIDI = True
N_MIDI_INPUTS = 4
GENRES = ["rock", "jazz", "latin"]
SERIALIZE_WHOLE_MODEL = True
SERIALIZE_MODEL_COMPONENTS = True

IS_TESTING = False


"""
name, epoch/version data, model class, 
"""

if __name__ == "__main__":

    # Get original normalization functions
    DOWNSAMPLE = 0.1 if IS_TESTING else None
    print("Loading Normalization Functions")
    norm_dataset = GrooveDataSet_Control(
        dataset_setting_json_path=DATASET_JSON_PATH,
        subset_tag="train",
        max_len=32,
        collapse_tapped_sequence=True,
        load_as_tensor=True,
        down_sampled_ratio=DOWNSAMPLE,
        normalize_densities=True,
        normalize_intensities=True)
    density_norm_fn = norm_dataset.normalize_density
    intensity_norm_fn = norm_dataset.normalize_intensity


    # Start wandb
    run = wandb.init()
    genre_dict = 0

    # load the models
    for model_name, artifact_path in MODELS_DICT.items():

        print(f"\n**Downloading {model_name}")

        path = WANDB_PATH + "model_epoch_" + artifact_path
        artifact = run.use_artifact(path, type="model")
        artifact_dir = artifact.download()
        epoch = path.split("model_epoch_")[-1].split(":")[0]
        model = load_vaeder_model(os.path.join(artifact_dir, f"{epoch}.pth"), genre_json_path=GENRE_JSON_PATH)

        genre_dict = model.get_genre_dict()
        dataset = GrooveDataSet_Control(
            dataset_setting_json_path=DATASET_JSON_PATH,
            subset_tag="test",
            max_len=32,
            collapse_tapped_sequence=True,
            load_as_tensor=True,
            down_sampled_ratio=DOWNSAMPLE,
            normalize_densities=False,
            normalize_intensities=False,
            custom_genre_mapping_dict=genre_dict)

        MODELS_DICT[model_name] = {"model": model,
                                   "genre_dict": genre_dict,
                                   "dataset": dataset}

    model_path = "models_eval"
    os.makedirs(model_path, exist_ok=True)
    os.chdir(model_path)

    # Umap Generation
    if GEN_UMAPS:
        print("Generating umaps")
        make_empty_folder("umaps")

        for model_name, model_data in MODELS_DICT.items():
            model = model_data["model"]
            dataset = model_data["dataset"]
            fig = generate_vaeder_umaps(model, dataset, density_norm_fn, intensity_norm_fn)
            fig.savefig(f"{model_name}_umaps.png", format="png", bbox_inches='tight', dpi=400)

        os.chdir("../")

    # Create subdirectory and generate MIDI files
    if GEN_MIDI:
        print("\nGenerating MIDI files")
        make_empty_folder("midi_files")

        sample_indices = [random.randint(0, 100) for _ in range(N_MIDI_INPUTS)]
        for model_name, model_data in MODELS_DICT.items():
            model = model_data["model"]
            dataset = model_data["dataset"]

            generate_vaeder_midi_examples(model_name, model, dataset, sample_indices,
                                          GENRES, density_norm_fn, intensity_norm_fn)

        os.chdir("../")



    # Serialize
    if SERIALIZE_WHOLE_MODEL:
        print("\nSerializing models")
        make_empty_folder("serialized_models")
        for model_name, model_data in MODELS_DICT.items():
            model_data['model'].serialize_whole_model(model_name, os.getcwd())

        os.chdir("../")

    if SERIALIZE_MODEL_COMPONENTS:
        print("\nSerializing model components")
        make_empty_folder("serialized_components")
        for model_name, model_data in MODELS_DICT.items():

            model_data['model'].serialize(model_name)
        os.chdir("../")




