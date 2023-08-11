# import matplotlib
# import matplotlib.pyplot as plt
# import torch
# import os
# import wandb
# import numpy as np
# import json
# import random
# from copy import deepcopy
# import shutil

from data.src.dataLoaders import GrooveDataSet_Control
from model import GrooveControl_VAE
from eval_vaeder_tools import *

# eval stuff
from eval.MultiSetEvaluator import MultiSetEvaluator
from eval.GrooveEvaluator import Evaluator
from bokeh.plotting import figure, show, save
from bokeh.io import output_notebook, reset_output


DATASET_JSON_PATH = "../../data/dataset_json_settings/4_4_BeatsAndFills_gmd.json"
WANDB_PATH = "mmil_julian/ControlAdversarial/"
MODELS_DICT = {"revived_3_epoch_490": "490:v2"}
GENRE_JSON_PATH = "../../data/control/gmd_genre_dict.json"
GEN_UMAPS = False
GEN_MIDI = False
N_MIDI_INPUTS = 3
GEN_EVALS = True
SERIALIZE_WHOLE_MODEL = False
SERIALIZE_MODEL_COMPONENTS = False

IS_TESTING = True




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


    # # Start wandb
    # # run = wandb.init()

    # load the models
    for model_name, artifact_path in MODELS_DICT.items():

        print(f"\n**Evaluating {model_name}")
    #     # path = WANDB_PATH + "model_epoch_" + artifact_path
    #     # artifact = run.use_artifact(path, type="model")
    #     # artifact_dir = artifact.download()
    #     # epoch = path.split("model_epoch_")[-1].split(":")[0]
    #     # model = load_vaeder_model(os.path.join(artifact_dir, f"{epoch}.pth"), genre_json_path=GENRE_JSON_PATH)

        # Todo: TEMP
        model = load_vaeder_model("/Users/jlenz/Desktop/Thesis/GrooveTransformer/eval/Control/artifacts/model_epoch_490:v2/490.pth",
                                  genre_json_path=GENRE_JSON_PATH)

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

        # Now for each model...
        # Create a subdirectory with model name
        model_path = "models_eval/" + model_name
        os.makedirs(model_path, exist_ok=True)
        os.chdir(model_path)
        # Generate two umaps; with and without control params
        if GEN_UMAPS:
            print("Generating umaps")
            fig = generate_vaeder_umaps(model, dataset)
            fig.savefig(f"{model_name}_umaps.png", format="png", bbox_inches='tight', dpi=400)

        # Create subdirectory and generate MIDI files
        if GEN_MIDI:
            generate_vaeder_midi_examples(model, dataset, N_MIDI_INPUTS, genre_dict, density_norm_fn, intensity_norm_fn)


        # Generate visual evals
        if GEN_EVALS:
            print("Generating Plots")
            gmd_hvo_seqs = dataset.hvo_sequences
            model_inputs = dataset.inputs
            densities_inputs = dataset.densities
            intensities_inputs = dataset.intensities
            genres_inputs = dataset.genres

            evaluator = Evaluator(
                gmd_hvo_seqs,
                list_of_filter_dicts_for_subsets=None,
                _identifier="test_set_full",
                n_samples_to_use=-1, #-1,
                max_hvo_shape=(32, 27),
                need_hit_scores=True,
                need_velocity_distributions=True,
                need_offset_distributions=False,
                need_rhythmic_distances=False,
                need_heatmap=False,
                need_global_features=False,
                need_audio=False,
                need_piano_roll=False,
                n_samples_to_synthesize_and_draw=False,   # "all",
                disable_tqdm=False)


            predictions, _, _, _ = model.predict(model_inputs, densities_inputs,
                                                 intensities_inputs, genres_inputs, return_concatenated=True)
            evaluator.add_predictions(predictions.detach().cpu().numpy())
            print(os.getcwd())
            evaluator.get_global_features_plot(save_path=f"GFPlotModel_{model_name}", only_combined_data_needed=False)



        # Serialize
        if SERIALIZE_WHOLE_MODEL:
            model.serialize_whole_model(model_name, os.getcwd())
            print("model serialized.")



