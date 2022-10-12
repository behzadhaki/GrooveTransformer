### USed for testing evaluator and WANDB Integration

import sys
sys.path.insert(1, "../../hvo_sequence")
sys.path.insert(1, "../hvo_sequence")

from hvo_sequence.drum_mappings import ROLAND_REDUCED_MAPPING
import wandb
import numpy as np

from GrooveEvaluator.evaluator import Evaluator

from bokeh.io import output_file, show, save

wandb_run = wandb.init(project="GMD Analysis3")

pickle_source_path = "../preprocessed_dataset/datasets_extracted_locally/GrooveMidi/hvo_0.5.2" \
                     "/Processed_On_21_07_2021_at_14_32_hrs"

# Create Subset Filters (styles both in train and test)
styles = ["hiphop", "funk", "reggae", "soul", "latin", "jazz", "pop", "afrobeat", "highlife", "punk", "rock"]


list_of_filter_dicts_for_subsets = []
for style in styles:
    list_of_filter_dicts_for_subsets.append(
        {"style_primary": [style], "beat_type": ["beat"], "time_signature": ["4-4"]}
    )

# todo implement distance difference calculator between hvo_sequences
train_set_evaluator = Evaluator(
    pickle_source_path=pickle_source_path, set_subfolder="GrooveMIDI_processed_train",
    hvo_pickle_filename="hvo_sequence_data.obj",
    list_of_filter_dicts_for_subsets=list_of_filter_dicts_for_subsets,
    max_hvo_shape=(32, 27),
    n_samples_to_use=2048,
    n_samples_to_synthesize_visualize_per_subset=10,
    disable_tqdm=False,
    analyze_heatmap=True,
    analyze_global_features=True
)

# Get data to be passed to model
gt_hvos_array = train_set_evaluator.get_ground_truth_hvos_array()

# Preprocess Ground truth data and forward it through the model
# (next two lines are to mock this process)
pred_hvos_array = train_set_evaluator.get_ground_truth_hvos_array()
_hvos_array_tags = train_set_evaluator._gt_hvos_array_tags
tags = list(set(_hvos_array_tags))

# Set all kicks of the first tag to hit 1 and vel 0.1
tags[0]
indices = [index for index, element in enumerate(_hvos_array_tags) if element == tags[0]]
pred_hvos_array[indices, :, 0] = 1
pred_hvos_array[indices, :, 9] = 0.1

# Set half the kicks of the third tag to hit 1 and vel 0.8
tags[2]
indices = [index for index, element in enumerate(_hvos_array_tags) if element == tags[2]]
pred_hvos_array[indices[::2], :, 0] = 1
pred_hvos_array[indices[::2], :, 9] = 0.8

# Set half the kicks of the fifth tag to hit 0 and vel 0.0
tags[4]
indices = [index for index, element in enumerate(_hvos_array_tags) if element == tags[4]]
pred_hvos_array[indices, :, :] = 0
pred_hvos_array[indices, :, :] = 0

(train_set_evaluator.get_ground_truth_hvos_array()[:, :, 9:18] - pred_hvos_array[:, :, 9:18]).mean()

# Pass the predictions back to the evaluator
train_set_evaluator.add_predictions(pred_hvos_array)

"""# Get Accuracies
accuracy_dict = train_set_evaluator.get_hits_accuracies(drum_mapping=ROLAND_REDUCED_MAPPING)
vel_mse_dict = train_set_evaluator.get_velocity_errors(drum_mapping=ROLAND_REDUCED_MAPPING)
off_mse_dict = train_set_evaluator.get_micro_timing_errors(drum_mapping=ROLAND_REDUCED_MAPPING)
wandb.log(accuracy_dict, commit=False)
wandb.log(vel_mse_dict, commit=False)
wandb.log(off_mse_dict, commit=False)

# Get Rhythmic Distances between gt and predicted
rhythmic_distances = train_set_evaluator.get_rhythmic_distances()
wandb.log(rhythmic_distances, commit=False)"""

# Get Heatmaps/Features per gt or predicted
wandb_media = train_set_evaluator.get_wandb_logging_media(sf_paths=[
    "../hvo_sequence/hvo_sequence/soundfonts/Standard_Drum_Kit.sf2"])
if len(wandb_media.keys()) > 0:
    wandb.log(wandb_media, commit=False)


epoch = 1
wandb.log({"epoch": epoch})

train_set_evaluator.dump(path="misc/evaluator_run_{}_Epoch_{}.Eval".format(wandb_run.name, epoch))



"""
# Pass to model
# predicted_hvos_array = model.predict(gt_hvos_array)
gt_log_dict, predicted_log_dict = train_set_evaluator.get_logging_dict(sf_paths=[
    "../../hvo_sequence/hvo_sequence/soundfonts/Standard_Drum_Kit.sf2"])
gt_log_dict.keys()
train_set_evaluator.add_predictions(gt_hvos_array)"""


