### USed for testing evaluator and WANDB Integration

import sys
sys.path.insert(1, "../../hvo_sequence")
sys.path.insert(1, "../hvo_sequence")

from hvo_sequence.drum_mappings import ROLAND_REDUCED_MAPPING
import numpy as np

from GrooveEvaluator.evaluator import Evaluator

from bokeh.io import output_file, show, save, export_png
from bokeh.models.ranges import Range1d

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
    n_samples_to_use=128,
    n_samples_to_synthesize_visualize_per_subset=10,
    disable_tqdm=False,
    analyze_heatmap=True,
    analyze_global_features=True
)

train_set_evaluator2 = Evaluator(
    pickle_source_path=pickle_source_path, set_subfolder="GrooveMIDI_processed_train",
    hvo_pickle_filename="hvo_sequence_data.obj",
    list_of_filter_dicts_for_subsets=list_of_filter_dicts_for_subsets,
    max_hvo_shape=(32, 27),
    n_samples_to_use=256,
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
pred_hvos_array2 = train_set_evaluator2.get_ground_truth_hvos_array()
_hvos_array_tags2 = train_set_evaluator2._gt_hvos_array_tags
tags2 = list(set(_hvos_array_tags2))

# Set all kicks of the first tag to hit 1 and vel 0.1
indices = [index for index, element in enumerate(_hvos_array_tags) if element == tags[0]]
pred_hvos_array[indices, :, 0] = 1
pred_hvos_array[indices, :, 9] = 0.1

# Set half the kicks of the third tag to hit 1 and vel 0.8
indices = [index for index, element in enumerate(_hvos_array_tags2) if element == tags2[2]]
pred_hvos_array2[indices[::2], :, 0] = 1
pred_hvos_array2[indices[::2], :, 9] = 0.8


# Pass the predictions back to the evaluator
train_set_evaluator.add_predictions(pred_hvos_array)
train_set_evaluator2.add_predictions(pred_hvos_array2)



# ###################
# ###################
# Export Velocity Heatmaps to PNG Starts here
# ###################

### Step 0. Get tabs and add them to a list export heatmaps from evaluators
_gt_heatmaps_epoch1_dict, _pred_heatmaps_epoch1_dict = train_set_evaluator.get_logging_dict(
    velocity_heatmap_html=True, global_features_html=True,
    piano_roll_html=False, audio_files=False
)
_gt_heatmaps_epoch2_dict, _pred_heatmaps_epoch2_dict = train_set_evaluator2.get_logging_dict(
    velocity_heatmap_html=True, global_features_html=True,
    piano_roll_html=False, audio_files=False)

### Step 1. Get heatmaps and add them to a list
# in reality velocity_heatmap_epoch1 and 2 come from dumped evaluators
velocity_heatmap_epoch1 = _pred_heatmaps_epoch1_dict["velocity_heatmaps"]
velocity_heatmap_epoch2 = _pred_heatmaps_epoch2_dict["velocity_heatmaps"]

heatmaps_per_epochs = [velocity_heatmap_epoch1, velocity_heatmap_epoch2]

### Step 2. Syncronize tabs in second heatmap to first one
### Step 3. Export PNGs

n_tabs = len(velocity_heatmap_epoch1.tabs[0].child.tabs)

n_epochs = 2
hide_scatters = True

for epoch in range(n_epochs):
    for tab_ix in range(n_tabs):
        if hide_scatters == True:
            del heatmaps_per_epochs[epoch].tabs[0].child.tabs[tab_ix].child.renderers[::2]

        voice = heatmaps_per_epochs[epoch].tabs[0].child.tabs[tab_ix].title
        heatmaps_per_epochs[epoch].tabs[0].child.tabs[tab_ix].child.y_range = Range1d(0, 1480)
        heatmaps_per_epochs[epoch].tabs[0].child.tabs[tab_ix].child.x_range = Range1d(0, 32)

        export_png(heatmaps_per_epochs[epoch].tabs[0].child.tabs[tab_ix].child,
                   filename="./misc/sample_exports/velo_heatmap_voice_{}_epoch_{}.png".format(voice, epoch))


# ###################
# ###################
# Export Global Features Heatmaps to PNG Starts here
# ###################

### Step 1. Get heatmaps and add them to a list
# in reality global_feature_pdf_epoch1 and 2 come from dumped evaluators
global_feature_pdf_epoch1 = _pred_heatmaps_epoch1_dict["global_feature_pdfs"]
global_feature_pdf_epoch2 = _pred_heatmaps_epoch2_dict["global_feature_pdfs"]

heatmaps_per_epochs = [global_feature_pdf_epoch1, global_feature_pdf_epoch2]

### Step 2. Syncronize tabs in second heatmap to first one
### Step 3. Export PNGs a


n_epochs = 2
for epoch in range(n_epochs):
    for major_tab_ix in range(len(heatmaps_per_epochs[epoch].tabs)):
        major_title = heatmaps_per_epochs[epoch].tabs[major_tab_ix].title
        for tab_ix in range(len(global_feature_pdf_epoch1.tabs[major_tab_ix].child.tabs)):
            feature_title = heatmaps_per_epochs[epoch].tabs[major_tab_ix].child.tabs[tab_ix].title
            if len(heatmaps_per_epochs[epoch].tabs[major_tab_ix].child.tabs)>=1:
                #heatmaps_per_epochs[epoch].tabs[major_tab_ix].child.tabs[tab_ix].child.y_range = Range1d(0, 20)
                #heatmaps_per_epochs[epoch].tabs[major_tab_ix].child.tabs[tab_ix].child.x_range = Range1d(-10, 10)
                try:
                    export_png(heatmaps_per_epochs[epoch].tabs[major_tab_ix].child.tabs[tab_ix].child,
                               filename="./misc/sample_exports/{}_{}_epoch_{}.png".format(major_title, feature_title, epoch))
                except:
                    continue


