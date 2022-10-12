import sys
sys.path.insert(1, "../../")
sys.path.insert(1, "../")
import wandb


from preprocessed_dataset.Subset_Creators import subsetters
# , Set_Sampler, convert_hvos_array_to_subsets
from GrooveEvaluator.evaluator import HVOSeq_SubSet_Evaluator, Evaluator

from bokeh.io import output_file, show, save


pickle_source_path = "../preprocessed_dataset/datasets_extracted_locally/GrooveMidi/hvo_0.4.2/" \
                     "Processed_On_17_05_2021_at_22_32_hrs"

# Create Subset Filters
styles = ["afrobeat", "afrocuban", "blues", "country", "dance", "funk", "gospel", "highlife", "hiphop", "jazz",
          "latin", "middleeastern", "neworleans", "pop", "punk", "reggae", "rock", "soul"]

list_of_filter_dicts_for_subsets = []
for style in styles:
    list_of_filter_dicts_for_subsets.append(
        {"style_primary": [style], "beat_type": ["beat"], "time_signature": ["4-4"]}
    )

train_set_sampler = subsetters.GrooveMidiSubsetterAndSampler(
    pickle_source_path=pickle_source_path, subset="GrooveMIDI_processed_train",
    hvo_pickle_filename="hvo_sequence_data.obj",
    list_of_filter_dicts_for_subsets=list_of_filter_dicts_for_subsets,
    number_of_samples=2048,
    max_hvo_shape=(32, 27)
    )

test_set_sampler = subsetters.GrooveMidiSubsetterAndSampler(
    pickle_source_path=pickle_source_path, subset="GrooveMIDI_processed_test",
    hvo_pickle_filename="hvo_sequence_data.obj",
    list_of_filter_dicts_for_subsets=list_of_filter_dicts_for_subsets,
    number_of_samples=2048,
    max_hvo_shape=(32, 27)
    )

# Get sampled subsets
train_set_gt_tags, train_set_gt_subsets = train_set_sampler.get_subsets()
test_set_gt_tags, test_set_gt_subsets = test_set_sampler.get_subsets()

# Get the numpy array containing all sequences, along with other necessary fields
train_set_hvos_tags_gt, train_set_hvos_array_gt, train_set_hvo_seq_templates = train_set_sampler.get_hvos_array()
test_set_hvos_tags_gt, test_set_hvos_array_gt, test_set_hvo_seq_templates = test_set_sampler.get_hvos_array()

# hvos_array --> pass through model to get predictions
train_hvos_array_predicted = train_set_hvos_array_gt    # in reality: train_hvos_array_predicted=model.predict(hvos_array)
test_hvos_array_predicted = test_set_hvos_array_gt      # in reality: test_hvos_array_predicted=model.predict(hvos_array)

# create a subset of predictions
train_set_pred_tags, train_set_pred_subsets = subsetters.convert_hvos_array_to_subsets(
    train_set_hvos_tags_gt, train_hvos_array_predicted, train_set_hvo_seq_templates)
test_set_pred_tags, test_set_pred_subsets = subsetters.convert_hvos_array_to_subsets(
    test_set_hvos_tags_gt, test_hvos_array_predicted, test_set_hvo_seq_templates)


train_gt_evaluator = HVOSeq_SubSet_Evaluator (
    set_subsets=subsets_by_style_and_beat,              # Ground Truth typically
    set_tags=tags_by_style_and_beat,
    set_identifier= "TRAIN",
    analyze_heatmap=True,
    analyze_global_features=True,
    n_samples_to_analyze=20,
    synthesize_sequences=True,
    n_samples_to_synthesize=10,
    shuffle_audio_samples=False,                 # if false, it will reuse the same samples
    disable_tqdm=False,
    group_by_minor_keys=False
)

import pickle
pickle.dump(train_ground_evaluator, open("temp.pk", "wb"))

train_ground_evaluator.dump()

wandb_media_dict, wandb_features_data = train_ground_evaluator.get_wandb_logging_dicts(
    sf_paths=["../hvo_sequence/hvo_sequence/soundfonts/Standard_Drum_Kit.sf2"])

wandb.init(project="GMD Analysis", entity="behzadhaki")
# wandb.log(wandb_log)
data_log = {"epoch": 3}
data_log.update(wandb_media_dict)
data_log.update(wandb_features_data)
wandb.log(data_log)

wandb.log({"epoch": 2, "Train.Rock": {'Rock': .5}, "Train.Funk": 0.5, "Test/Loss": .1, })


