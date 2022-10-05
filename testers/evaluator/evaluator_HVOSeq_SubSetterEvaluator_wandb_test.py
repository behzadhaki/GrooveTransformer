import sys
sys.path.insert(1, "../../")
sys.path.insert(1, "../")
import wandb

#wandb.init(project="GMD Analysis", entity="behzadhaki")


from preprocessed_dataset.Subset_Creators import subsetters
# , Set_Sampler, convert_hvos_array_to_subsets
from GrooveEvaluator.evaluator import HVOSeq_SubSet_Evaluator, Evaluator

from bokeh.io import output_file, show, save


pickle_source_path = "../preprocessed_dataset/datasets_extracted_locally/GrooveMidi/hvo_0.4.2/" \
                     "Processed_On_17_05_2021_at_22_32_hrs"

"""# Create Subset Filters
styles = ["afrobeat", "afrocuban", "blues", "country", "dance", "funk", "gospel", "highlife", "hiphop", "jazz",
          "latin", "middleeastern", "neworleans", "pop", "punk", "reggae", "rock", "soul"]"""

# Create Subset Filters --> Test Set Styles
styles = ["hiphop", "funk", "reggae", "soul", "latin", "jazz", "pop", "afrobeat", "highlife", "punk", "rock"]


list_of_filter_dicts_for_subsets = []
for style in styles:
    list_of_filter_dicts_for_subsets.append(
        {"style_primary": [style], "beat_type": ["beat"], "time_signature": ["4-4"]}
    )

test_set_sampler = subsetters.GrooveMidiSubsetterAndSampler(
    pickle_source_path=pickle_source_path, subset="GrooveMIDI_processed_test",
    hvo_pickle_filename="hvo_sequence_data.obj",
    list_of_filter_dicts_for_subsets=list_of_filter_dicts_for_subsets,
    number_of_samples=1024,
    max_hvo_shape=(32, 27)
    )

# Get sampled subsets
test_set_gt_tags, test_set_gt_subsets = test_set_sampler.get_subsets()

test_gt_evaluator = HVOSeq_SubSet_Evaluator (
    set_subsets=test_set_gt_subsets,              # Ground Truth typically
    set_tags=test_set_gt_tags,
    set_identifier= "TEST",
    analyze_heatmap=True,
    analyze_global_features=True,
    max_samples_in_subset=None,
    n_samples_to_synthesize_visualize=30,
    disable_tqdm=False,
    group_by_minor_keys=True
)


wandb_log = test_gt_evaluator.get_wandb_logging_media(
    sf_paths="../hvo_sequence/hvo_sequence/soundfonts/Standard_Drum_Kit.sf2",
    use_specific_samples_at=[(0,0), (0,2), (10, 2), (10,5), (8,1)]
)

logging_dict = test_gt_evaluator.get_logging_dict(
    sf_paths="../hvo_sequence/hvo_sequence/soundfonts/Standard_Drum_Kit.sf2",
    use_specific_samples_at=[(0,0), (0,2), (10, 2), (10,5), (8,1)]
)



save(logging_dict["velocity_heatmaps"], "misc/temp.html")
#show(logging_dict["velocity_heatmaps"])
#show(logging_dict["global_feature_pdfs"])
save(logging_dict["piano_rolls"], "misc/temp.html")
#captions_audios = logging_dict["captions_audios"]

#wandb_logging_dict = test_gt_evaluator.get_wandb_logging_media(
#    sf_paths="../hvo_sequence/hvo_sequence/soundfonts/Standard_Drum_Kit.sf2")

#wandb.log({"epoch": 10, "test.eval.media":wandb_logging_dict})

