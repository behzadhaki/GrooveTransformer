import sys
sys.path.append("../../")
sys.path.append("../")
import wandb


from data.gmd.src import subsetters
# , Set_Sampler, convert_hvos_array_to_subsets
from eval.GrooveEvaluator.src.evaluator import HVOSeq_SubSet_Evaluator, Evaluator

from bokeh.io import output_file, show, save




# Create Subset Filters
styles = ["afrobeat", "afrocuban", "blues", "country", "dance", "funk", "gospel", "highlife", "hiphop", "jazz",
          "latin", "middleeastern", "neworleans", "pop", "punk", "reggae", "rock", "soul"]

list_of_filter_dicts_for_subsets = []
for style in styles:
    list_of_filter_dicts_for_subsets.append(
        {"style_primary": [style], "beat_type": ["beat"], "time_signature": ["4-4"]}
    )


gmd_pickle_path = "data/gmd/resources/cached/beat_division_factor_[4]/drum_mapping_label_['ROLAND_REDUCED_MAPPING']/beat_type_['beat']_time_signature_['4-4']/train.bz2pickle"

train_set_evaluator = Evaluator(
    gmd_pickle_path,
    list_of_filter_dicts_for_subsets=list_of_filter_dicts_for_subsets,
    max_hvo_shape=(32, 27),
    n_samples_to_use=4
)

gt_hvos_array = train_set_evaluator.get_ground_truth_hvos_array()
train_set_evaluator.add_predictions(gt_hvos_array)

train_set_evaluator.dump(path="eval/saved", fname="epoch-0")

audios_gt, audios_pd = train_set_evaluator.get_logging_dict()
max(audios_gt['captions_audios'][0][1])

# Pass to model
# predicted_hvos_array = model.predict(gt_hvos_array)
