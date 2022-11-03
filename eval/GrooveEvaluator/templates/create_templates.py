from data import load_gmd_hvo_sequences
from eval.GrooveEvaluator import Evaluator

# ======================================================================================================================
# Load Data
# ======================================================================================================================
dataset_setting_json_path = "data/dataset_json_settings/4_4_Beats_gmd.json"
train_set = load_gmd_hvo_sequences(dataset_setting_json_path, "train")
test_set = load_gmd_hvo_sequences(dataset_setting_json_path, "test")
validation_set = load_gmd_hvo_sequences(dataset_setting_json_path, "validation")

# ======================================================================================================================
# Create Evaluators for Complete Sets Without Genre Division
# ======================================================================================================================
evaluator_test_set = Evaluator(
    test_set,
    list_of_filter_dicts_for_subsets=None,
    _identifier="test_set_full",
    n_samples_to_use=20, #-1,
    max_hvo_shape=(32, 27),
    need_hit_scores=True,
    need_velocity_distributions=True,
    need_offset_distributions=True,
    need_rhythmic_distances=True,
    need_heatmap=True,
    need_global_features=True,
    need_audio=True,
    need_piano_roll=True,
    n_samples_to_synthesize_and_draw=5,   # "all",
    disable_tqdm=False
)

