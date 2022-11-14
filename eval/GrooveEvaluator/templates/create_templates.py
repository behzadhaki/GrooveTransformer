import sys, os
from data import load_gmd_hvo_sequences, load_down_sampled_gmd_hvo_sequences
from eval.GrooveEvaluator import Evaluator
sys.path.insert(0, os.path.join(os.getcwd().split("GrooveTransformer")[0], "GrooveTransformer"))

if __name__ == "__main__":
    dataset = "4_4_Beats_gmd"
    dataset_setting_json_path = os.path.join(sys.path[0], f"data/dataset_json_settings/{dataset}.json")

    for dataset in ["4_4_Beats_gmd", "4_4_BeatsAndFills_gmd"]:
        for subset_name in ["train", "test", "validation"]:

            # =============================================================================================================
            #
            # Create Evaluators for Complete Sets Without Genre Division
            #
            #   Given that there are thousands of samples in these evaluators, certain analysis can be quite slow.
            #   As a result, do not use these evaluators for the following analysis:
            #         1. Audio Synthesis
            #         2. Piano Roll Synthesis
            #         3. KL/OA Analysis
            #
            # =============================================================================================================
            hvo_seq_set = load_gmd_hvo_sequences(dataset_setting_json_path, subset_name)
            Evaluator(
                hvo_sequences_list_=hvo_seq_set,
                list_of_filter_dicts_for_subsets=None,
                _identifier=f"{dataset}_{subset_name}_full",
                n_samples_to_use=-1,
                max_hvo_shape=(32, 27),
                need_hit_scores=True,
                need_velocity_distributions=True,
                need_offset_distributions=True,
                need_rhythmic_distances=True,
                need_heatmap=True,
                need_global_features=False,
                need_audio=False,
                need_piano_roll=False,
                need_kl_oa=False,
                n_samples_to_synthesize_and_draw="all",
                disable_tqdm=False).dump(
                path=f"eval/GrooveEvaluator/templates/")

            # =============================================================================================================
            #
            # Create Evaluators for dataset down-sampled to 0.05 of the original size
            #
            #   Given that there are thousands of samples in these evaluators, certain analysis can be quite slow.
            #   As a result, it is recommended to use the full set of samples for the complete set for
            #   the following analysis:
            #     1. Piano Rolls
            #     2. Audio Synthesis
            #     3. KL/OA Analysis
            #
            # =============================================================================================================

            hvo_seq_set = load_down_sampled_gmd_hvo_sequences(dataset_setting_json_path, subset_name,
                                                              down_sampled_ratio=0.05,
                                                              cache_down_sampled_set=False)
            Evaluator(
                hvo_sequences_list_=hvo_seq_set,
                list_of_filter_dicts_for_subsets=None,
                _identifier=f"5_percent_of_{dataset}_{subset_name}",
                n_samples_to_use=-1,
                max_hvo_shape=(32, 27),
                need_hit_scores=False,
                need_velocity_distributions=False,
                need_offset_distributions=False,
                need_rhythmic_distances=False,
                need_heatmap=False,
                need_global_features=False,
                need_audio=True,
                need_piano_roll=True,
                need_kl_oa=True,
                n_samples_to_synthesize_and_draw="all",
                disable_tqdm=False).dump(
                path=f"eval/GrooveEvaluator/templates/")