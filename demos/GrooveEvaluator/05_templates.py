#  Copyright (c) 2022. \n Created by Behzad Haki. behzad.haki@upf.edu
import sys, os
sys.path.insert(0, os.path.join(os.getcwd().split("GrooveTransformer")[0], "GrooveTransformer"))

from eval.GrooveEvaluator import create_template, load_evaluator_template

if __name__ == "__main__":
    dataset = "4_4_Beats_gmd"
    dataset_setting_json_path = f"{sys.path[0]}/data/dataset_json_settings/{dataset}.json"

    # creating a template
    # An evaluator template containing ALL the test data, not separated by genre
    full_set_eval = create_template(
        dataset_setting_json_path=dataset_setting_json_path,
        subset_name="test",
        down_sampled_ratio=None,
        cached_folder="eval/GrooveEvaluator/templates/",
        divide_by_genre=True
    )

    # loading the template
    #   If a template already exists, it will be loaded instead of creating a new one.
    #     otherwise, it will be created and saved in the cached_folder.
    # Example: An evaluator template containing 0.01 ration of the test data, separated by genre
    partial_set_eval = load_evaluator_template(
        dataset_setting_json_path=dataset_setting_json_path,
        subset_name="test",
        down_sampled_ratio=0.01,
        cached_folder="eval/GrooveEvaluator/templates/",
        divide_by_genre=True
    )