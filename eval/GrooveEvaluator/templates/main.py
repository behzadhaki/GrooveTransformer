from data import load_gmd_hvo_sequences, load_down_sampled_gmd_hvo_sequences
from eval.GrooveEvaluator import Evaluator
from eval.GrooveEvaluator import load_evaluator
import os

import logging
import bz2
import pickle

logger = logging.getLogger("eval.GrooveEvaluator.templates.main")
logger.setLevel("DEBUG")


def create_template(dataset_setting_json_path, subset_name, down_sampled_ratio=None,
                    cached_folder="eval/GrooveEvaluator/templates/", divide_by_genre=True):
    """
    Create a template for the given dataset and subset. The template will ALWAYS be saved in the cached_folder.
    :param dataset_setting_json_path:   The path to the json file that contains the dataset settings.
                                        (e.g. "data/dataset_json_settings/4_4_Beats_gmd.json")
    :param subset_name:                 The name of the subset to be used. (e.g. "train", "test", "validation")
    :param down_sampled_ratio:          The ratio of the down-sampled set. (e.g. 0.02)
    :param cache_down_sampled_set:      Whether to cache the down-sampled set.
    :param cached_folder:               The folder to save the template.
    :param divide_by_genre:             Whether to divide the dataset by genre.
    :return:
    """
    # load data using the settings specified in the dataset_setting_json_path
    if dataset_setting_json_path.endswith(".json"):
        if down_sampled_ratio is None:
            hvo_seq_set = load_gmd_hvo_sequences(dataset_setting_json_path, subset_name)
        else:
            hvo_seq_set = load_down_sampled_gmd_hvo_sequences(dataset_setting_json_path, subset_name,
                                                              down_sampled_ratio=down_sampled_ratio,
                                                              cache_down_sampled_set=False)
    else:
        with bz2.BZ2File(dataset_setting_json_path, "rb") as file:
            if down_sampled_ratio is None:
                loaded_data = pickle.load(file)
                hvo_seq_set = loaded_data[subset_name]["outputs_hvo_seqs"]
            else:
                loaded_data = pickle.load(file)
                hvo_seq_set = loaded_data[subset_name]["outputs_hvo_seqs"]
                hvo_seq_set = hvo_seq_set[:int(len(hvo_seq_set) * down_sampled_ratio)]

    # create a list of filter dictionaries for each genre if divide_by_genre is True
    if divide_by_genre:
        list_of_filter_dicts_for_subsets = []
        styles = list(set([hvo_seq.metadata["style_primary"] for hvo_seq in hvo_seq_set]))
        for style in styles:
            list_of_filter_dicts_for_subsets.append(
                {"style_primary": [style]}  # , "beat_type": ["beat"], "time_signature": ["4-4"]}
            )
    else:
        list_of_filter_dicts_for_subsets = None

    dataset_name = dataset_setting_json_path.split("/")[-1].split(".")[0]

    _identifier = f"_{down_sampled_ratio}_ratio_of_{dataset_name}_{subset_name}" \
        if down_sampled_ratio is not None else f"complete_set_of_{dataset_name}_{subset_name}"

    last_dim = hvo_seq_set[0].hvo.shape[-1]

    # create the evaluator
    eval = Evaluator(
        hvo_sequences_list_=hvo_seq_set,
        list_of_filter_dicts_for_subsets=list_of_filter_dicts_for_subsets,
        _identifier=_identifier,
        n_samples_to_use=-1,
        max_hvo_shape=(32, int(last_dim)),
        need_hit_scores=False,
        need_velocity_distributions=False,
        need_offset_distributions=False,
        need_rhythmic_distances=False,
        need_heatmap=False,
        need_global_features=False,
        need_audio=False,
        need_piano_roll=False,
        need_kl_oa=False,
        n_samples_to_synthesize_and_draw="all",
        disable_tqdm=False)

    eval.dump(path=cached_folder)

    logger.info(f"Template created and cached at {cached_folder}")
    return eval


def load_evaluator_template(dataset_setting_json_path, subset_name,
                            down_sampled_ratio, cached_folder="eval/GrooveEvaluator/templates/",
                            divide_by_genre=True, disable_logging=True):
    """
    Load a template for the given dataset and subset. If the template does not exist, it will be created and
    automatically saved in the cached_folder.

    :param dataset_setting_json_path:  The path to the json file that contains the dataset settings.

    :param subset_name:             The name of the subset to be used. (e.g. "train", "test", "validation")
    :param down_sampled_ratio:      The ratio of the down-sampled set. (e.g. 0.02)
    :param cached_folder:           The folder to save the template.
    :return:                        The evaluator template.
    """
    dataset_name = dataset_setting_json_path.split("/")[-1].split(".")[0]

    _identifier = f"_{down_sampled_ratio}_ratio_of_{dataset_name}_{subset_name}" \
        if down_sampled_ratio is not None else f"complete_set_of_{dataset_name}_{subset_name}"
    path = os.path.join(cached_folder, f"{_identifier}_evaluator.Eval.bz2")

    # if os.path.exists(path):
    #     if not disable_logging:
    #         logger.info(f"Loading template from {path}")
    #     return load_evaluator(path)
    # else:

    return create_template(
        dataset_setting_json_path=dataset_setting_json_path,
        subset_name=subset_name,
        down_sampled_ratio=down_sampled_ratio,
        cached_folder=cached_folder,
        divide_by_genre=divide_by_genre)