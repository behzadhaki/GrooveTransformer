#  Copyright (c) 2022. \n Created by Behzad Haki. behzad.haki@upf.edu
import torch
import numpy as np
from eval.GrooveEvaluator import load_evaluator
from eval.GrooveEvaluator import Evaluator
from model import GrooveTransformerEncoderVAE
from eval.GrooveEvaluator import load_evaluator_template

def get_logging_media_for_vae_model_wandb(groove_transformer_vae, device, dataset_setting_json_path, subset_name,
                            down_sampled_ratio, cached_folder="eval/GrooveEvaluator/templates/",
                            divide_by_genre=True, **kwargs):
    """
    Prepare the media for logging in wandb. Can be easily used with an evaluator template
    (A template can be created using the code in eval/GrooveEvaluator/templates/main.py)

    :param evaluator: either a path to an evaluator template or an evaluator object
    :param groove_transformer_vae:  a GrooveTransformerEncoderVAE model
    :param device:                  the device to run the model on
    :param kwargs:                  additional arguments: need_hit_scores, need_velocity_distributions,
                                    need_offset_distributions, need_rhythmic_distances, need_heatmap
    :return:                        a ready to use dictionary to be logged using wandb.log()
    """

    # and model is correct type
    assert isinstance(groove_transformer_vae, GrooveTransformerEncoderVAE)

    # load the evaluator template (or create a new one if it does not exist)
    evaluator = load_evaluator_template(
        dataset_setting_json_path=dataset_setting_json_path,
        subset_name=subset_name,
        down_sampled_ratio=down_sampled_ratio,
        cached_folder=cached_folder,
        divide_by_genre=divide_by_genre
    )

    # Prepare the flags for require media
    # ----------------------------------
    need_hit_scores = kwargs["need_hit_scores"] if "need_hit_scores" in kwargs.keys() else False
    need_velocity_distributions = kwargs["need_velocity_distributions"] \
        if "need_velocity_distributions" in kwargs.keys() else False
    need_offset_distributions = kwargs["need_offset_distributions"] \
        if "need_offset_distributions" in kwargs.keys() else False
    need_rhythmic_distances = kwargs["need_rhythmic_distances"] \
        if "need_rhythmic_distances" in kwargs.keys() else False
    need_heatmap = kwargs["need_heatmap"] if "need_heatmap" in kwargs.keys() else False
    need_global_features = kwargs["need_global_features"] \
        if "need_global_features" in kwargs.keys() else False
    need_piano_roll = kwargs["need_piano_roll"] if "need_piano_roll" in kwargs.keys() else False
    need_audio = kwargs["need_audio"] if "need_audio" in kwargs.keys() else False
    need_kl_oa = kwargs["need_kl_oa"] if "need_kl_oa" in kwargs.keys() else False

    # (1) Get the targets, (2) tapify and pass to the model (3) add the predictions to the evaluator
    # ------------------------------------------------------------------------------------------
    print("**" * 20)
    print(type(evaluator))
    hvo_seqs = evaluator.get_ground_truth_hvo_sequences()
    in_groove = torch.tensor(
        np.array([hvo_seq.flatten_voices() for hvo_seq in hvo_seqs]), dtype=torch.float32).to(
        device)
    hvos_array, _, _, _ = groove_transformer_vae.predict(in_groove, return_concatenated=True)
    evaluator.add_predictions(hvos_array.detach().cpu().numpy())

    # Get the media from the evaluator
    # -------------------------------
    media = evaluator.get_logging_media(
        prepare_for_wandb=True,
        need_hit_scores=need_hit_scores,
        need_velocity_distributions=need_velocity_distributions,
        need_offset_distributions=need_offset_distributions,
        need_rhythmic_distances=need_rhythmic_distances,
        need_heatmap=need_heatmap,
        need_global_features=need_global_features,
        need_piano_roll=need_piano_roll,
        need_audio=need_audio,
        need_kl_oa=need_kl_oa)

    return media