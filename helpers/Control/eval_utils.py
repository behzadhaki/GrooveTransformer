#  Copyright (c) 2022. \n Created by Behzad Haki. behzad.haki@upf.edu
import torch
import numpy as np
from model import GrooveTransformerEncoderVAE
from eval.GrooveEvaluator import load_evaluator_template
from eval.UMAP import UMapper
from eval.GrooveEvaluator import Evaluator
from data import load_gmd_hvo_sequences, load_down_sampled_gmd_hvo_sequences
from data.control.control_utils import calculate_density

from logging import getLogger
logger = getLogger("helpers.VAE.eval_utils")
logger.setLevel("DEBUG")


def generate_umap_for_control_model_wandb(
        model, device, test_dataset, subset_name,
        collapse_tapped_sequence):
    """
    Generate the umap for the given model and dataset setting.
    Args:
        :param groove_transformer_vae: The model to be evaluated
        :param device: The device to be used for evaluation
        :param dataset_setting_json_path: The path to the dataset setting json file
        :param subset_name: The name of the subset to be evaluated
        :param collapse_tapped_sequence: Whether to collapse the tapped sequence or not (input will have 1 voice only)

    Returns:
        dictionary ready to be logged by wandb {f"{subset_name}_{umap}": wandb.Html}
    """

    in_groove = torch.tensor(
        np.array([hvo_seq.flatten_voices(reduce_dim=collapse_tapped_sequence)
                  for hvo_seq in test_dataset.hvo_sequences]), dtype=torch.float32).to(
        device)
    densities = torch.zeros(len(test_dataset.hvo_sequences), dtype=torch.float32).to(device)
    for idx, hvo_seq in enumerate(test_dataset.hvo_sequences):
        densities[idx] = calculate_density(hvo_seq.hits)

    #tags = [(hvo_seq.metadata["style_primary"], density) for hvo_seq, density in zip(test_dataset.hvo_sequences, densities)]
    tags = [str(round(density.item(), 2)) for density in densities]

    _, _, _, latents_z = model.predict(in_groove, densities, return_concatenated=True)

    umapper = UMapper(subset_name)
    umapper.fit(latents_z.detach().cpu().numpy(), tags_=tags)
    p = umapper.plot(show_plot=False, prepare_for_wandb=True)
    return {f"{subset_name}_umap": p}


def get_logging_media_for_control_model_wandb(
        model, device, dataset_setting_json_path,
        collapse_tapped_sequence,
        divide_by_genre=True, **kwargs):
    """
    Prepare a list of media dicts for logging in wandb. The list will be the # of params
    to be tested * the number of samples. For each sample, it will loop through
    the parameter options, run it through the model, and return an input/output pair
    :param model: The model to be evaluated
    :param device: The device to be used for evaluation
    :param dataset_setting_json_path: The path to the dataset setting json file
    :param subset_name: The name of the subset to be evaluated
    :param down_sampled_ratio: The ratio of the subset to be evaluated
    :param collapse_tapped_sequence: Whether to collapse the tapped sequence or not (input will have 1 voice only)
    :param cached_folder: The folder to be used for caching the evaluator template
    :param divide_by_genre: Whether to divide the subset by genre or not
    :param kwargs:                  additional arguments: need_hit_scores, need_velocity_distributions,
                                    need_offset_distributions, need_rhythmic_distances, need_heatmap
    :return:                        a ready to use dictionary to be logged using wandb.log()
    """

    hvo_seq_set = load_gmd_hvo_sequences(dataset_setting_json_path=dataset_setting_json_path,
                                         subset_tag='test')

    evaluator = Evaluator(
        hvo_sequences_list_=hvo_seq_set,
        list_of_filter_dicts_for_subsets=None,
        _identifier="test set",
        n_samples_to_use=3,
        max_hvo_shape=(32, 27),
        need_hit_scores=False,
        need_velocity_distributions=False,
        need_offset_distributions=False,
        need_rhythmic_distances=False,
        need_heatmap=False,
        need_global_features=False,
        need_audio=False,
        need_piano_roll=True,
        need_kl_oa=False,
        n_samples_to_synthesize_and_draw=0,
        disable_tqdm=False)

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

    # (1) Get the targets, (2) tapify, (3) iterate through parameters, (4) add results to list
    # ------------------------------------------------------------------------------------------
    hvo_seqs = evaluator.get_ground_truth_hvo_sequences()
    in_grooves = torch.tensor(
        np.array([hvo_seq.flatten_voices(reduce_dim=collapse_tapped_sequence)
                  for hvo_seq in hvo_seqs]), dtype=torch.float32).to(
        device)
    results = list()
    density_values = ["real_input", 0.05, 0.5, 0.95]
    print("Preparing media for logging")

    for param in density_values:
        densities = torch.zeros(len(hvo_seqs), dtype=torch.float32).to(device)
        for idx, hvo_seq in enumerate(hvo_seqs):
            densities[idx] = calculate_density(hvo_seq.hits) if param == "real_input" else param
            outputs, mu, log_var, latent_z = model.predict(in_grooves, densities, return_concatenated=True)
            evaluator.add_predictions(outputs.detach().cpu().numpy())

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

            results.append(media)

    return results

def get_hit_scores_for_vae_model(groove_transformer_vae, device, dataset_setting_json_path, subset_name,
                            down_sampled_ratio, collapse_tapped_sequence,
                                 cached_folder="eval/GrooveEvaluator/templates/",
                            divide_by_genre=True):

    # logger.info("Generating the hit scores for subset: {}".format(subset_name))
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

    # (1) Get the targets, (2) tapify and pass to the model (3) add the predictions to the evaluator
    # ------------------------------------------------------------------------------------------
    hvo_seqs = evaluator.get_ground_truth_hvo_sequences()

    in_groove = torch.tensor(
        np.array([hvo_seq.flatten_voices(reduce_dim=collapse_tapped_sequence)
                  for hvo_seq in hvo_seqs]), dtype=torch.float32)
    predictions = []

    # batchify the input
    groove_transformer_vae.eval()
    with torch.no_grad():
        for batch_ix, batch_in in enumerate(torch.split(in_groove, 32)):
            hvos_array, _, _, _ = groove_transformer_vae.predict(
                batch_in.to(device),
                return_concatenated=True)
            predictions.append(hvos_array.detach().cpu().numpy())

    evaluator.add_predictions(np.concatenate(predictions))

    hit_dict = evaluator.get_statistics_of_pos_neg_hit_scores()

    score_dict = {f"{subset_name}/{key}_mean".replace(" ","_").replace("-","_"): float(value['mean']) for key, value in hit_dict.items()}
    score_dict.update({f"{subset_name}/{key}_std".replace(" ","_").replace("-","_"): float(value['std']) for key, value in hit_dict.items()})
    return score_dict
