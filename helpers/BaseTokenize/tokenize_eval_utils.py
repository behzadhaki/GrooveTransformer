#  Copyright (c) 2022. \n Created by Behzad Haki. behzad.haki@upf.edu
import torch
import numpy as np
# from model import GrooveTransformerEncoderVAE
from model import TokenizedTransformerEncoder
from eval.GrooveEvaluator import load_evaluator_template
from data import MonotonicGrooveTokenizedDataset
from hvo_sequence.tokenization import *

from logging import getLogger
logger = getLogger("helpers.VAE.eval_utils")
logger.setLevel("DEBUG")


def get_logging_media_for_tokenize_model_wandb(
        groove_transformer_tokenize, device, dataset_setting_json_path, subset_name,
        down_sampled_ratio, vocab,
        cached_folder="eval/GrooveEvaluator/templates/",
        divide_by_genre=True, max_length=400,
        **kwargs):
    """
    Prepare the media for logging in wandb. Can be easily used with an evaluator template
    (A template can be created using the code in eval/GrooveEvaluator/templates/main.py)
    :@param groove_transformer_tokenize: The model to be evaluated
    :@param device: The device to be used for evaluation
    :@param dataset_setting_json_path: The path to the dataset setting json file
    :@param subset_name: The name of the subset to be evaluated
    :@param down_sampled_ratio: The ratio of the subset to be evaluated
    :@param collapse_tapped_sequence: Whether to collapse the tapped sequence or not (input will have 1 voice only)
    :@param vocab: dictionary of {token: int} to be used to tokenize the subset and outputs
    :@param cached_folder: The folder to be used for caching the evaluator template
    :@param divide_by_genre: Whether to divide the subset by genre or not
    :@param max_length: padding/slicing factor when tokenizing dataset.
    :@param kwargs:                  additional arguments: need_hit_scores, need_velocity_distributions,
                                    need_offset_distributions, need_rhythmic_distances, need_heatmap
    :@return:                        a ready to use dictionary to be logged using wandb.log()
    """

    # and model is correct type
    assert isinstance(groove_transformer_tokenize, TokenizedTransformerEncoder)

    # load the evaluator template (or create a new one if it does not exist)
    evaluator = load_evaluator_template(
        dataset_setting_json_path=dataset_setting_json_path,
        subset_name=subset_name,
        down_sampled_ratio=down_sampled_ratio,
        cached_folder=cached_folder,
        divide_by_genre=divide_by_genre
    )

    # logger.info("Generating the PianoRolls for subset: {}".format(subset_name))

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

    # New version:
    # Get the sequences (as hvo objects)
    # Tokenize them (with pre-defined vocab)
    # Run through the model
    # Convert outputs to standard HVO arrays
    # Feed back into evaluator as predictions

    input_hvo_seqs = evaluator.get_ground_truth_hvos_sequences()

    eval_dataset = MonotonicGrooveTokenizedDataset(dataset_setting_json_path=dataset_setting_json_path,
                                                   vocab=vocab,
                                                   subset_tag="Test",
                                                   subset=input_hvo_seqs,
                                                   tapped_voice_idx=2,
                                                   flatten_velocities=False,
                                                   ticks_per_beat=96,
                                                   max_length=max_length)

    # Get input data, run it through the model
    input_tokens, input_hv, masks = eval_dataset.get_input_arrays()
    t, h, v = groove_transformer_tokenize.predict(input_tokens, input_hv, masks)

    # Convert model outputs into a single batched HVO array
    # (tokejns, hits, vels) -> tokenized_sequence(list) -> hvo array(numpy) -> Slice & batch
    #Todo: Is there some issue with how I'm going from torch->numpy?
    reverse_vocab = {v: k for k, v in eval_dataset.vocab().items()}
    output_hvo_arrays = list()
    for tokens, hits, velocities in zip(t, h, v):
        tokenized_seq = convert_model_output_to_tokenized_sequence(tokens, hits, velocities, reverse_vocab=reverse_vocab)
        hvo_array = convert_tokenized_sequence_to_hvo_array(tokenized_seq)
        output_hvo_arrays.append(hvo_array)

    #Todo: How can we deal with sequences of different lengths?
    seq_slice = 1000
    output_hvo_arrays = [array[:seq_slice, :] for array in output_hvo_arrays]
    output_hvo_arrays = np.stack(output_hvo_arrays)

    evaluator.add_predictions(output_hvo_arrays)

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


