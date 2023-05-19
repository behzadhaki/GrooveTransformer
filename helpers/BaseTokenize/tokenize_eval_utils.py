#  Copyright (c) 2022. \n Created by Behzad Haki. behzad.haki@upf.edu

from model import TokenizedTransformerEncoder
from eval.GrooveEvaluator import load_evaluator_template
from data import MonotonicGrooveTokenizedDataset
from hvo_sequence.tokenization import *
from hvo_sequence import note_sequence_to_hvo_sequence

from logging import getLogger
logger = getLogger("helpers.VAE.eval_utils")
logger.setLevel("DEBUG")


def get_logging_media_for_tokenize_model_wandb(
        model, device, dataset_setting_json_path, subset_name,
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
    assert isinstance(model, TokenizedTransformerEncoder)

    # load the evaluator template (or create a new one if it does not exist)
    evaluator = load_evaluator_template(
        dataset_setting_json_path="/data/dataset_json_settings/4_4_Beats_gmd.json",
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

    # Get ground truth sequences and tokenize them
    input_hvo_seqs = evaluator.get_ground_truth_hvo_sequences()
    eval_dataset = MonotonicGrooveTokenizedDataset(dataset_setting_json_path=dataset_setting_json_path,
                                                   vocab=vocab,
                                                   subset_tag="Test",
                                                   subset=input_hvo_seqs,
                                                   tapped_voice_idx=2,
                                                   flatten_velocities=False,
                                                   ticks_per_beat=96,
                                                   max_length=max_length)
    reverse_vocab = {v: k for k, v in eval_dataset.get_vocab_dictionary().items()}
    input_tokens, input_hv, masks = eval_dataset.get_input_arrays()

    output_hvo_arrays = list()
    # Initialize un-quantized HVO sequence
    hvo_seq = HVO_Sequence(beat_division_factors=[96],
                           drum_mapping=ROLAND_REDUCED_MAPPING)
    hvo_seq.add_time_signature(time_step=0, numerator=4, denominator=4)
    hvo_seq.add_tempo(time_step=0, qpm=120)

    """
    For each sequence:
    1. Convert dimensionality to 3D and run through model
    2. Convert outputs to the tokenized format (string_token, HV_array)
    3. Convert tokenized format to HVO array
    4. Save HVO to a note sequence
    5. Reload as a new HVO object, with standard HVO beat quantization (16ths)
    6. Append array to the list 
    """

    for token, hv, mask in zip(input_tokens, input_hv, masks):
        token = token.unsqueeze(0)
        hv = hv.unsqueeze(0)
        mask = mask.unsqueeze(0)
        t, h, v = model.predict(token, hv, mask)
        t = t.squeeze(0)
        h = h.squeeze(0)
        v = v.squeeze(0)
        tokenized_seq = convert_model_output_to_tokenized_sequence(t, h, v, reverse_vocab=reverse_vocab)
        hvo_array = convert_tokenized_sequence_to_hvo_array(tokenized_seq)

        # New HVO Sequence object
        hvo_seq.hvo = hvo_array
        print(hvo_seq.hits.shape)
        if hvo_seq.hits.sum() > 0:
            note_seq = hvo_seq.to_note_sequence()

            # Load into quantized HVO and add to list of output HVO arrays
            hvo_seq_quant = note_sequence_to_hvo_sequence(note_seq,
                                                          drum_mapping=ROLAND_REDUCED_MAPPING,
                                                          beat_division_factors=[4])
            hvo_seq_quant.adjust_length(32)
            output_hvo_arrays.append(hvo_seq_quant.hvo)
        else:
            output_hvo_arrays.append(np.zeros((32, 9)))

    evaluator.add_predictions(np.stack(output_hvo_arrays))

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


