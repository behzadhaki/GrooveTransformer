import copy
import torch
import numpy as np
from hvo_sequence import HVO_Sequence
from hvo_sequence import ROLAND_REDUCED_MAPPING
from model import GrooveTransformerEncoderVAE
from eval.GrooveEvaluator import load_evaluator_template
from eval.UMAP import UMapper
from data.control.control_utils import calculate_density
import random
from bokeh.embed import file_html
from bokeh.resources import CDN
from bokeh.models.widgets import Panel, Tabs
from bokeh.io import save
from bokeh.models import DataRange1d
import os
import wandb

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
    #densities = torch.zeros(len(test_dataset.hvo_sequences), dtype=torch.float32).to(device)
    # for idx, hvo_seq in enumerate(test_dataset.hvo_sequences):
    #     densities[idx] = calculate_density(hvo_seq.hits)

    densities = test_dataset.densities

    # #tags = [(hvo_seq.metadata["style_primary"], density) for hvo_seq, density in zip(test_dataset.hvo_sequences, densities)]
    # tags = [str(round(density.item(), 2)) for density in densities]

    tags = []
    density_buckets = [0.01, 0.25, 0.5, 0.75, 0.99]
    for density in densities:
        rounded_density = min(density_buckets, key=lambda x: abs(x-density))
        tags.append(str(rounded_density))


    _, _, _, latents_z = model.predict(in_groove, densities, return_concatenated=True)

    umapper = UMapper(subset_name)
    umapper.fit(latents_z.detach().cpu().numpy(), tags_=tags)
    p = umapper.plot(show_plot=False, prepare_for_wandb=True)
    return {f"{subset_name}_umap": p}


def get_piano_rolls(patterns, save_path=None, prepare_for_wandb=True, x_range_pad=0.5):

    # patterns: list of dicts of type {definition: sequences} i.e. {'ground_truth': hvo_sequence}
    full_figure = []
    subset_panels = []

    for idx, pattern in enumerate(patterns):

        sample_panels = []
        pattern_title = f"Sample_{idx}: " + str(pattern["input"].metadata['master_id']) if "master_id" in pattern[
            "input"].metadata.keys() \
            else f"Sample_{idx}"

        if "input" in pattern.keys():
            p_roll = pattern["input"].to_html_plot(filename=pattern_title)
            p_roll.x_range = DataRange1d(range_padding=x_range_pad)
            sample_panels.append(Panel(child=p_roll, title="GT"))

        for label, sequence in pattern.items():
            if label != "input":
                p_roll = sequence.to_html_plot()
                p_roll.x_range = DataRange1d(range_padding=x_range_pad)
                rounded_label = round(float(label), 2)
                title = "density: " + str(rounded_label)
                sample_panels.append(Panel(child=p_roll, title=title))

        sample_tabs = Tabs(tabs=sample_panels)
        subset_panels.append(Panel(child=sample_tabs, title=pattern_title))

    full_figure.append(Panel(child=Tabs(tabs=subset_panels), title="Complete Set"))
    final_tabs = Tabs(tabs=full_figure)

    if save_path is not None:
        # make sure file ends with .html
        if not save_path.endswith(".html"):
            save_path += ".html"

        # make sure directory exists
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

        save(final_tabs, save_path)

    return final_tabs if prepare_for_wandb is False else wandb.Html(file_html(final_tabs, CDN, "Piano Rolls"))

def get_piano_rolls_for_control_model_wandb(vae_model, test_dataset, device,
                                            num_examples=8, normalizing_fn=None):
    """
    Obtain a selection of input and output piano rolls for evaluation purposes on the Density
    control models. Intended to provide the ground truth HVO, the prediction when passed the 'real' (normalized)
    density, as well predicts when given other density values. This can help determine the models
    ability to change its behavior based on different control values.
    """

    # get a random selection of sequences
    hvo_seq_set = test_dataset.get_hvo_sequences()
    density_values = [0.0, 0.01, 0.5, 0.99]
    hvo_sequences = random.sample(hvo_seq_set, num_examples)

    patterns = []

    for hvo_seq in hvo_sequences:
        hvo_sequence_dict = {}
        hvo_sequence_dict["input"] = hvo_seq

        density_values[0] = normalizing_fn(calculate_density(hvo_seq.hits)) if normalizing_fn is not None \
                else calculate_density(hvo_seq.hits)
        densities = torch.tensor(density_values, dtype=torch.float32).to(device)

        # Create a batch of duplicate inputs based on number of densities we are testing
        in_grooves = torch.tensor(np.repeat(hvo_seq.flatten_voices(reduce_dim=True)[np.newaxis, :], len(density_values), axis=0),
                                  dtype=torch.float32)

        outputs, mu, log_var, latent_z = vae_model.predict(in_grooves.to(device), densities.to(device), return_concatenated=True)
        output_hvo_arrays = [seq.squeeze(0).cpu().numpy() for seq in torch.split(outputs, 1, dim=0)]

        for density, prediction in zip(density_values, output_hvo_arrays):
            output_hvo = copy.deepcopy(hvo_seq)
            output_hvo.hvo = prediction
            hvo_sequence_dict[str(density)] = output_hvo

        patterns.append(hvo_sequence_dict)

    piano_rolls = dict()
    piano_rolls["piano_rolls"] = get_piano_rolls(patterns)

    return piano_rolls


def get_hit_scores_for_density_model(model, device, dataset_setting_json_path, subset_name,
                                     down_sampled_ratio, collapse_tapped_sequence,
                                     normalizing_fn=None,
                                     cached_folder="eval/GrooveEvaluator/templates/",
                                     divide_by_genre=True):


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

    densities = torch.zeros(len(hvo_seqs), dtype=torch.float32)
    for idx, hvo_seq in enumerate(hvo_seqs):
        densities[idx] = normalizing_fn(calculate_density(hvo_seq.hits)) if normalizing_fn is not None \
            else calculate_density(hvo_seq.hits)


    predictions = []

    # batchify the input
    model.eval()
    with torch.no_grad():
        for batch_ix, (hvo_batch_in, density_batch_in)\
                in enumerate(zip(torch.split(in_groove, 32), torch.split(densities, 32))):

            hvos_array, _, _, _ = model.predict(
                hvo_batch_in.to(device),
                density_batch_in.to(device),
                return_concatenated=True)
            predictions.append(hvos_array.detach().cpu().numpy())

    evaluator.add_predictions(np.concatenate(predictions))

    hit_dict = evaluator.get_statistics_of_pos_neg_hit_scores()

    score_dict = {f"{subset_name}/{key}_mean".replace(" ","_").replace("-","_"): float(value['mean']) for key, value in hit_dict.items()}
    score_dict.update({f"{subset_name}/{key}_std".replace(" ","_").replace("-","_"): float(value['std']) for key, value in hit_dict.items()})
    return score_dict

def get_density_prediction_averages(model, test_dataset, device, batch_size=64, normalizing_fn=None):

    hvo_seq_set = test_dataset.get_hvo_sequences()
    hvo_sequences = random.sample(hvo_seq_set, batch_size)
    in_grooves = torch.tensor(np.array([hvo_seq.flatten_voices(reduce_dim=True) for hvo_seq in hvo_sequences]), dtype=torch.float32)

    densities = [0.01, 0.5, 0.99]

    predicted_densities = {}

    for density in densities:

        density_inputs = torch.full((batch_size,), density, dtype=torch.float32)

        predictions, _, _, _ = model.predict(
            in_grooves.to(device),
            density_inputs.to(device),
            return_concatenated=True)

        hits = predictions[:, :, :9]
        num_hits = torch.sum(hits)
        num_potential_hits = torch.numel(hits)
        predictions_density = (num_hits / num_potential_hits).item()

        if normalizing_fn is not None:
            predictions_density = min(normalizing_fn(predictions_density), 1.0)

        predictions_density = round(predictions_density, 2)

        title = "Density: " + str(density)
        predicted_densities[title] = predictions_density

    return predicted_densities



