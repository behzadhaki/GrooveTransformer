import copy
import torch
import numpy as np
from eval.GrooveEvaluator import load_evaluator_template
from eval.UMAP import UMapper
from data.control.control_utils import calculate_density, calculate_intensity
import random
from bokeh.embed import file_html
from bokeh.resources import CDN
from bokeh.models.widgets import Panel, Tabs
from bokeh.io import save
from bokeh.models import DataRange1d
import umap
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import pandas as pd
import os
import wandb


def create_parameters_dataframe(latent_z, densities, intensities, genres):
    # Convert to pandas df
    z_df = pd.DataFrame(latent_z.detach().cpu().numpy())
    densities_series = pd.Series(densities.squeeze().detach().cpu().numpy(), name='density')
    intensities_series = pd.Series(intensities.squeeze().detach().cpu().numpy(), name='intensity')
    genres_np = genres.squeeze().detach().cpu().numpy()
    genres_np = np.argmax(genres_np, axis=1)
    genres_series = pd.Series(genres_np, name='genre')  # this is an integer representation of the genre

    # Combine and collapse to 2D umap
    df = pd.concat([z_df, densities_series, intensities_series, genres_series], axis=1)
    reducer = umap.UMAP()
    embedding = reducer.fit_transform(df.drop(columns=['density', 'intensity']))

    # Create a DataFrame from the embedding
    umap_df = pd.DataFrame(embedding, columns=['UMAP1', 'UMAP2'])
    umap_df['density'] = df['density']
    umap_df['intensity'] = df['intensity']
    umap_df['genre'] = df['genre']

    return umap_df


def generate_density_intensity_umap(model, device, dataset,
                                    collapse_tapped_sequence, density_norm_fn=None, intensity_norm_fn=None):
    print("\nGenerating UMap")
    in_groove = torch.tensor(
        np.array([hvo_seq.flatten_voices(reduce_dim=collapse_tapped_sequence)
                  for hvo_seq in dataset.hvo_sequences]), dtype=torch.float32).to(
        device)

    densities = dataset.densities
    intensities = dataset.intensities
    genres = dataset.genres

    if density_norm_fn is not None:
        densities = density_norm_fn(densities)
    if intensity_norm_fn is not None:
        intensities = intensity_norm_fn(intensities)

    _, _, _, latents_z = model.predict(in_groove.to(device),
                                       densities.to(device),
                                       intensities.to(device),
                                       genres.to(device),
                                       return_concatenated=True)

    umap_df = create_parameters_dataframe(latents_z, densities, intensities, genres)
    plt.clf()
    cmap = mcolors.LinearSegmentedColormap.from_list(
        name='blue_green_red',
        colors=['blue', 'deepskyblue', 'green', 'yellow', 'red']
    )

    # Scale intensity to a suitable range for dot sizes
    size_scale = 500  # Adjust as needed
    umap_df['size'] = umap_df['intensity'] * size_scale
    plt.figure(figsize=(14, 12))
    scatter = sns.scatterplot(x='UMAP1', y='UMAP2', hue='density', size='size', sizes=(20, 300), palette=cmap,
                              data=umap_df, legend=False)

    # Remove axis labels and ticks
    plt.xlabel("")
    plt.ylabel("")
    plt.xticks([])
    plt.yticks([])

    # Scale so the data goes to the edges
    bleed = 0.3
    # scatter.set_xlim(umap_df['UMAP1'].min() - bleed, umap_df['UMAP1'].max() + bleed)
    # scatter.set_ylim(umap_df['UMAP2'].min() - bleed, umap_df['UMAP2'].max() + bleed)

    plt.xlim(umap_df['UMAP1'].min() - bleed, umap_df['UMAP1'].max() + bleed)
    plt.ylim(umap_df['UMAP2'].min() - bleed, umap_df['UMAP2'].max() + bleed)

    # Background color
    plt.gca().set_facecolor('snow')

    # Legend color bar
    norm = plt.Normalize(0, 1)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    plt.colorbar(sm, label="Density")

    plt.title("")
    plt.style.use("default")
    img = wandb.Image(plt)

    return {"parameters_umap": img}


def get_piano_rolls(patterns, save_path=None, prepare_for_wandb=True, x_range_pad=0.5):
    full_figure = []

    for idx, pattern in enumerate(patterns):
        density_panels = []

        pattern_title = f"Sample_{idx}: " + str(pattern["input"].metadata['master_id']) if "master_id" in pattern[
            "input"].metadata.keys() \
            else f"Sample_{idx}"

        # Add the 'ground truth' tab
        p_roll_gt = pattern["input"].to_html_plot()
        p_roll_gt.x_range = DataRange1d(range_padding=x_range_pad)
        ground_truth_panel = Panel(child=p_roll_gt, title="Ground Truth")
        density_panels.append(ground_truth_panel)

        # Add other 'Density' tabs
        for density_label, density_data in pattern.items():
            if density_label != "input":
                intensity_panels = []

                for intensity_label, sequence in density_data.items():
                    p_roll = sequence.to_html_plot()
                    p_roll.x_range = DataRange1d(range_padding=x_range_pad)
                    value = intensity_label.split(":")[
                        1].strip() if ":" in intensity_label else intensity_label.strip()
                    title = "intensity: " + str(round(float(value), 2))

                    intensity_panels.append(Panel(child=p_roll, title=title))

                density_tabs = Tabs(tabs=intensity_panels)
                density_panels.append(Panel(child=density_tabs, title=f"density: {density_label}"))

        pattern_tabs = Tabs(tabs=density_panels)
        full_figure.append(Panel(child=pattern_tabs, title=pattern_title))

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


def get_piano_rolls_for_control_model(vae_model, dataset, device, genre_mapping_dict, num_examples=2,
                                      density_normalizing_fn=None, intensity_normalizing_fn=None,
                                      reduce_dim=True):
    """
    Obtain a selection of input and output piano rolls for evaluation purposes on the Density
    control models. Intended to provide the ground truth HVO, the prediction when passed the 'real' (normalized)
    density, as well predicts when given other density values. This can help determine the models
    ability to change its behavior based on different control values.
    """
    print("\nGenerating Piano Rolls")

    # get a random selection of sequences
    hvo_seq_set = dataset.get_hvo_sequences()
    density_values = [0.0, 0.01, 0.5, 0.99]
    intensity_values = [0.0, 0.01, 0.5, 0.99]
    hvo_sequences = random.sample(hvo_seq_set, num_examples)

    patterns = []
    n_examples = len(density_values) * len(intensity_values)

    for hvo_seq in hvo_sequences:
        # hvo_sequence_dict = {"input": hvo_seq}

        density_values[0] = density_normalizing_fn(calculate_density(hvo_seq.hits)) \
            if density_normalizing_fn is not None else calculate_density(hvo_seq.hits)
        densities = torch.tensor(density_values, dtype=torch.float32).to(device)
        densities = densities.unsqueeze(1).repeat(1, 4).view(-1)

        intensity_values[0] = intensity_normalizing_fn(calculate_intensity(hvo_seq.hvo[:, 9:18])) \
            if intensity_normalizing_fn is not None else calculate_intensity(hvo_seq.hvo[:, 9:18])
        intensities = torch.tensor(intensity_values, dtype=torch.float32).to(device)
        intensities = intensities.repeat(len(density_values))

        # Obtain the genre of the sample
        genre = genre_mapping_dict.get(hvo_seq.metadata["style_primary"], genre_mapping_dict["other"])
        genre = torch.nn.functional.one_hot(torch.tensor(genre), num_classes=len(genre_mapping_dict))
        genre = genre.repeat(n_examples, 1).to(dtype=torch.float32)

        # Create a batch of duplicate inputs based on number of densities we are testing
        in_grooves = torch.tensor(
            np.repeat(hvo_seq.flatten_voices(reduce_dim=reduce_dim)[np.newaxis, :], n_examples, axis=0),
            dtype=torch.float32)

        outputs, mu, log_var, latent_z = vae_model.predict(in_grooves.to(device), densities.to(device),
                                                           intensities.to(device), genre.to(device),
                                                           return_concatenated=True)
        output_hvo_arrays = [seq.squeeze(0).cpu().numpy() for seq in torch.split(outputs, 1, dim=0)]

        # Create nested dictionary
        hvo_sequence_dict = {"input": hvo_seq}
        output_idx = 0
        for d in density_values:
            density_key = f"{d:.2f}"  # Unique key for each density value

            density_dict = {}
            for i in intensity_values:
                intensity_key = f"{i:.2f}"  # Unique key for each intensity value

                output_hvo = copy.deepcopy(hvo_seq)
                output_hvo.hvo = output_hvo_arrays[output_idx]

                density_dict[intensity_key] = output_hvo
                output_idx += 1

            hvo_sequence_dict[density_key] = density_dict

        patterns.append(hvo_sequence_dict)

    piano_rolls = dict()
    piano_rolls["piano_rolls"] = get_piano_rolls(patterns)

    return piano_rolls


def get_hit_scores_for_control_model(model, device, dataset_setting_json_path, subset_name,
                                     down_sampled_ratio, collapse_tapped_sequence, genre_mapping_dict,
                                     density_normalizing_fn=None,
                                     intensity_normalizing_fn=None,
                                     cached_folder="eval/GrooveEvaluator/templates/",
                                     divide_by_genre=False):
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
    intensities = torch.zeros(len(hvo_seqs), dtype=torch.float32)
    genres = torch.zeros((len(hvo_seqs), len(genre_mapping_dict)), dtype=torch.float32)

    for idx, hvo_seq in enumerate(hvo_seqs):
        densities[idx] = density_normalizing_fn(calculate_density(hvo_seq.hits)) if density_normalizing_fn is not None \
            else calculate_density(hvo_seq.hits)
        intensities[idx] = intensity_normalizing_fn(calculate_intensity(hvo_seq.hvo[9:18])) \
            if intensity_normalizing_fn is not None else calculate_intensity(hvo_seq.hvo[9:18])
        genre = genre_mapping_dict.get(hvo_seq.metadata["style_primary"], genre_mapping_dict["other"])
        genre = torch.nn.functional.one_hot(torch.tensor(genre), num_classes=len(genre_mapping_dict))
        genres[idx] = genre.to(dtype=torch.float32)


    predictions = []

    # batchify the input
    model.eval()
    with torch.no_grad():
        for batch_ix, (hvo_batch_in, density_batch_in, intensity_batch_in, genre_batch_in) \
                in enumerate(zip(torch.split(in_groove, 32),
                                 torch.split(densities, 32),
                                 torch.split(intensities, 32),
                                 torch.split(genres, 32))):
            hvos_array, _, _, _ = model.predict(
                hvo_batch_in.to(device),
                density_batch_in.to(device),
                intensity_batch_in.to(device),
                genre_batch_in.to(device),
                return_concatenated=True)
            predictions.append(hvos_array.detach().cpu().numpy())

    evaluator.add_predictions(np.concatenate(predictions))
    hit_dict = evaluator.get_statistics_of_pos_neg_hit_scores()

    score_dict = {f"{subset_name}/{key}_mean".replace(" ", "_").replace("-", "_"): float(value['mean']) for key, value
                  in hit_dict.items()}
    score_dict.update(
        {f"{subset_name}/{key}_std".replace(" ", "_").replace("-", "_"): float(value['std']) for key, value in
         hit_dict.items()})
    return score_dict


def get_control_model_density_prediction_averages(model, dataset, device, n_genres,
                                                  batch_size=64,
                                                  density_normalizing_fn=None,
                                                  reduce_dim=True):
    print("\n GEN DENSITIES")
    hvo_seq_set = dataset.get_hvo_sequences()
    hvo_sequences = random.sample(hvo_seq_set, batch_size)
    in_grooves = torch.tensor(np.array([hvo_seq.flatten_voices(reduce_dim=reduce_dim) for hvo_seq in hvo_sequences]),
                              dtype=torch.float32)

    densities = [0.01, 0.5, 0.99]
    predicted_densities = {}

    for density in densities:

        density_inputs = torch.full((batch_size,), density, dtype=torch.float32)
        intensity_inputs = torch.rand(batch_size, dtype=torch.float32)
        genre_inputs = create_random_genre_onehot_inputs(batch_size, n_genres).to(dtype=torch.float32)

        predictions, _, _, _ = model.predict(
            in_grooves.to(device),
            density_inputs.to(device),
            intensity_inputs.to(device),
            genre_inputs.to(device),
            return_concatenated=True)
        hits = predictions[:, :, :9]

        num_hits = torch.sum(hits)
        num_potential_hits = torch.numel(hits)
        predictions_density = (num_hits / num_potential_hits).item()

        if density_normalizing_fn is not None:
            predictions_density = min(density_normalizing_fn(predictions_density), 1.0)
        predictions_density = round(predictions_density, 2)
        title = "Density: " + str(density)
        predicted_densities[title] = predictions_density

    return predicted_densities


def get_control_model_intensity_prediction_averages(model, dataset, device, n_genres,
                                                    batch_size=64,
                                                    intensity_normalizing_fn=None,
                                                    reduce_dim=True):
    print("\n Generating Intensity Predictions")
    hvo_seq_set = dataset.get_hvo_sequences()
    hvo_sequences = random.sample(hvo_seq_set, batch_size)
    in_grooves = torch.tensor(np.array([hvo_seq.flatten_voices(reduce_dim=reduce_dim) for hvo_seq in hvo_sequences]),
                              dtype=torch.float32)

    intensities = [0.01, 0.5, 0.99]
    predicted_intensities = {}

    for intensity in intensities:
        density_inputs = torch.rand(batch_size, dtype=torch.float32)
        intensity_inputs = torch.full((batch_size,), intensity, dtype=torch.float32)
        genre_inputs = create_random_genre_onehot_inputs(batch_size, n_genres).to(dtype=torch.float32)

        predictions, _, _, _ = model.predict(
            in_grooves.to(device),
            density_inputs.to(device),
            intensity_inputs.to(device),
            genre_inputs.to(device),
            return_concatenated=False)

        # Remove non-hit velocity values
        hits, velocities, _ = predictions
        mask = hits > 0.5
        velocities_masked = velocities[mask]
        predictions_intensity = velocities_masked.mean().item()

        if intensity_normalizing_fn is not None:
            predictions_intensity = min(intensity_normalizing_fn(predictions_intensity), 1.0)
        predictions_intensity = round(predictions_intensity, 2)
        title = "Intensity: " + str(intensity)
        predicted_intensities[title] = predictions_intensity

    return predicted_intensities


def create_random_genre_onehot_inputs(batch_size, n_genres):
    one_hot = torch.zeros(batch_size, n_genres)
    indices = torch.randint(0, n_genres, (batch_size,))
    one_hot[torch.arange(batch_size), indices] = 1
    return one_hot
