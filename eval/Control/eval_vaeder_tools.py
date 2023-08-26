import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import torch
import os
import wandb
import random
from copy import deepcopy
import numpy as np
import json
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import shutil
import seaborn as sns
from tqdm import tqdm
from data.control.control_utils import calculate_density, calculate_intensity

from model import GrooveControl_VAE

# Umap generates crazy debug statements; this is the only fix I can find
import logging
matplotlib_logger = logging.getLogger('matplotlib')
matplotlib_logger.setLevel(logging.WARNING)
numba_logger = logging.getLogger('numba')
numba_logger.setLevel(logging.WARNING)  # or logging.INFO if you want to see INFO messages
from umap import umap_ as UMAP


def load_vaeder_model(model_path, params_dict=None, is_evaluating=True, device=None,
                      genre_json_path=None):
    try:
        if device is not None:
            loaded_dict = torch.load(model_path, map_location=device)
        else:
            loaded_dict = torch.load(model_path)
    except:
        loaded_dict = torch.load(model_path, map_location=torch.device('cpu'))

    if params_dict is None:
        if 'params' in loaded_dict:
            params_dict = loaded_dict['params']

            # Todo: TEMPORARY FIX
            if isinstance(genre_json_path, str):
                with open(genre_json_path, 'r') as f:
                    genre_dict = json.load(f)
                    params_dict['genre_dict'] = genre_dict

        else:
            raise Exception(f"Could not instantiate model as params_dict is not found. "
                            f"Please provide a params_dict either as a json path or as a dictionary")

    if isinstance(params_dict, str):
        with open(params_dict, 'r') as f:
            params_dict = json.load(f)

    model = GrooveControl_VAE(params_dict)
    model.load_state_dict(loaded_dict["model_state_dict"])
    if is_evaluating:
        model.eval()

    return model


def generate_vaeder_umaps(model, dataset, density_norm_fn=None, intensity_norm_fn=None):
    in_groove = torch.tensor(
        np.array([hvo_seq.flatten_voices(reduce_dim=True)
                  for hvo_seq in dataset.hvo_sequences]), dtype=torch.float32)

    densities = dataset.densities
    intensities = dataset.intensities
    genres = dataset.genres
    genre_dict = model.get_genre_dict()

    if density_norm_fn is not None:
        densities = density_norm_fn(densities)
    if intensity_norm_fn is not None:
        intensities = intensity_norm_fn(intensities)

    dataloader = DataLoader(in_groove, batch_size=128, shuffle=False)
    total_z = []

    for in_hvo in dataloader:
        mu, log_var, latent_z = model.encode(in_hvo)
        total_z.append(latent_z)

    z = torch.cat(total_z, dim=0)
    # z_params = torch.cat((z, densities.unsqueeze(dim=-1), intensities.unsqueeze(dim=-1), genres), dim=1)

    umap_without_params = create_parameters_eval_dataframe(z, densities, intensities, genres, genre_dict)
    # umap_with_params = create_parameters_eval_dataframe(z_params, densities, intensities, genres,
    #                                                     genre_dict)
    # return create_combined_scatter(umap_without_params, umap_with_params)

    return generate_single_scatter(umap_without_params)


def create_parameters_eval_dataframe(latent_z, densities, intensities, genres, genre_dict):
    """
    Given a series of parameter inputs + latent z, created a dataframe that correlates each z space to the
    given parameter
    @param latent_z: (tensor) model z encoding
    @param densities: (tensor) densities
    @param intensities: (tensor) intensities
    @param genres: (tensor) one-hot genre encodings
    @param genre_dict: (dictionary) of type {"label": int} to provide the value of the genre, i.e. {"rock": 2}
    @return: Dataframe with 5 columns: [umap1, umap2, density, intensity, genre]
    """
    # Convert to separate pandas dataframes
    z_df = pd.DataFrame(latent_z.detach().cpu().numpy())
    densities_series = pd.Series(densities.squeeze().detach().cpu().numpy(), name='density')
    intensities_series = pd.Series(intensities.squeeze().detach().cpu().numpy(), name='intensity')
    genres_np = genres.squeeze().detach().cpu().numpy()
    genres_np = np.argmax(genres_np, axis=1)

    # Convert integer genres to string using genre_dict
    rev_genre_dict = {v: k for k, v in genre_dict.items()}
    genres_str = [rev_genre_dict.get(g, "Unknown") for g in genres_np]
    genres_series = pd.Series(genres_str, name='genre')

    # Combine and collapse to 2D umap
    df = pd.concat([z_df, densities_series, intensities_series, genres_series], axis=1)
    reducer = UMAP.UMAP()
    embedding = reducer.fit_transform(df.drop(columns=['density', 'intensity', 'genre']))

    # Create a DataFrame from the embedding
    umap_df = pd.DataFrame(embedding, columns=['UMAP1', 'UMAP2'])
    umap_df['density'] = df['density']
    umap_df['intensity'] = df['intensity']
    umap_df['genre'] = df['genre']

    return umap_df


# def create_combined_scatter(umap_df1, umap_df2):
#     fig, axes = plt.subplots(2, 3, figsize=(21, 14))  # 2 rows, 3 columns
#
#     # List of UMAP DataFrames and titles for the rows
#     umap_dfs = [umap_df1, umap_df2]
#     row_titles = ['Pre Parameter Injection', 'Post Parameter Injection']
#
#     # Features to plot
#     features = ['density', 'intensity', 'genre']
#     col_titles = ['Density', 'Intensity', 'Genre']
#     colormaps = ['coolwarm', 'PiYG', 'tab20']
#
#     for row, (umap_df, row_title) in enumerate(zip(umap_dfs, row_titles)):
#         for col, (feature, col_title, cmap) in enumerate(zip(features, col_titles, colormaps)):
#             ax = axes[row, col]
#             sns.scatterplot(x='UMAP1', y='UMAP2', hue=feature, data=umap_df, palette=cmap, ax=ax)
#
#             # Remove legend box for the first two columns
#             if col != 2:
#                 ax.legend().remove()
#
#             # Set limits, labels, ticks
#             bleed = 0.3
#             ax.set_xlim(umap_df['UMAP1'].min() - bleed, umap_df['UMAP1'].max() + bleed)
#             ax.set_ylim(umap_df['UMAP2'].min() - bleed, umap_df['UMAP2'].max() + bleed)
#             ax.set_xlabel("")
#             ax.set_ylabel("")
#             ax.set_xticks([])
#             ax.set_yticks([])
#
#             # Add color bar for the first two columns only
#             if col != 2:
#                 norm = plt.Normalize(0, 1)
#                 sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
#                 plt.colorbar(sm, label=feature, ax=ax)
#
#         # Add title for each row, centered and larger font size
#         fig.text(0.5, 0.975 - 0.5 * row, row_title, ha='center', fontsize=14)
#
#     plt.tight_layout(rect=[0, 0, 1, 0.96])
#     return fig


def generate_single_scatter(umap_df):
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))  # 1 row, 3 columns

    # List of UMAP DataFrames

    # Features to plot
    features = ['density', 'intensity', 'genre']
    col_titles = ['Density', 'Intensity', 'Genre']
    colormaps = ['coolwarm', 'PiYG', 'tab20']

    for col, (feature, col_title, cmap) in enumerate(zip(features, col_titles, colormaps)):
        ax = axes[col]
        sns.scatterplot(x='UMAP1', y='UMAP2', hue=feature, data=umap_df, palette=cmap, ax=ax)

        # Remove legend box for the first two columns
        if col != 2:
            ax.legend().remove()

        # Set limits, labels, ticks
        bleed = 0.3
        ax.set_xlim(umap_df['UMAP1'].min() - bleed, umap_df['UMAP1'].max() + bleed)
        ax.set_ylim(umap_df['UMAP2'].min() - bleed, umap_df['UMAP2'].max() + bleed)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticks([])
        ax.set_yticks([])

        # Add color bar for the first two columns only
        if col != 2:
            norm = plt.Normalize(0, 1)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            plt.colorbar(sm, label=feature, ax=ax)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig






def generate_vaeder_midi_examples(model_name, model, dataset, sample_indices, genres,
                                  density_norm_fn, intensity_norm_fn):

    densities = [0.1, 0.3, 0.5, 0.7, 0.9]
    intensities = densities
    genre_dict = model.get_genre_dict()
    rev_genre_dict = {v: k for k, v in genre_dict.items()}



    for idx in sample_indices:
        hvo_gt_seq = dataset.get_hvo_sequences_at(idx)
        hvo_gt_seq.save_hvo_to_midi(filename=f"{idx}_gt.mid")
        hvo_tap_seq = deepcopy(hvo_gt_seq)
        hvo_tap_seq.hvo = hvo_gt_seq.flatten_voices(voice_idx=2, reduce_dim=False)
        hvo_tap_seq.save_hvo_to_midi(filename=f"{idx}_tapped.mid")

        gt_density = torch.unsqueeze(density_norm_fn(dataset.densities[idx]), dim=0)
        gt_intensity = torch.unsqueeze(intensity_norm_fn(dataset.intensities[idx]), dim=0)
        gt_genre = torch.unsqueeze(dataset.genres[idx], dim=0)
        input_hvo = torch.unsqueeze(dataset.get_inputs_at(idx), dim=0)

        output_seq = deepcopy(hvo_gt_seq)
        hvo, _, _, _ = model.predict(input_hvo, gt_density, gt_intensity, gt_genre, return_concatenated=True)
        output_seq.hvo = torch.squeeze(hvo, dim=0).numpy()
        output_seq.save_hvo_to_midi(filename=f"{idx}_{model_name}_gt_pred.mid")




        for density in densities:
            output_hvo = np.empty((0, 27))
            d = torch.unsqueeze(torch.tensor(density), dim=0)
            for intensity in intensities:
                i = torch.unsqueeze(torch.tensor(intensity), dim=0)

                hvo, _, _, _ = model.predict(input_hvo, d, i, gt_genre, return_concatenated=True)
                #output_seq.hvo = torch.squeeze(hvo, dim=0).numpy()
                hvo = torch.squeeze(hvo, dim=0).numpy()
                output_hvo = np.concatenate((output_hvo, hvo), axis=0)

            output_seq.hvo = output_hvo


            output_seq.save_hvo_to_midi(filename=f"{idx}_{model_name}_d{density}.mid")

        for genre in genres:
            genre_id = genre_dict[genre]
            genre_input = torch.nn.functional.one_hot(torch.tensor([genre_id]),
                                                      num_classes=len(genre_dict)).to(dtype=torch.float32)
            hvo, _, _, _ = model.predict(input_hvo, gt_density, gt_intensity,
                                         genre_input, return_concatenated=True)
            output_seq.hvo = torch.squeeze(hvo, dim=0).numpy()
            output_seq.save_hvo_to_midi(filename=f"{idx}_{model_name}_genre_{genre}.mid")



def test_control_values(model,  density_norm_fn, intensity_norm_fn, n_genres, n_examples=100):
    batch_size = 50
    hits_list = []
    velocities_list = []

    for i in tqdm(range(0, n_examples, batch_size)):
        batch_n = min(batch_size, n_examples - i)

        # Generate data for this batch
        mean = 0.5
        std_dev = 0.15  # Adjust this value as needed
        density_batch = torch.clamp(mean + std_dev * torch.randn(batch_n), 0.01, 0.99)
        intensity_batch = torch.clamp(mean + std_dev * torch.randn(batch_n), 0.01, 0.99)

        indices_batch = torch.randint(0, 16, (batch_n,))
        genre_batch = torch.nn.functional.one_hot(indices_batch, num_classes=n_genres).to(dtype=torch.float32)

        latent_dim = model.get_latent_dim()
        z_batch = torch.rand(batch_n, latent_dim).to(dtype=torch.float32)
        hits_batch, velocities_batch, _ = model.decode(z_batch, density_batch, intensity_batch, genre_batch)

        mask_batch = hits_batch < 0.5
        velocities_batch[mask_batch] = 0

        hits_list.append(hits_batch)
        velocities_list.append(velocities_batch)

    # Concatenate results from all batches
    hits = torch.cat(hits_list, dim=0)
    velocities = torch.cat(velocities_list, dim=0)

    # Convert tensors to numpy arrays
    hits_array = [hits[i].numpy() for i in range(hits.shape[0])]
    velocities_array = [velocities[i].numpy() for i in range(velocities.shape[0])]

    densities, intensities = [], []

    for h, v in zip(hits_array, velocities_array):
        density = density_norm_fn(calculate_density(h))
        intensity = intensity_norm_fn(calculate_intensity(v))
        densities.append(density)
        intensities.append(intensity)

    # Display as heatmap
    H, xedges, yedges = np.histogram2d(densities, intensities, bins=(30, 30), range=[[0, 1], [0, 1]])
    plt.clf()
    plt.imshow(H.T, interpolation='spline16', origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
               cmap='plasma', aspect='equal')

    plt.title("")
    plt.xlabel("Calculated Density")
    plt.ylabel("Calculated Intensity")

    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)

    # Display the legend
    return plt


def get_ground_truth_control_heatmap(num_samples=100):
    mean = 0.5
    std_dev = 0.15
    densities = np.random.normal(mean, std_dev, num_samples)
    intensities = np.random.normal(mean, std_dev, num_samples)

    densities = np.clip(densities, 0, 1)
    intensities = np.clip(intensities, 0, 1)

    plt.clf()
    H, xedges, yedges = np.histogram2d(densities, intensities, bins=(30, 30), range=[[0, 1], [0, 1]])
    plt.clf()
    plt.imshow(H.T, interpolation='spline16', origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
               cmap='plasma', aspect='equal')

    plt.title("")
    plt.xlabel("Density Distribution")
    plt.ylabel("Intensity Distribution")

    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)

    return plt


def make_empty_folder(folder_name, delete_existing_files=True):
    os.makedirs(folder_name, exist_ok=True)
    # Delete all contents inside the folder
    if delete_existing_files:
        for filename in os.listdir(folder_name):
            file_path = os.path.join(folder_name, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')
    os.chdir(folder_name)





