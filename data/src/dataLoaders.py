from data.src.utils import get_data_directory_using_filters, get_drum_mapping_using_label, load_original_gmd_dataset_pickle, extract_hvo_sequences_dict, pickle_hvo_dict
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from math import ceil
import json
import os
import pickle
import bz2
import logging
logging.basicConfig(level=logging.DEBUG)
dataLoaderLogger = logging.getLogger("data.Base.dataLoaders")

try:
    import pandas as pd
    from bokeh.palettes import Category20c
    from bokeh.plotting import figure, show, output_file, gridplot
    from bokeh.transform import cumsum
    import holoviews as hv
    from holoviews import opts
    hv.extension('bokeh')
    _CAN_PLOT = True
except ImportError:
    dataLoaderLogger.warning("Cannot import plotting libraries. Dataset Statistics Can't be Plotted.")
    _CAN_PLOT = False


def load_gmd_hvo_sequences(dataset_setting_json_path, subset_tag, force_regenerate=False):
    """
    Loads the hvo_sequences using the settings provided in the json file.

    :param dataset_setting_json_path: path to the json file containing the dataset settings (see data/dataset_json_settings/4_4_Beats_gmd.json)
    :param subset_tag: [str] whether to load the train/test/validation set
    :param force_regenerate:
    :return:
    """

    # load settings
    dataset_setting_json = json.load(open(dataset_setting_json_path, "r"))

    # load datasets
    dataset_tags = [key for key in dataset_setting_json["settings"].keys()]

    for dataset_tag in dataset_tags:
        dataLoaderLogger.info(f"Loading {dataset_tag} dataset")
        raw_data_pickle_path = dataset_setting_json["raw_data_pickle_path"][dataset_tag]
        assert os.path.exists(raw_data_pickle_path), "path to gmd dict pickle is incorrect --- " \
                                                "look into data/gmd/resources/storedDicts/groove-*.bz2pickle"
        dir__ = get_data_directory_using_filters(dataset_tag, dataset_setting_json_path)
        beat_division_factor = dataset_setting_json["global"]["beat_division_factor"]
        drum_mapping_label = dataset_setting_json["global"]["drum_mapping_label"]

        if (not os.path.exists(dir__)) or force_regenerate is True:
            dataLoaderLogger.info(f"No Cached Version Available Here: {dir__}. ")
            dataLoaderLogger.info(
                f"extracting data from raw pickled midi/note_sequence/metadata dictionaries at {raw_data_pickle_path}")
            gmd_dict = load_original_gmd_dataset_pickle(raw_data_pickle_path)
            drum_mapping = get_drum_mapping_using_label(drum_mapping_label)
            hvo_dict = extract_hvo_sequences_dict(gmd_dict, beat_division_factor, drum_mapping)
            pickle_hvo_dict(hvo_dict, dataset_tag, dataset_setting_json_path)
            dataLoaderLogger.info(f"Cached Version available at {dir__}")
        else:
            dataLoaderLogger.info(f"Loading Cached Version from: {dir__}")

        ifile = bz2.BZ2File(os.path.join(dir__, f"{subset_tag}.bz2pickle"), 'rb')
        data = pickle.load(ifile)
        ifile.close()

    return data


class MonotonicGrooveDataset(Dataset):
    def __init__(self, dataset_setting_json_path, subset_tag, max_len, tapped_voice_idx=2,
                 collapse_tapped_sequence=False, load_as_tensor=True, sort_by_metadata_key=None,
                 down_sampled_ratio=None, move_all_to_gpu=False, genre_loss_balancing_beta=0,
                 voice_loss_balancing_beta=0):
        """

        :param dataset_setting_json_path:   path to the json file containing the dataset settings (see data/dataset_json_settings/4_4_Beats_gmd.json)
        :param subset_tag:                [str] whether to load the train/test/validation set
        :param max_len:              [int] maximum length of the sequences to be loaded
        :param tapped_voice_idx:    [int] index of the voice to be tapped (default is 2 which is usually closed hat)
        :param collapse_tapped_sequence:  [bool] returns a Tx3 tensor instead of a Tx(3xNumVoices) tensor
        :param load_as_tensor:      [bool] loads the data as a tensor of torch.float32 instead of a numpy array
        :param sort_by_metadata_key: [str] sorts the data by the metadata key provided (e.g. "tempo")
        :param genre_loss_balancing_beta: [float] beta parameter for balancing the loss according to the genre
                (see:
                            References:
                                     https://arxiv.org/pdf/1901.05555.pdf
                )
        :param voice_loss_balancing_beta [float] beta parameter for balancing the loss according to the voice
                (see:
                            References:
                                        https://arxiv.org/pdf/1901.05555.pdf
                )
        """

        # Get processed inputs, outputs and hvo sequences
        self.inputs = list()
        self.outputs = list()
        self.hvo_sequences = list()

        if down_sampled_ratio is None:
            subset = load_gmd_hvo_sequences(dataset_setting_json_path, subset_tag, force_regenerate=False)
        else:
            subset = load_down_sampled_gmd_hvo_sequences(
                dataset_setting_json_path=dataset_setting_json_path,
                subset_tag=subset_tag,
                force_regenerate=False,
                down_sampled_ratio=down_sampled_ratio,
                cache_down_sampled_set=True
            )

        if sort_by_metadata_key:
            if sort_by_metadata_key in subset[0].metadata[sort_by_metadata_key]:
                subset = sorted(subset, key=lambda x: x.metadata[sort_by_metadata_key])

        # collect input tensors, output tensors, and hvo_sequences
        for idx, hvo_seq in enumerate(tqdm(subset)):
            if hvo_seq.hits is not None:
                hvo_seq.adjust_length(max_len)
                if np.any(hvo_seq.hits):
                    # Ensure all have a length of max_len
                    self.hvo_sequences.append(hvo_seq)
                    self.outputs.append(hvo_seq.hvo)
                    flat_seq = hvo_seq.flatten_voices(voice_idx=tapped_voice_idx, reduce_dim=collapse_tapped_sequence)
                    self.inputs.append(flat_seq)

        # calculate the class balanced weights
        genres_count_dict = self.get_genre_distributions_dict()
        self.genre_balancing_loss_weighting_factor = []
        for hvo_seq in self.hvo_sequences:
            genre = hvo_seq.metadata["style_primary"]
            self.genre_balancing_loss_weighting_factor.append(
                (1 - genre_loss_balancing_beta) / (1 - genre_loss_balancing_beta ** genres_count_dict[genre]))


        # calculate the voice balanced weights
        # There are n_voices available, to balance them, we calculate the effective weight loss for each voice
        # by using the formula:
        #   w_voice_i = (1 - beta) / (1 - beta**(num_examples with this voice existing)) ))
        voice_counts_dict=self.get_voice_counts()
        self.voice_balancing_loss_weighting_factor = []
        for voice in range(self.hvo_sequences[0].hits.shape[1]):
            self.voice_balancing_loss_weighting_factor.append(
                (1 - voice_loss_balancing_beta) / (1 - voice_loss_balancing_beta ** voice_counts_dict[voice]))

        # convert to tensors if required
        self.inputs = torch.tensor(np.array(self.inputs), dtype=torch.float32) \
            if load_as_tensor else np.array(self.inputs)
        self.outputs = torch.tensor(np.array(self.outputs), dtype=torch.float32) \
            if load_as_tensor else np.array(self.outputs)
        self.genre_balancing_loss_weighting_factor = \
            torch.tensor(np.array(self.genre_balancing_loss_weighting_factor), dtype=torch.float32) \
                if load_as_tensor else np.array(self.genre_balancing_loss_weighting_factor)
        self.voice_balancing_loss_weighting_factor = \
            torch.tensor(np.array(self.voice_balancing_loss_weighting_factor), dtype=torch.float32) \
                if load_as_tensor else np.array(self.voice_balancing_loss_weighting_factor)

        # move to gpu if required
        if move_all_to_gpu and torch.cuda.is_available():
            self.inputs = self.inputs.to('cuda')
            self.outputs = self.outputs.to('cuda')

        dataLoaderLogger.info(f"Loaded {len(self.inputs)} sequences")

    def __len__(self):
        return len(self.hvo_sequences)

    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx], self.genre_balancing_loss_weighting_factor[idx], idx

    def get_hvo_sequences_at(self, idx):
        return self.hvo_sequences[idx]

    def get_inputs_at(self, idx):
        return self.inputs[idx]

    def get_outputs_at(self, idx):
        return self.outputs[idx]

    def get_global_hit_count_ratios(self):
        return self.outputs[:, :, :(self.outputs.shape[-1]//3)].sum(dim=0) / self.outputs.shape[0]

    def visualize_global_hit_count_ratios_heatmap(self, show_fig=True):
        if not _CAN_PLOT:
            return None
        hit_ratios = self.get_global_hit_count_ratios().round(decimals=3)
        voice_tags = list(self.hvo_sequences[0].drum_mapping.keys())
        aggregate_data = []
        for t_ in range(hit_ratios.shape[0]):
            for voice_ in range(hit_ratios.shape[1]):
                aggregate_data.append((t_, voice_tags[voice_], hit_ratios[t_, voice_].item()))

        output_file("global_hit_count_ratios.html")
        hm_sessions = hv.HeatMap(aggregate_data)
        hm_sessions = hm_sessions * hv.Labels(hm_sessions).opts(padding=0, text_color='white', text_font_size='6pt')
        fig_sessions = hv.render(hm_sessions, backend='bokeh')
        fig_sessions.plot_height = 400
        fig_sessions.plot_width = 1200
        fig_sessions.title = f"Hit Count Ratios (Total Hits at each location divided by total number of sequences)"
        fig_sessions.xaxis.axis_label = ""
        fig_sessions.yaxis.axis_label = ""
        fig_sessions.xaxis.major_label_orientation = 45
        fig_sessions.xaxis.major_label_text_font_size = "6pt"
        fig_sessions.yaxis.major_label_text_font_size = "6pt"

        if show_fig:
            show(fig_sessions)

        return fig_sessions

    def visualize_genre_distributions(self, show_fig=True, show_inverted_weights=True):
        output_file("genre_distributions.html")
        def get_hist(hist_dict, title):
            p = figure(title="Genre Distributions", toolbar_location="below")
            xtick_locs = np.arange(0.5, len(hist_dict), 1)
            hist = np.array(list(hist_dict.values()))
            p.quad(top=hist, bottom=0, left=xtick_locs - 0.3, right=xtick_locs + 0.3, line_color="white")
            p.xaxis.ticker = xtick_locs
            p.xaxis.major_label_overrides = {xtick_locs[i]: list(hist_dict.keys())[i] for i in range(len(xtick_locs))}
            p.xaxis.major_label_orientation = 45
            return p

        genres = [hvo_seq.metadata["style_primary"] for hvo_seq in self.hvo_sequences]
        hist_dict = {genre: genres.count(genre)/len(genres) for genre in set(genres)}

        p1 = get_hist(hist_dict, "Genre Distributions")

        if show_inverted_weights:
            # balance the histogram
            hist_dict = {genre: 1 - genres.count(genre)/len(genres) for genre in set(genres)}
            p2 = get_hist(hist_dict, "Inverted Genre Distributions")
            # align the y-axis
            p2.y_range = p1.y_range
            fig = gridplot([[p1], [p2]])
        else:
            fig = gridplot([[p1]])

        if show_fig:
            show(fig)

        return fig

    def get_genre_distributions_dict(self, use_ratios=True):
        genres = [hvo_seq.metadata["style_primary"] for hvo_seq in self.hvo_sequences]
        hist_dict = {genre: genres.count(genre) for genre in set(genres)}
        if use_ratios:
            min_val = min(hist_dict.values())
            hist_dict = {genre: hist_dict[genre]/min_val for genre in hist_dict}
        return hist_dict

    def get_voice_counts(self, use_ratios=True):
        voice_count_dict = {
            voice_idx: 0 for voice_idx in range(self.hvo_sequences[0].hits.shape[-1])
        }
        for hvo_seq in self.hvo_sequences:
            for voice_idx in range(hvo_seq.hits.shape[-1]):
                if hvo_seq.hits[:, voice_idx].sum()>0:
                    voice_count_dict[voice_idx] += 1
        if use_ratios:
            min_val = min(voice_count_dict.values())
            voice_count_dict = {voice_idx: voice_count_dict[voice_idx]/min_val for voice_idx in voice_count_dict}

        return voice_count_dict


# ---------------------------------------------------------------------------------------------- #
# loading a down sampled dataset
# ---------------------------------------------------------------------------------------------- #


def down_sample_dataset(hvo_seq_list, down_sample_ratio):
    """
    Down sample the dataset by a given ratio, the ratio of the performers and the ratio of the performances
    are kept the same as much as possible.
    :param hvo_seq_list:
    :param down_sample_ratio:
    :return:
    """
    down_sampled_size = ceil(len(hvo_seq_list) * down_sample_ratio)

    # =================================================================================================
    # Divide up the performances by performer
    # =================================================================================================
    per_performer_per_performance_data = dict()
    for hs in tqdm(hvo_seq_list):
        performer = hs.metadata["drummer"]
        performance_id = hs.metadata["master_id"]
        if performer not in per_performer_per_performance_data:
            per_performer_per_performance_data[performer] = {}
        if performance_id not in per_performer_per_performance_data[performer]:
            per_performer_per_performance_data[performer][performance_id] = []
        per_performer_per_performance_data[performer][performance_id].append(hs)

    # =================================================================================================
    # Figure out how many loops to grab from each performer
    # =================================================================================================
    def flatten(l):
        if isinstance(l[0], list):
            return [item for sublist in l for item in sublist]
        else:
            return l

    ratios_to_other_performers = dict()

    # All samples per performer
    existing_sample_ratios_by_performer = dict()
    for performer, performances in per_performer_per_performance_data.items():
        existing_sample_ratios_by_performer[performer] = \
            len(flatten([performances[p] for p in performances])) / len(hvo_seq_list)

    new_samples_per_performer = dict()
    for performer, ratio in existing_sample_ratios_by_performer.items():
        samples = ceil(down_sampled_size * ratio)
        if samples > 0:
            new_samples_per_performer[performer] = samples

    # =================================================================================================
    # Figure out for each performer, how many samples to grab from each performance
    # =================================================================================================
    num_loops_from_each_performance_compiled_for_all_performers = dict()
    for performer, performances in per_performer_per_performance_data.items():
        total_samples = len(flatten([performances[p] for p in performances]))
        if performer in new_samples_per_performer:
            needed_samples = new_samples_per_performer[performer]
            num_loops_from_each_performance = dict()
            for performance_id, hs_list in performances.items():
                samples_to_select = ceil(needed_samples * len(hs_list) / total_samples)
                if samples_to_select > 0:
                    num_loops_from_each_performance[performance_id] = samples_to_select
            if num_loops_from_each_performance:
                num_loops_from_each_performance_compiled_for_all_performers[performer] = \
                    num_loops_from_each_performance

    # =================================================================================================
    # Sample required number of loops from each performance
    # THE SELECTION IS DONE BY RANKING THE TOTAL NUMBER OF HITS / TOTAL NUMBER OF VOICES ACTIVE
    # then selecting N equally spaced samples from the ranked list
    # =================================================================================================
    for performer, performances in per_performer_per_performance_data.items():
        for performance, hvo_seqs in performances.items():
            seqs_sorted = sorted(
                hvo_seqs,
                key=lambda x: x.hits.sum() / x.get_number_of_active_voices(), reverse=True)
            indices = np.linspace(
                0,
                len(seqs_sorted) - 1,
                num_loops_from_each_performance_compiled_for_all_performers[performer][performance],
                dtype=int)
            per_performer_per_performance_data[performer][performance] = [seqs_sorted[i] for i in indices]

    downsampled_hvo_sequences = []
    for performer, performances in per_performer_per_performance_data.items():
        for performance, hvo_seqs in performances.items():
            downsampled_hvo_sequences.extend(hvo_seqs)

    return downsampled_hvo_sequences


def load_down_sampled_gmd_hvo_sequences(
        dataset_setting_json_path, subset_tag, down_sampled_ratio, cache_down_sampled_set=True, force_regenerate=False):
    """
    Loads the hvo_sequences using the settings provided in the json file.

    :param dataset_setting_json_path: path to the json file containing the dataset settings (see data/dataset_json_settings/4_4_Beats_gmd.json)
    :param subset_tag: [str] whether to load the train/test/validation set
    :param down_sampled_ratio: [float] the ratio of the dataset to downsample to
    :param cache_downsampled_set: [bool] whether to cache the down sampled dataset
    :param force_regenerate: [bool] if True, will regenerate the hvo_sequences from the raw data regardless of cache
    :return:
    """
    dataset_tag = "gmd"
    dir__ = get_data_directory_using_filters(dataset_tag,
                                             dataset_setting_json_path,
                                             down_sampled_ratio=down_sampled_ratio)
    if (not os.path.exists(dir__)) or force_regenerate is True or cache_down_sampled_set is False:
        dataLoaderLogger.info(f"No Cached Version Available Here: {dir__}. ")
        dataLoaderLogger.info(f"Downsampling the dataset to {down_sampled_ratio} and saving to {dir__}.")

        down_sampled_dict = {}
        for subset_tag in ["train", "validation", "test"]:
            hvo_seq_set = load_gmd_hvo_sequences(
                dataset_setting_json_path=dataset_setting_json_path,
                subset_tag=subset_tag,
                force_regenerate=False)
            down_sampled_dict.update({subset_tag: down_sample_dataset(hvo_seq_set, down_sampled_ratio)})

        # collect and dump samples that match filter
        if cache_down_sampled_set:
            # create directories if needed
            if not os.path.exists(dir__):
                os.makedirs(dir__)
            for set_key_, set_data_ in down_sampled_dict.items():
                ofile = bz2.BZ2File(os.path.join(dir__, f"{set_key_}.bz2pickle"), 'wb')
                pickle.dump(set_data_, ofile)
                ofile.close()

        dataLoaderLogger.info(f"Loaded {len(down_sampled_dict[subset_tag])} {subset_tag} samples from {dir__}")
        return down_sampled_dict[subset_tag]
    else:
        dataLoaderLogger.info(f"Loading Cached Version from: {dir__}")
        ifile = bz2.BZ2File(os.path.join(dir__, f"{subset_tag}.bz2pickle"), 'rb')
        set_data_ = pickle.load(ifile)
        ifile.close()
        dataLoaderLogger.info(f"Loaded {len(set_data_)} {subset_tag} samples from {dir__}")
        return set_data_


if __name__ == "__main__":
    # tester
    dataLoaderLogger.info("Run demos/data/demo.py to test")
