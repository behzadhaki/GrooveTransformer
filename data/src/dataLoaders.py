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
        for path_prepend in ["./", "../", "../../"]:
            if os.path.exists(path_prepend + raw_data_pickle_path):
                raw_data_pickle_path = path_prepend + raw_data_pickle_path
                break

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
                 down_sampled_ratio=None, move_all_to_gpu=False,
                 hit_loss_balancing_beta=0, genre_loss_balancing_beta=0):
        """

        :param dataset_setting_json_path:   path to the json file containing the dataset settings (see data/dataset_json_settings/4_4_Beats_gmd.json)
        :param subset_tag:                [str] whether to load the train/test/validation set
        :param max_len:              [int] maximum length of the sequences to be loaded
        :param tapped_voice_idx:    [int] index of the voice to be tapped (default is 2 which is usually closed hat)
        :param collapse_tapped_sequence:  [bool] returns a Tx3 tensor instead of a Tx(3xNumVoices) tensor
        :param load_as_tensor:      [bool] loads the data as a tensor of torch.float32 instead of a numpy array
        :param sort_by_metadata_key: [str] sorts the data by the metadata key provided (e.g. "tempo")
        :param down_sampled_ratio: [float] down samples the data by the ratio provided (e.g. 0.5)
        :param move_all_to_gpu: [bool] moves all the data to the gpu
        :param hit_loss_balancing_beta: [float] beta parameter for hit balancing
                            (if 0 or very small, no hit balancing weights are returned)
        :param genre_loss_balancing_beta: [float] beta parameter for genre balancing
                            (if 0 or very small, no genre balancing weights are returned)
                hit_loss_balancing_beta and genre_balancing_beta are used to balance the data
                according to the hit and genre distributions of the dataset
                (reference: https://arxiv.org/pdf/1901.05555.pdf)
        """

        # Get processed inputs, outputs and hvo sequences
        self.inputs = list()
        self.outputs = list()
        self.hvo_sequences = list()

        # load pre-stored hvo_sequences or
        #   a portion of them uniformly sampled if down_sampled_ratio is provided
        # ------------------------------------------------------------------------------------------
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

        # Sort data by a given metadata key if provided (e.g. "style_primary")
        # ------------------------------------------------------------------------------------------
        if sort_by_metadata_key:
            if sort_by_metadata_key in subset[0].metadata[sort_by_metadata_key]:
                subset = sorted(subset, key=lambda x: x.metadata[sort_by_metadata_key])

        # collect input tensors, output tensors, and hvo_sequences
        # ------------------------------------------------------------------------------------------
        for idx, hvo_seq in enumerate(tqdm(subset)):
            if hvo_seq.hits is not None:
                hvo_seq.adjust_length(max_len)
                if np.any(hvo_seq.hits):
                    # Ensure all have a length of max_len
                    self.hvo_sequences.append(hvo_seq)
                    self.outputs.append(hvo_seq.hvo)
                    flat_seq = hvo_seq.flatten_voices(voice_idx=tapped_voice_idx, reduce_dim=collapse_tapped_sequence)
                    self.inputs.append(flat_seq)

        self.inputs = np.array(self.inputs)
        self.outputs = np.array(self.outputs)

        # Get hit balancing weights if a beta parameter is provided
        # ------------------------------------------------------------------------------------------
        # get the effective number of hits per step and voice
        hits = self.outputs[:, :, :self.outputs.shape[-1] // 3]
        total_hits = hits.sum(0) + 1e-6
        effective_num_hits = 1.0 - np.power(hit_loss_balancing_beta, total_hits)
        hit_balancing_weights = (1.0 - hit_loss_balancing_beta) / effective_num_hits
        # normalize
        num_classes = hit_balancing_weights.shape[0] * hit_balancing_weights.shape[1]
        hit_balancing_weights = hit_balancing_weights / hit_balancing_weights.sum() * num_classes
        self.hit_balancing_weights_per_sample = [hit_balancing_weights for _ in range(len(self.outputs))]

        # Get genre balancing weights if a beta parameter is provided
        # ------------------------------------------------------------------------------------------
        # get the effective number of genres
        genres_per_sample = [sample.metadata["style_primary"] for sample in self.hvo_sequences]
        genre_counts = {genre: genres_per_sample.count(genre) for genre in set(genres_per_sample)}
        effective_num_genres = 1.0 - np.power(genre_loss_balancing_beta, list(genre_counts.values()))
        genre_balancing_weights = (1.0 - genre_loss_balancing_beta) / effective_num_genres
        # normalize
        genre_balancing_weights = genre_balancing_weights / genre_balancing_weights.sum() * len(genre_counts)
        genre_balancing_weights = {genre: weight for genre, weight in
                                   zip(genre_counts.keys(), genre_balancing_weights)}
        t_steps = self.outputs.shape[1]
        n_voices = self.outputs.shape[2] // 3
        temp_row = np.ones((t_steps, n_voices))
        self.genre_balancing_weights_per_sample = np.array(
            [temp_row * genre_balancing_weights[sample.metadata["style_primary"]]
             for sample in self.hvo_sequences])

        # Load as tensor if requested
        # ------------------------------------------------------------------------------------------
        if load_as_tensor or move_all_to_gpu:
            self.inputs = torch.tensor(self.inputs, dtype=torch.float32)
            self.outputs = torch.tensor(self.outputs, dtype=torch.float32)
            if hit_loss_balancing_beta is not None:
                self.hit_balancing_weights_per_sample = torch.tensor(self.hit_balancing_weights_per_sample,
                                                                     dtype=torch.float32)
            if genre_loss_balancing_beta is not None:
                self.genre_balancing_weights_per_sample = torch.tensor(self.genre_balancing_weights_per_sample,
                                                                       dtype=torch.float32)

        # Move to GPU if requested and GPU is available
        # ------------------------------------------------------------------------------------------
        if move_all_to_gpu and torch.cuda.is_available():
            self.inputs = self.inputs.to('cuda')
            self.outputs = self.outputs.to('cuda')
            if hit_loss_balancing_beta is not None:
                self.hit_balancing_weights_per_sample = self.hit_balancing_weights_per_sample.to('cuda')
            if genre_loss_balancing_beta is not None:
                self.genre_balancing_weights_per_sample = self.genre_balancing_weights_per_sample.to('cuda')

        dataLoaderLogger.info(f"Loaded {len(self.inputs)} sequences")

    def __len__(self):
        return len(self.hvo_sequences)

    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx], \
               self.hit_balancing_weights_per_sample[idx], self.genre_balancing_weights_per_sample[idx], idx

    def get_hvo_sequences_at(self, idx):
        return self.hvo_sequences[idx]

    def get_inputs_at(self, idx):
        return self.inputs[idx]

    def get_outputs_at(self, idx):
        return self.outputs[idx]

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
