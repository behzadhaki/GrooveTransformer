from data.src.utils import *
import numpy as np
import torch
from torch.utils.data import Dataset
import logging
logging.basicConfig(level=logging.DEBUG)
dataLoaderLogger = logging.getLogger("data.src.dataLoaders")


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
                 collapse_tapped_sequence=False, load_as_tensor=True, sort_by_metada_key=None):
        """

        :param dataset_setting_json_path:   path to the json file containing the dataset settings (see data/dataset_json_settings/4_4_Beats_gmd.json)
        :param subset_tag:                [str] whether to load the train/test/validation set
        :param max_len:              [int] maximum length of the sequences to be loaded
        :param tapped_voice_idx:    [int] index of the voice to be tapped (default is 2 which is usually closed hat)
        :param collapse_tapped_sequence:  [bool] returns a Tx3 tensor instead of a Tx(3xNumVoices) tensor
        :param load_as_tensor:      [bool] loads the data as a tensor of torch.float32 instead of a numpy array
        """

        # Get processed inputs, outputs and hvo sequences
        self.inputs = list()
        self.outputs = list()
        self.hvo_sequences = list()

        subset = load_gmd_hvo_sequences(dataset_setting_json_path, subset_tag, force_regenerate=False)

        if sort_by_metada_key:
            if sort_by_metada_key in subset[0].metadata[sort_by_metada_key]:
                subset = sorted(subset, key=lambda x: x.metadata[sort_by_metada_key])

        # collect input tensors, output tensors, and hvo_sequences
        for idx, hvo_seq in enumerate(tqdm(subset)):
            if hvo_seq.hits is not None:
                hvo_seq.adjust_length(max_len)
                if np.any(hvo_seq.hits):
                    # Ensure all have a length of max_len
                    self.hvo_sequences.append(hvo_seq)
                    flat_seq = hvo_seq.flatten_voices(voice_idx=tapped_voice_idx, reduce_dim=collapse_tapped_sequence)
                    self.inputs.append(flat_seq)
                    self.outputs.append(hvo_seq.hvo)

        if load_as_tensor:
            self.inputs = torch.tensor(np.array(self.inputs), dtype=torch.float32)
            self.outputs = torch.tensor(np.array(self.outputs), dtype=torch.float32)

        dataLoaderLogger.info(f"Loaded {len(self.inputs)} sequences")

    def __len__(self):
        return len(self.hvo_sequences)

    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx], idx

    def get_hvo_sequences_at(self, idx):
        return self.hvo_sequences[idx]

    def get_inputs_at(self, idx):
        return self.inputs[idx]

    def get_outputs_at(self, idx):
        return self.outputs[idx]


if __name__ == "__main__":
    # tester
    dataLoaderLogger.info("Run testers/data/demo.py to test")
