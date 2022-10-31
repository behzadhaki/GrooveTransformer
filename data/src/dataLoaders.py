from data.src.utils import *
import numpy as np
import torch
from torch.utils.data import Dataset

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
        print("loading dataset: ", dataset_tag)
        raw_data_pickle_path = dataset_setting_json["raw_data_pickle_path"][dataset_tag]
        assert os.path.exists(raw_data_pickle_path), "path to gmd dict pickle is incorrect --- " \
                                                "look into data/gmd/resources/storedDicts/groove-*.bz2pickle"
        dir__ = get_data_directory_using_filters(dataset_tag, dataset_setting_json_path)
        print("dir__: ", dir__)
        beat_division_factor = dataset_setting_json["global"]["beat_division_factor"]
        drum_mapping_label = dataset_setting_json["global"]["drum_mapping_label"]
        print(drum_mapping_label)
        if (not os.path.exists(dir__)) or force_regenerate is True:
            print(f"No Cached Version Available Here: {dir__}")
            print("|----> extracting data from original groove midi data")
            gmd_dict = load_original_gmd_dataset_pickle(raw_data_pickle_path)
            drum_mapping = get_drum_mapping_using_label(drum_mapping_label)
            hvo_dict = extract_hvo_sequences_dict(gmd_dict, beat_division_factor, drum_mapping)
            pickle_hvo_dict(hvo_dict, dataset_tag, dataset_setting_json_path)
        else:
            print(f"Using cached data available at {dir__}")

        print(os.path.join(dir__, f"{subset_tag}.pickle"))

        ifile = bz2.BZ2File(os.path.join(dir__, f"{subset_tag}.bz2pickle"), 'rb')
        data = pickle.load(ifile)
        ifile.close()

    return data


class MonotonicGrooveDataset(Dataset):
    def __init__(self, dataset_setting_json_path, subset_tag, max_len, tapped_voice_idx=2, collapse_tapped_sequence=False):
        """

        :param dataset_setting_json_path:   path to the json file containing the dataset settings (see data/dataset_json_settings/4_4_Beats_gmd.json)
        :param subset_tag:                [str] whether to load the train/test/validation set
        :param max_len:              [int] maximum length of the sequences to be loaded
        :param tapped_voice_idx:    [int] index of the voice to be tapped (default is 2 which is usually closed hat)
        :param collapse_tapped_sequence:  [bool] returns a Tx3 tensor instead of a Tx(3xNumVoices) tensor
        """

        # Get processed inputs, outputs and hvo sequences
        self.inputs = list()
        self.outputs = list()
        self.hvo_sequences = list()
        self.info = {
            "style_primary": list(),
            "master_id": list(),
            "bpm": list(),
        }

        subset = load_gmd_hvo_sequences(dataset_setting_json_path, subset_tag, force_regenerate=False)

        # collect input tensors, output tensors, and hvo_sequences
        for idx, hvo_seq in enumerate(tqdm(subset)):
            all_zeros = not np.any(hvo_seq.hvo.flatten())
            if not all_zeros:
                # Ensure all have a length of max_len
                pad_count = max(max_len - hvo_seq.hvo.shape[0], 0)
                hvo_seq.hvo = np.pad(hvo_seq.hvo, ((0, pad_count), (0, 0)), "constant")
                hvo_seq.hvo = hvo_seq.hvo[:max_len, :]  # in case seq exceeds max len
                self.hvo_sequences.append(hvo_seq)

                flat_seq = hvo_seq.flatten_voices(voice_idx=tapped_voice_idx, reduce_dim=collapse_tapped_sequence)
                self.inputs.append(flat_seq)
                self.outputs.append(hvo_seq.hvo)

        # find voice and genre distributions


        # wandb.config.update({"set_length": len(self.sequences)})
        print(f"{subset_tag} Dataset loaded\n")

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

    dataset_setting_json_path = "data/dataset_json_settings/4_4_Beats_gmd.json"
    beat_division_factor = [4]
    drum_mapping_label = "ROLAND_REDUCED_MAPPING"
    subset_tag = "train"

    gmd_dict = load_original_gmd_dataset_pickle(gmd_pickle_path = "data/gmd/resource/storedDicts/groove_2bar_midionly.bz2pickle")
    # hvo_dict = extract_hvo_sequences_dict (gmd_dict, [4], get_drum_mapping_using_label("ROLAND_REDUCED_MAPPING"))
    # pickle_hvo_dict(hvo_dict, dataset_tag, dataset_setting_json_path)

    train_set = load_gmd_hvo_sequences(dataset_setting_json_path, subset_tag)


    # load dataset as torch.utils.data.Dataset
    training_dataset = MonotonicGrooveDataset(
        dataset_setting_json_path="data/dataset_json_settings/4_4_Beats_gmd.json",
        subset_tag="train",
        max_len=32,
        tapped_voice_idx=2,
        collapse_tapped_sequence=False)