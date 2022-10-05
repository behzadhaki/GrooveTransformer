import os, sys
import json
import pickle
import bz2
from tqdm import tqdm

from hvo_sequence.io_helpers import note_sequence_to_hvo_sequence
from hvo_sequence.drum_mappings import get_drum_mapping_using_label

def does_pass_filter(hvo_sample, filter_dict):   # FIXME THERE IS AN ISSUE HERE
    passed_conditions = [True]  # initialized with true in case no filters are required
    for fkey_, fval_ in filter_dict.items():
        if fval_ is not None:
            passed_conditions.append(True if hvo_sample.metadata[fkey_] in fval_ else False)

    return all(passed_conditions)


def get_data_directory_using_filters(dataset_tag, filter_json_path):
    """
    returns the directory path from which the data corresponding to the
    specified filter.json file is located or should be stored

    :param dataset_tag: [str] (use "gmd" for groove midi dataset)
    :param filter_json_path: [file.json path] (path to filter.json or similar filter jsons)
    :return: path to save/load from the train.pickle/test.pickle/validation.pickle hvo_sequence subsets
    """
    main_path = f"data/{dataset_tag}/resources/cached/"
    last_directory = ""
    filter_dict = json.load(open(filter_json_path, "r"))[dataset_tag]
    global_dict = json.load(open(filter_json_path, "r"))["global"]
    for key_, val_ in global_dict.items():
        main_path += f"{key_}_{val_}/"

    for key_, val_ in filter_dict.items():
        if val_ is not None:
            print(key_, " ", val_)
            last_directory += f"{key_}_{val_}_"

    return main_path + last_directory[:-1]  # remove last underline


def pickle_hvo_dict(hvo_dict, dataset_tag, filter_json_path):
    """ pickles a dictionary of train/test/validation hvo_sequences (see below for dict structure)

    :param hvo_dict: [dict] dict of format {"train": [hvo_seqs], "test": [hvo_seqs], "validation": [hvo_seqs]}
    :param dataset_tag: [str] (use "gmd" for groove midi dataset)
    :param filter_json_path: [file.json path] (path to filter.json or similar filter jsons)
    """


    # create directories if needed
    dir__ = get_data_directory_using_filters(dataset_tag, filter_json_path)
    if not os.path.exists(dir__):
        os.makedirs(dir__)

    # collect and dump samples that match filter
    filter_dict_ = json.load(open(filter_json_path, "r"))[dataset_tag]
    for set_key_, set_data_ in hvo_dict.items():
        filtered_samples = []
        num_samples = len(set_data_)
        for sample in tqdm(set_data_, total = num_samples, desc=f"filtering HVO_Sequences in subset {set_key_}"):
            if does_pass_filter(sample, filter_dict_):
                filtered_samples.append(sample)

        ofile = bz2.BZ2File(os.path.join(dir__, f"{set_key_}.bz2pickle"), 'wb')
        pickle.dump(filtered_samples, ofile)
        ofile.close()



def load_original_gmd_dataset_pickle(gmd_pickle_path):
    ifile = bz2.BZ2File(open(gmd_pickle_path, "rb"), 'rb')
    gmd_dict = pickle.load(ifile)
    ifile.close()
    return gmd_dict


def extract_hvo_sequences_dict(gmd_dict, beat_division_factor, drum_mapping):
    """

    :param gmd_dict:
    :param beat_division_factor:
    :param use_cached:
    :return:
    """
    gmd_hvo_seq_dict = dict()

    for set in gmd_dict.keys():             # train, test, validation
        hvo_seqs = []
        n_samples = len(gmd_dict[set]["note_sequence"])
        for ix in tqdm(range(n_samples), desc=f"converting to hvo_sequence --> {set} subset"):
            # convert to note_sequence
            note_sequence = gmd_dict[set]["note_sequence"][ix]
            _hvo_seq = note_sequence_to_hvo_sequence(
                ns=note_sequence,
                drum_mapping=drum_mapping,
                beat_division_factors=beat_division_factor
            )
            if len(_hvo_seq.time_signatures) == 1 and len(_hvo_seq.tempos) == 1:
                # get metadata
                for key_ in gmd_dict[set].keys():
                    if key_ not in ["midi", "note_sequence", "hvo_sequence"]:
                        _hvo_seq.metadata.update({key_: str(gmd_dict[set][key_][ix])})

                hvo_seqs.append(_hvo_seq)

        # add hvo_sequences to dictionary
        gmd_hvo_seq_dict.update({set: hvo_seqs})

    return gmd_hvo_seq_dict



if __name__ == "__main__":

    gmd_pickle_path = "gmd/resources/storedDicts/groove_2bar-midionly.bz2pickle"
    dataset_tag = "gmd"
    filter_json_path = "filter.json"
    beat_division_factor = [4]
    drum_mapping_label = "ROLAND_REDUCED_MAPPING"
    subset_tag = "train"

    gmd_dict = load_original_gmd_dataset_pickle(gmd_pickle_path)
    hvo_dict = extract_hvo_sequences_dict (gmd_dict, [4], get_drum_mapping_using_label("ROLAND_REDUCED_MAPPING"))
    pickle_hvo_dict(hvo_dict, dataset_tag, filter_json_path)


