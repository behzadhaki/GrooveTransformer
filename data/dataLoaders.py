from data.src.utils import *

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

if __name__ == "__main__":

    dataset_setting_json_path = "data/dataset_json_settings/4_4_Beats_gmd.json"
    beat_division_factor = [4]
    drum_mapping_label = "ROLAND_REDUCED_MAPPING"
    subset_tag = "train"

    # gmd_dict = load_original_gmd_dataset_pickle(gmd_pickle_path = "data/gmd/resource/storedDicts/groove_2bar_midionly.bz2pickle")
    # hvo_dict = extract_hvo_sequences_dict (gmd_dict, [4], get_drum_mapping_using_label("ROLAND_REDUCED_MAPPING"))
    # pickle_hvo_dict(hvo_dict, dataset_tag, dataset_setting_json_path)

    train_set = load_gmd_hvo_sequences(dataset_setting_json_path, subset_tag)