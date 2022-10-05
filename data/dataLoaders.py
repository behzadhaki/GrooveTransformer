from data.src.utils import *

def load_gmd_hvo_sequences(gmd_pickle_path, dataset_tag, filter_json_path, beat_division_factor, drum_mapping_label,
                           subset_tag, force_regenerate=False):
    assert os.path.exists(gmd_pickle_path), "path to gmd dict pickle is incorrect --- " \
                                            "look into data/gmd/resources/storedDicts/groove-*.bz2pickle"

    dir__ = get_data_directory_using_filters(dataset_tag, filter_json_path)

    if (not os.path.exists(dir__)) or force_regenerate is True:
        print("No Cached Version Available --> extracting data from original groove midi data")
        gmd_dict = load_original_gmd_dataset_pickle(gmd_pickle_path)
        drum_mapping = get_drum_mapping_using_label(drum_mapping_label)
        hvo_dict = extract_hvo_sequences_dict(gmd_dict, beat_division_factor, drum_mapping)
        pickle_hvo_dict(hvo_dict, dataset_tag, filter_json_path)
    else:
        print(f"Using cached data available at {dir__}")

    print(os.path.join(dir__, f"{subset_tag}.pickle"))

    ifile = bz2.BZ2File(os.path.join(dir__, f"{subset_tag}.bz2pickle"), 'rb')
    data = pickle.load(ifile)
    ifile.close()

    return data


if __name__ == "__main__":

    gmd_pickle_path = "data/gmd/resources/storedDicts/groove_2bar-midionly.bz2pickle"
    dataset_tag = "gmd"
    filter_json_path = "filter.json"
    beat_division_factor = [4]
    drum_mapping_label = "ROLAND_REDUCED_MAPPING"
    subset_tag = "train"

    # gmd_dict = load_original_gmd_dataset_pickle(gmd_pickle_path)
    # hvo_dict = extract_hvo_sequences_dict (gmd_dict, [4], get_drum_mapping_using_label("ROLAND_REDUCED_MAPPING"))
    # pickle_hvo_dict(hvo_dict, dataset_tag, filter_json_path)

    train_set = load_gmd_hvo_sequences(
        gmd_pickle_path, dataset_tag, filter_json_path, beat_division_factor, drum_mapping_label,
        subset_tag)