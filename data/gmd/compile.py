from src.utils import *

import argparse

parser = argparse.ArgumentParser(description='Script to download groove midi dataset from tfds and compile into'
                                             'a single dictionary and stored as a pickle')

parser.add_argument('--set', type=str,
                    default="resource/compiled_in_dict/",
                    help='path to store the pickled dictionary of dataset',
                    required=False)

parser.add_argument('--save_at', type=str,
                    default="resource/compiled_in_dict/",
                    help='path to store the pickled dictionary of dataset',
                    required=False)

parser.add_argument('--SetID', type=int,
                    default=0,
                    help='SetID 0 is 2bar set, SetID 1 is 4bar set, SetID 2 is Full (unsplit) set',
                    required=False)

args = parser.parse_args()

if __name__ == "__main__":

    setTags = ["groove/2bar-midionly", "groove/4bar-midionly", "groove/full-midionly"]

    # Step 1 --- get subsets from tfds
    train_tfds, test_tfds, validation_tfds = load_midionly_gmd_subsets_from_tfds(setTags[args.SetID])
    metadata_pd = get_gmd_metadata_as_pd("resources/source_dataset/info.csv")

    # Step 2 --- mix tfds sets with metadata into a dictionary
    train_dict = get_gmd_dict(train_tfds, metadata_pd)
    test_dict = get_gmd_dict(test_tfds, metadata_pd)
    validation_dict = get_gmd_dict(validation_tfds, metadata_pd)

    # Step 3 --- sort based on loop_id
    train_dict = sort_dictionary_by_key(train_dict, "loop_id")
    test_dict = sort_dictionary_by_key(test_dict, "loop_id")
    validation_dict = sort_dictionary_by_key(validation_dict, "loop_id")
    
    # Step 4 --- store as a single dict
    gmd_dict = {"train": train_dict, "test": test_dict, "validation": validation_dict}
    fname = setTags[args.SetID].replace("/", "_")
    pickle_dict = pickle_dict(gmd_dict, "resources/storedDicts", fname)


