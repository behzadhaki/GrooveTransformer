import tensorflow as tf
tf.enable_eager_execution()
# For some reason, tfds import gives an error on the first attempt but works second time around
try:
    import tensorflow_datasets as tfds
except:
    import tensorflow_datasets as tfds

# Import necessary libraries for processing/loading/storing the dataset
import numpy as np
import pickle
import pandas as pd
from shutil import copy2
from tqdm import tqdm

# Import libraries for creating/naming folders/files
import os, sys

# Import magenta's note_seq
import note_seq


# -----------------------------------------------------------------
# ----------------     General Utilities    -----------------------
# -----------------------------------------------------------------
def dict_append(dictionary, key, vals):
    """Appends a value or a list of values to a key in a dictionary"""

    # if the values for a key are not a list, they are converted to a list and then extended with vals
    dictionary[key] = list(dictionary[key]) if not isinstance(dictionary[key], list) else dictionary[key]

    # if vals is a single value (not a list), it's converted to a list so as to be iterable
    vals = [vals] if not isinstance(vals, list) else vals

    # append new values
    for val in vals:
        dictionary[key].append(val)

    return dictionary


def sort_dictionary_by_key (dictionary_to_sort, key_used_to_sort):
    """sorts a dictionary according to the list within a given key"""
    sorted_ids=np.argsort(dictionary_to_sort[key_used_to_sort])
    for key in dictionary_to_sort.keys():
        dictionary_to_sort[key]=[dictionary_to_sort[key][i] for i in sorted_ids]
    return dictionary_to_sort

# -----------------------------------------------------------------
# ----------------     GrooveMIDI Loaders   -----------------------
# -----------------------------------------------------------------


def load_midionly_gmd_subsets_from_tfds(subset_tag):
    """gets "groove/2bar-midionly" subsets from TFDS

    :param subset_tag: str, ( must be "groove/2bar-midionly" or "groove/4bar-midionly" or "groove/full-midionly")
    :return: train, test, validation subsets
    """

    assert subset_tag in ["groove/2bar-midionly", "groove/4bar-midionly", "groove/full-midionly"], \
        "must be groove/2bar-midionly or groove/4bar-midionly or groove/full-midionly"

    train = tfds.load(
        name=subset_tag,
        split=tfds.Split.TRAIN,
        try_gcs=True)

    test = tfds.load(
        name=subset_tag,
        split=tfds.Split.TEST,
        try_gcs=True)

    validation = tfds.load(
        name=subset_tag,
        split=tfds.Split.VALIDATION,
        try_gcs=True)

    return train.batch(1), test.batch(1), validation.batch(1)


def get_gmd_metadata_as_pd(csv_path="resources/source_dataset/info.csv"):
    """loads the info.csv of gmd dataset into a pandas dataframe

    :return: metadata_pd (pandas.DataFrame)"""
    return pd.read_csv(csv_path, delimiter=',')


def get_gmd_dict(dataset, csv_dataframe_info=None):
    """
    mixes tfds dataset and metadata into a dictionary

    :param dataset: DatasetV1Adapter (TFDS loaded dataset)
    :param beat_division_factors: list, default ([4])
    :param csv_dataframe_info: pandas.df (default None)
    :return:
    """

    dataset_dict_processed = dict()
    dataset_dict_processed.update({
        "drummer": [],
        "session": [],
        "loop_id": [],  # the id of the recording from which the loop is extracted
        "master_id": [],  # the id of the recording from which the loop is extracted
        "style_primary": [],
        "style_secondary": [],
        "bpm": [],
        "beat_type": [],
        "time_signature": [],
        "full_midi_filename": [],
        "full_audio_filename": [],
        "midi": [],
        "note_sequence": [],
    })

    dataset_length = [i for i,_ in enumerate(dataset)][-1] + 1

    for features in tqdm(dataset, total=dataset_length):

        # Features to be extracted from the dataset

        note_sequence = note_seq.midi_to_note_sequence(tfds.as_numpy(features["midi"][0]))

        if note_sequence.notes:  # ignore if no notes in note_sequence (i.e. empty 2 bar sequence)


            if (not csv_dataframe_info.empty):

                # Master ID for the Loop
                main_id = features["id"].numpy()[0].decode("utf-8").split(":")[0]

                # Get the relevant series from the dataframe
                df = csv_dataframe_info[csv_dataframe_info.id == main_id]

                # Update the dictionary associated with the metadata
                dict_append(dataset_dict_processed, "drummer", df["drummer"].to_numpy()[0])
                dict_append(dataset_dict_processed, "session", df["session"].to_numpy()[0].split("/")[-1])
                dict_append(dataset_dict_processed, "loop_id", features["id"].numpy()[0].decode("utf-8"))
                dict_append(dataset_dict_processed, "master_id", main_id)

                style_full = df["style"].to_numpy()[0]
                style_primary = style_full.split("/")[0]
                dict_append(dataset_dict_processed, "style_primary", style_primary)
                if "/" in style_full:
                    style_secondary = style_full.split("/")[1]
                    dict_append(dataset_dict_processed, "style_secondary", style_secondary)
                else:
                    dict_append(dataset_dict_processed, "style_secondary", ["None"])

                dict_append(dataset_dict_processed, "bpm", df["bpm"].to_numpy()[0])
                dict_append(dataset_dict_processed, "beat_type", df["beat_type"].to_numpy()[0])
                dict_append(dataset_dict_processed, "time_signature", df["time_signature"].to_numpy()[0])
                dict_append(dataset_dict_processed, "full_midi_filename", df["midi_filename"].to_numpy()[0])
                dict_append(dataset_dict_processed, "full_audio_filename", df["audio_filename"].to_numpy()[0])
                dict_append(dataset_dict_processed, "midi", features["midi"].numpy()[0])
                dict_append(dataset_dict_processed, "note_sequence", [note_sequence])

        else:
            pass

    return dataset_dict_processed


def pickle_dict(gmd_dict, path, filename):

    if not os.path.exists(path):
        os.makedirs(path)

    dataset_filehandler = open(os.path.join(path, filename), "wb")
    pickle.dump(gmd_dict,  dataset_filehandler)



