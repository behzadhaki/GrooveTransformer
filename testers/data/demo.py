from data.dataLoaders import load_original_gmd_dataset_pickle

# Load 2bar gmd dataset as a dictionary
gmd_pickle_path = "data/gmd/resources/storedDicts/groove_2bar-midionly.bz2pickle"
gmd_dict = load_original_gmd_dataset_pickle(gmd_pickle_path)

gmd_dict.keys()          # dict_keys(['train', 'test', 'validation'])
gmd_dict['train'].keys()        #dict_keys(['drummer', 'session', 'loop_id', 'master_id', 'style_primary', 'style_secondary', 'bpm', 'beat_type', 'time_signature', 'full_midi_filename', 'full_audio_filename', 'midi', 'note_sequence'])


# Extract HVO_Sequences from the dictionaries

from data.dataLoaders import extract_hvo_sequences_dict, get_drum_mapping_using_label

hvo_dict = extract_hvo_sequences_dict (gmd_dict, [4], get_drum_mapping_using_label("ROLAND_REDUCED_MAPPING"))


# Load GMD Dataset in `HVO_Sequence` format using a single command

from data.dataLoaders import load_gmd_hvo_sequences

gmd_pickle_path = "data/gmd/resources/storedDicts/groove_2bar-midionly.bz2pickle"
dataset_tag = "gmd"
filter_json_path = "filter.json"
beat_division_factor = [4]
drum_mapping_label = "ROLAND_REDUCED_MAPPING"
subset_tag = "train"
force_regenerate = False        # set true if you don't want to use the cached version

train_set = load_gmd_hvo_sequences(
    gmd_pickle_path, dataset_tag, filter_json_path, beat_division_factor, drum_mapping_label,
    subset_tag, force_regenerate)