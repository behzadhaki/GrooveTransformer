
from data import load_original_gmd_dataset_pickle
import os


os.chdir("../../")


# Load 2bar gmd dataset as a dictionary
gmd_dict = load_original_gmd_dataset_pickle(
    gmd_pickle_path="../../data/gmd/resources/storedDicts/groove_2bar-midionly.bz2pickle")

gmd_dict.keys()          # dict_keys(['train', 'test', 'validation'])
gmd_dict['train'].keys()        #dict_keys(['drummer', 'session', 'loop_id', 'master_id', 'style_primary', 'style_secondary', 'bpm', 'beat_type', 'time_signature', 'full_midi_filename', 'full_audio_filename', 'midi', 'note_sequence'])


from data import extract_hvo_sequences_dict, get_drum_mapping_using_label

hvo_dict = extract_hvo_sequences_dict (
    gmd_dict=gmd_dict,
    beat_division_factor=[96],
    drum_mapping=get_drum_mapping_using_label("ROLAND_REDUCED_MAPPING"))


"""
1. Load pre-token dataset (GMD at 96)
2. Tokenize
3. Convert to Dataloader
4. Print out some examples, try enumerating
"""
pretokenized_dataset = MonotonicGrooveDataset(
    dataset_setting_json_path="data/dataset_json_settings/BeatsAndFills_gmd.json",
    subset_tag="train",
    max_len=500, #what is a good value for this?
    tapped_voice_idx=2,
    collapse_tapped_sequence=False,
    load_as_tensor=False,
    sort_by_metadata_key="loop_id",
    genre_loss_balancing_beta=0.5,
    voice_loss_balancing_beta=0.5
)