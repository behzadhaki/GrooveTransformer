from data.dataLoaders import load_original_gmd_dataset_pickle

# Load 2bar gmd dataset as a dictionary
gmd_dict = load_original_gmd_dataset_pickle(
    gmd_pickle_path="data/gmd/resources/storedDicts/groove_2bar-midionly.bz2pickle")

gmd_dict.keys()          # dict_keys(['train', 'test', 'validation'])
gmd_dict['train'].keys()        #dict_keys(['drummer', 'session', 'loop_id', 'master_id', 'style_primary', 'style_secondary', 'bpm', 'beat_type', 'time_signature', 'full_midi_filename', 'full_audio_filename', 'midi', 'note_sequence'])


# Extract HVO_Sequences from the dictionaries

from data.dataLoaders import extract_hvo_sequences_dict, get_drum_mapping_using_label

hvo_dict = extract_hvo_sequences_dict (
    gmd_dict=gmd_dict,
    beat_division_factor=[4],
    drum_mapping=get_drum_mapping_using_label("ROLAND_REDUCED_MAPPING"))


# Load GMD Dataset in `HVO_Sequence` format using a single command

from data.dataLoaders import load_gmd_hvo_sequences
train_set = load_gmd_hvo_sequences(
    dataset_setting_json_path="data/dataset_json_settings/4_4_Beats_gmd.json",
    subset_tag="train",
    force_regenerate=False)


# =================================================================================================
# Load dataset as torch.utils.data.Dataset
from data.dataLoaders import MonotonicGrooveDataset

# load dataset as torch.utils.data.Dataset
training_dataset = MonotonicGrooveDataset(
    dataset_setting_json_path="data/dataset_json_settings/4_4_Beats_gmd.json",
    subset_tag="train",
    max_len=32,
    tapped_voice_idx=2,
    collapse_tapped_sequence=False)


# use the above dataset in the training pipeline, you need to use torch.utils.data.DataLoader
from torch.utils.data import DataLoader
train_dataloader = DataLoader(training_dataset, batch_size=32, shuffle=True)


epochs = 10
for epoch in range(epochs):
    # in each epoch we iterate over the entire dataset
    for batch_count, (inputs, outputs, indices) in enumerate(train_dataloader):
        print(f"Epoch {epoch} - Batch #{batch_count} - inputs.shape {inputs.shape} - "
              f"outputs.shape {outputs.shape} - indices.shape {indices.shape} ")