import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from itertools import combinations
from tqdm import tqdm
from bokeh.layouts import gridplot
from data.src.utils import get_bokeh_histogram
import note_seq as ns

if __name__ == '__main__':
    # ==================================================================================================================
    # Load Data
    # ==================================================================================================================
    from data import load_original_gmd_dataset_pickle

    for nbars in [2, 4]:

        assert nbars in [2, 4], "nbars must be either 2 or 4"

        # Load gmd dataset as a dictionary
        gmd_dict = load_original_gmd_dataset_pickle(
            gmd_pickle_path=f"data/gmd/resources/storedDicts/groove_{nbars}bar-midionly.bz2pickle")

        for subset in ["train", "validation", "test"]:
            data_dict = gmd_dict[subset]

            # compile all the samples into a single list of tuples
            n_samples = len(data_dict["master_id"])

            for i in tqdm(range(n_samples), desc=f"compiling all {subset} samples into a single list of tuples"):
                # get the midi
                note_sequence = data_dict["note_sequence"][i]
                count = data_dict['loop_id'][i].split(":")[-1]
                bpm = data_dict['bpm'][i]
                beat_type = data_dict['beat_type'][i]
                style_primary = data_dict['style_primary'][i]
                style_secondary = data_dict['style_secondary'][i]
                time_signature = data_dict['time_signature'][i]
                fname = f"data/gmd/resources/source_dataset/{nbars}_bar_version/{data_dict['master_id'][i]}/{count}_{time_signature}_{beat_type}_{bpm}_{style_primary}_{style_secondary}_{subset}.mid"
                os.makedirs(os.path.dirname(fname), exist_ok=True)
                pm = ns.note_sequence_to_pretty_midi(note_sequence)
                pm.write(fname)

        # zip the folder
        import shutil
        shutil.make_archive(f"data/gmd/resources/source_dataset/groove-{nbars}_bar_version-midionly", 'zip',
                            f"data/gmd/resources/source_dataset/{nbars}_bar_version")