from torch.utils.data import Dataset, DataLoader
from copy import deepcopy
import pandas as pd
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
import seaborn as sns
from bokeh.io import output_file, show
from bokeh.layouts import layout, grid

import sys
sys.path.insert(1, "../../")
sys.path.insert(1, "../")

from preprocessed_dataset.Subset_Creators import subsetters

from GrooveEvaluator.plotting_utils import multi_feature_plotter, multi_voice_velocity_profile_plotter

import warnings
warnings.filterwarnings("ignore")

from GrooveEvaluator.feature_extractor import Feature_Extractor_From_HVO_Set, Interset_Distance_Calculator, Intraset_Distance_Calculator, Distance_to_PDF_Converter, convert_distances_dict_to_pdf_histograms_dict

filters = {
    "drummer": None,  # ["drummer1", ..., and/or "session9"]
    "session": None,  # ["session1", "session2", and/or "session3"]
    "loop_id": None,
    "master_id": None,
    "style_primary": None,  # ["afrobeat", "afrocuban", "blues", "country", "dance", "funk", "gospel", "highlife", "hiphop", "jazz","latin", "middleeastern", "neworleans", "pop", "punk", "reggae", "rock", "soul"]
    # "style_secondary" None,       # [fast, brazilian_baiao, funk, halftime, purdieshuffle, None, samba, chacarera, bomba, brazilian, brazilian_sambareggae, brazilian_samba, venezuelan_joropo, brazilian_frevo, fusion, motownsoft]
    "bpm": None,  # [(range_0_lower_bound, range_0_upper_bound), ..., (range_n_lower_bound, range_n_upper_bound)]
    "beat_type": ["beat"],  # ["beat" or "fill"]
    "time_signature": ["4-4"],  # ["4-4", "3-4", "6-8"]
    "full_midi_filename": None,  # list_of full_midi_filenames
    "full_audio_filename": None  # list_of full_audio_filename
}

Styles_complete = ["afrobeat", "afrocuban", "blues", "country", "dance", "funk", "gospel", "highlife", "hiphop", "jazz",
                   "latin", "middleeastern", "neworleans", "pop", "punk", "reggae", "rock", "soul"]


def check_if_passes_filters(df_row, filters):
    meets_filter = []
    for filter_key, filter_values in zip(filters.keys(), filters.values()):
        if filters[filter_key] is not None:
            if df_row.at[filter_key] in filter_values:
                meets_filter.append(True)
            else:
                meets_filter.append(False)
    return all(meets_filter)


class GrooveMidiDataset(Dataset):
    def __init__(
            self,
            source_path="../preprocessed_dataset/datasets_extracted_locally/GrooveMidi/"
                        "hvo_0.3.0/Processed_On_13_05_2021_at_12_56_hrs",
            subset="GrooveMIDI_processed_test",
            metadata_csv_filename="metadata.csv",
            hvo_pickle_filename="hvo_sequence_data.obj",
            filters=filters,
            max_len=32
    ):

        train_file = open(os.path.join(source_path, subset, hvo_pickle_filename), 'rb')
        train_set = pickle.load(train_file)
        metadata = pd.read_csv(os.path.join(source_path, subset, metadata_csv_filename))

        self.hvo_sequences = []

        for ix, hvo_seq in enumerate(train_set):
            if len(hvo_seq.time_signatures) == 1:       # ignore if time_signature change happens
                all_zeros = not np.any(hvo_seq.hvo.flatten())
                if not all_zeros:  # Ignore silent patterns
                    if check_if_passes_filters(metadata.loc[ix], filters):
                        # add metadata to hvo_seq scores
                        hvo_seq.drummer = metadata.loc[ix].at["drummer"]
                        hvo_seq.session = metadata.loc[ix].at["session"]
                        hvo_seq.master_id = metadata.loc[ix].at["master_id"]
                        hvo_seq.style_primary = metadata.loc[ix].at["style_primary"]
                        hvo_seq.style_secondary = metadata.loc[ix].at["style_secondary"]
                        hvo_seq.beat_type = metadata.loc[ix].at["beat_type"]
                        # pad with zeros to match max_len
                        pad_count = max(max_len - hvo_seq.hvo.shape[0], 0)
                        hvo_seq.hvo = np.pad(hvo_seq.hvo, ((0, pad_count), (0, 0)), 'constant')
                        hvo_seq.hvo = hvo_seq.hvo [:max_len, :]         # In case, sequence exceeds max_len
                        self.hvo_sequences.append(hvo_seq)

    def __len__(self):
        return len(self.hvo_sequences)

    def __getitem__(self, idx):
        return self.hvo_sequences[idx].hvo, idx

feature_extractors_list_test = []

set_ = "test"
gmds = []
for style in Styles_complete:
    # Create a filter for style
    filters_for_style = deepcopy(filters)
    filters_for_style["style_primary"] = [style]
    # Load style subset of the GrooveMIDI set
    # torch_set = GrooveMidiDataset(filters=filters_for_style, subset="GrooveMIDI_processed_{}".format(set_))
    gmd = GrooveMidiDataset(filters=filters_for_style, subset="GrooveMIDI_processed_{}".format(set_))
    gmds.append(gmd)
    if len(gmd.hvo_sequences) > 0:
        feature_extractors_list_test.append(Feature_Extractor_From_HVO_Set(
            gmd.hvo_sequences, name="{}".format(style))
        )


if __name__ == '__main__':
    output_file("{} Set - All samples - Vel Profiles.html".format(set_))
    p = multi_voice_velocity_profile_plotter(feature_extractors_list=feature_extractors_list_test,
                                             title_prefix = "{}_set".format(set_),
                                             legend_fnt_size="18px",
                                             plot_height_per_set=100
    )
    show(grid(p, ncols=3))

    # todo implement random sampling n_examples from set

    output_file("{} Set - All samples - Features.html".format(set_))
    p = multi_feature_plotter(feature_extractors_list=feature_extractors_list_test,
                              title_prefix="{}_set".format(set_),
                              normalize_data=False,
                              analyze_combined_sets=True,
                              force_extract=False,
                              plot_width=650, plot_height=600,
                              legend_fnt_size="8px", scale_y=False, resolution=200)

    show(grid(p, ncols=3))

    output_file("{} Set - All samples - Features - Normalized.html".format(set_))
    p = multi_feature_plotter(feature_extractors_list=feature_extractors_list_test,
                              title_prefix="{}_set".format(set_),
                              normalize_data=True,
                              analyze_combined_sets=True,
                              force_extract=False,
                              plot_width=650, plot_height=600,
                              legend_fnt_size="8px", scale_y=False, resolution=200)

    show(grid(p, ncols=3))