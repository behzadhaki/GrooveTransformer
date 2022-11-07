from torch.utils.data import Dataset, DataLoader
from copy import deepcopy
import pandas as pd
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
import seaborn as sns

import sys
sys.path.insert(1, "../../hvo_sequence")
sys.path.insert(1, "../hvo_sequence")

from hvo_sequence.hvo_seq import HVO_Sequence

import warnings
warnings.filterwarnings("ignore")

from GrooveEvaluator.feature_extractor import Feature_Extractor_From_HVO_Set, Interset_Distance_Calculator, Intraset_Distance_Calculator, Distance_to_PDF_Converter, convert_distances_dict_to_pdf_histograms_dict

filters = {
    "drummer": None,  # ["drummer1", ..., and/or "session9"]
    "session": None,  # ["session1", "session2", and/or "session3"]
    "loop_id": None,
    "master_id": None,
    "style_primary": None,  # [funk, latin, jazz, rock, gospel, punk, hiphop, pop, soul, neworleans, afrobeat]
    # "style_secondary" None,       # [fast, brazilian_baiao, funk, halftime, purdieshuffle, None, samba, chacarera, bomba, brazilian, brazilian_sambareggae, brazilian_samba, venezuelan_joropo, brazilian_frevo, fusion, motownsoft]
    "bpm": None,  # [(range_0_lower_bound, range_0_upper_bound), ..., (range_n_lower_bound, range_n_upper_bound)]
    "beat_type": ["beat"],  # ["beat" or "fill"]
    "time_signature": ["4-4"],  # ["4-4", "3-4", "6-8"]
    "full_midi_filename": None,  # list_of full_midi_filenames
    "full_audio_filename": None  # list_of full_audio_filename
}

# Styles_complete = [afrobeat, afrocuban, blues, country, dance, funk, gospel, highlife, hiphop, jazz, latin, middleeastern, neworleans, pop, punk, reggae, rock, soul]


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
                        "hvo_0.2.0/Processed_On_09_05_2021_at_23_06_hrs",
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
        return self.hvo_sequences.hvo, idx


# Filters for grabbing subsets of the dataset
filters_rock = deepcopy(filters)
filters_rock["style_primary"] = ["rock"]
filters_funk = deepcopy(filters)
filters_funk["style_primary"] = ["afrobeat"]

# Load Rock and Funk subsets of the GrooveMIDI set
test_set = GrooveMidiDataset(filters=filters_rock)
test_set_funk = GrooveMidiDataset(filters=filters_funk)

# Create two Feature Extractor Instances for Rock and Funk
test_set_feature_extractor = Feature_Extractor_From_HVO_Set(
    test_set.hvo_sequences, feature_list_to_extract=[None], name="test_set_groundT")

funk_set_feature_extractor = Feature_Extractor_From_HVO_Set(
    test_set_funk.hvo_sequences, feature_list_to_extract=[None], name="funk_groundT")

"""# Calculate Interset Distances for Rock
test_set_intraset_distances = Intraset_Distance_Calculator(
    test_set_feature_extractor.extracted_features, name=test_set_feature_extractor.name).intraset_distances_per_feat"""
#funk_intraset_distances = Intraset_Distance_Calculator(
#    funk_set_feature_extractor.extracted_features, name=funk_set_feature_extractor.name).intraset_distances_per_feat
"""

# Calculate Intraset Distances between Rock and Funk
interset_distances = Interset_Distance_Calculator(
    test_set_feature_extractor.extracted_features, funk_set_feature_extractor.extracted_features,
    name_a=test_set_feature_extractor.name, name_b=funk_set_feature_extractor.name).interset_distances_per_feat

# convert_distances_dict_to_pdf_histograms_dict
test_set_intraset_pdf = convert_distances_dict_to_pdf_histograms_dict(test_set_intraset_distances)
funk_intraset_pdf = convert_distances_dict_to_pdf_histograms_dict(funk_intraset_distances)
interset_pdf = convert_distances_dict_to_pdf_histograms_dict(interset_distances)

plot_set = funk_intraset_pdf
for key in plot_set.keys():
    # key = list(intraset_pdfs_rock.keys())[1]
    
    pdf, bins = plot_set[key]
    print(bins)
    width = (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, pdf, align='center', width=width)
    plt.plot(bins[:-1], pdf)
    plt.title(key)
    plt.show()"""

"""for feature in funk_intraset_distances.keys():
    # Find kernel bandwidth using Scott's Rule of Thumb
    # https://en.wikipedia.org/wiki/Histogram#Scott's_normal_reference_rule
    distances_in_feat = funk_intraset_distances[feature].flatten()
    sns.kdeplot(distances_in_feat)
    plt.title(feature)
    plt.show()"""

extracted_set = funk_set_feature_extractor.extract()
for feature in extracted_set.keys():
    # Find kernel bandwidth using Scott's Rule of Thumb
    # https://en.wikipedia.org/wiki/Histogram#Scott's_normal_reference_rule
    distances_in_feat = extracted_set[feature].flatten()
    sns.kdeplot(distances_in_feat)
    plt.title(feature)
    plt.show()
