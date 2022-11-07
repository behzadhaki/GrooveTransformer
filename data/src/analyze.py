import os, sys

from itertools import combinations
from tqdm import tqdm
from bokeh.layouts import gridplot
from data.src.utils import get_per_performer_bokeh_histogram, get_genre_performer_heatmaps
from data import load_gmd_hvo_sequences

if __name__ == '__main__':
    data_identifier = "4_4_Beats_gmd"

    histograms = []
    heatmaps = []

    for subset_tag in ["train", "test", "validation"]:
        # ==================================================================================================================
        # Load Data
        # ==================================================================================================================
        data_set = load_gmd_hvo_sequences(
            dataset_setting_json_path=f"data/dataset_json_settings/{data_identifier}.json",
            subset_tag=subset_tag,
            force_regenerate=False)

        # ==================================================================================================================
        # group by master_id
        # ==================================================================================================================
        from data.src.utils import group_by_master_id
        # save_midis_at = f"data/misc/{data_identifier}_{subset_tag}_sorted_midis
        grouped_by_master_id = group_by_master_id(data_set, save_midis_at=None)

        # ==================================================================================================================
        # group by genre
        # ==================================================================================================================
        grouped_by_genre = {}
        for hvo_sample in data_set:
            if hvo_sample.metadata["style_primary"] not in grouped_by_genre:
                grouped_by_genre[hvo_sample.metadata["style_primary"]] = []
            grouped_by_genre[hvo_sample.metadata["style_primary"]].append(hvo_sample)
        for key, item in grouped_by_genre.items():
            grouped_by_genre[key] = group_by_master_id(item, save_midis_at=None)

        # Get stats per genre
        genre_stats = {}
        for genre, master_id_dict in grouped_by_genre.items():
            genre_stats[genre] = {}
            for master_id, hvo_seq_list in master_id_dict.items():
                genre_stats[genre][master_id] = len(hvo_seq_list)
        for genre, master_id_dict in grouped_by_genre.items():
            genre_stats[genre]["overal"] = sum(genre_stats[genre].values())

        # ==================================================================================================================
        # calculate number of events per each bar
        # NOTE
        #   each loop is one bar ahead of the previous
        # ==================================================================================================================
        NoI = {}       # number of voices in each sample
        NoH = {}       # number of hits in each sample
        WH = {}        # weighted hits by number of voices

        for key, items in tqdm(grouped_by_master_id.items(), desc="calculating number of events per bar"):
            for ix, item in enumerate(items):
                if key not in NoI:
                    NoI[key] = []
                    NoH[key] = []
                    WH[key] = []
                NoI[key].append(item.get_number_of_active_voices())
                NoH[key].append(item.hits.sum())
                WH[key].append(item.hits.sum()/item.get_number_of_active_voices())

        # sort groups by number of associated weighted hits
        grouped_by_master_id_sorted_by_WH = {}
        for key, items in tqdm(grouped_by_master_id.items(), desc="sorting groups by number of associated weighted hits"):
            grouped_by_master_id_sorted_by_WH[key] = [y for _, y in sorted(zip(WH[key], items), key=lambda tup: tup[0])]


