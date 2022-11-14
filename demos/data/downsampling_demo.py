#  Copyright (c) 2022. \n Created by Behzad Haki. behzad.haki@upf.edu

from data import load_down_sampled_gmd_hvo_sequences

if __name__ == "__main__":

    test_data_down_sampled = load_down_sampled_gmd_hvo_sequences(
        dataset_setting_json_path="data/dataset_json_settings/4_4_BeatsAndFills_gmd.json",
        subset_tag="test",
        down_sampled_ratio=0.1,
        force_regenerate=False)