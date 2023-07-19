from data import GrooveDataSet_Control

# =================================================================================================
genre_mapping = {
        'rock': 0,
         'latin': 1,
         'jazz': 2,
         'funk': 3,
         'afrobeat': 4,
         'afrocuban': 5,
         'hiphop': 6,
         'dance': 7,
         'soul': 8,
         'reggae': 9,
         'country': 10,
         'pop': 11,
         'punk': 12,
         'blues': 13,
         'highlife': 14,
         'other': 15}

train_dataset = GrooveDataSet_Control(
        dataset_setting_json_path="data/dataset_json_settings/4_4_BeatsAndFills_gmd.json",
        subset_tag="train",
        max_len=32,
        tapped_voice_idx=2,
        collapse_tapped_sequence=False,
        down_sampled_ratio=None,    # 0.1
        move_all_to_gpu=False,
        hit_loss_balancing_beta=0,
        genre_loss_balancing_beta=0,
        custom_genre_mapping_dict=genre_mapping
    )

# genre_mapping_dict = train_dataset.get_genre_mapping_dict()
# train_dataset.genres

test_dataset = GrooveDataSet_Control(
        dataset_setting_json_path="data/dataset_json_settings/4_4_BeatsAndFills_gmd.json",
        subset_tag="test",
        max_len=32,
        tapped_voice_idx=2,
        collapse_tapped_sequence=False,
        down_sampled_ratio=None,    # 0.1
        move_all_to_gpu=False,
        hit_loss_balancing_beta=0,
        genre_loss_balancing_beta=0,
        custom_genre_mapping_dict=genre_mapping
    )

# plot syncopation distribution
import matplotlib.pyplot as plt
import numpy as np

def plot_histogram(data):
    plt.hist(data, bins=100)
    plt.show()

plot_histogram(test_dataset.syncopations.detach().cpu().numpy())
plot_histogram(train_dataset.densities.detach().cpu().numpy())
