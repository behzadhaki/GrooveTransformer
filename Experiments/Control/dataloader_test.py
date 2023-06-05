from data.src.dataLoaders import GrooveDataSet_Density
import torch

training_dataset = GrooveDataSet_Density(
        dataset_setting_json_path="../../data/dataset_json_settings/4_4_BeatsAndFills_gmd.json",
        subset_tag="train",
        max_len=32,
        tapped_voice_idx=2,
        collapse_tapped_sequence=False,
        down_sampled_ratio=None,
        move_all_to_gpu=False,
        hit_loss_balancing_beta=0,
        genre_loss_balancing_beta=0
    )

densities = training_dataset.get_densities()

print(torch.max(densities))
print(torch.min(densities))