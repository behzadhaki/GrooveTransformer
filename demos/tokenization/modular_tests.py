import os
import torch
import numpy
from data.src.dataLoaders import MonotonicGrooveTokenizedDataset

os.chdir("../../")

dataset = MonotonicGrooveTokenizedDataset(
        dataset_setting_json_path="data/dataset_json_settings/BeatsAndFills_gmd_96.json",
        subset_tag="test", max_length=500)

idx, in_token, in_hv, out_token, out_hv, masks = dataset[3]

print(idx)
print(in_token.shape)
print(in_hv.shape)
print(out_token.shape)
print(out_hv.shape)
print(masks.shape)
print(masks)