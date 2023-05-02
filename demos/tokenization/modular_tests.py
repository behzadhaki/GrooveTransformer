import os
import torch
import numpy
from data.src.dataLoaders import MonotonicGrooveTokenizedDataset
from torch.utils.data import DataLoader
from model.BaseTokenize.shared_model_components import *

os.chdir("../../")

dataset = MonotonicGrooveTokenizedDataset(
        dataset_setting_json_path="data/dataset_json_settings/BeatsAndFills_gmd_96.json",
        subset_tag="test", max_length=500)

data_loader = DataLoader(dataset, batch_size=16, shuffle=True)

for data in data_loader:
        single_batch = data
        break

idx, in_token, in_hv, out_token, out_hv, masks = single_batch

input_layer = InputLayer(embedding_size=18, d_model=32, n_token_types=)

# idx, in_token, in_hv, out_token, out_hv, masks = dataset[3]
#
# print(idx)
# print(in_token.shape)
# print(in_hv.shape)
# print(out_token.shape)
# print(out_hv.shape)
# print(masks.shape)
# print(masks)







