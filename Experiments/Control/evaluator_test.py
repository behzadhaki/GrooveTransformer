import os
import torch
from model import GrooVAEDensity1D
from helpers import vae_train_utils, vae_test_utils
from data.src.dataLoaders import MonotonicGrooveDataset, GrooveDataSet_Density
from torch.utils.data import DataLoader
from logging import getLogger, DEBUG
import yaml
import argparse
import numpy as np
from helpers.Control.eval_utils import get_logging_media_for_control_model_wandb




hparams = dict(
            d_model_enc = 32,
            d_model_dec = 32,
            embedding_size_src = 27,
            embedding_size_tgt = 27,
            nhead_enc = 4,
            nhead_dec = 4,
            dim_feedforward_enc = 128,
            dim_feedforward_dec = 128,
            num_encoder_layers = 2,
            num_decoder_layers = 2,
            dropout = 0.1,
            latent_dim = 16,
            max_len_enc = 32,
            max_len_dec = 32,
            device = 'cpu',
            o_activation = 'tanh',
            n_params = 1,
            add_params = True) # only to be used in horizontal concatenation (1D)



model = GrooVAEDensity1D(hparams)

media_list = get_logging_media_for_control_model_wandb(
            model=model,
            device='cpu',
            dataset_setting_json_path="../../data/dataset_json_settings/4_4_BeatsAndFills_gmd.json",
            subset_name='test',
            collapse_tapped_sequence=False,
            cached_folder="eval/GrooveEvaluator/templates",
            divide_by_genre=False,
            need_piano_roll=True,
            need_kl_plot=False,
            need_audio=False)

for item in media_list:
    for k, v in item.items():
        print(k)
        print(v)
    break
