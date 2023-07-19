import os
import torch
from model import Density1D
from helpers.Control.density_eval import get_piano_rolls_for_control_model_wandb
from data.src.dataLoaders import GrooveDataSet_Density


os.chdir("/Users/jlenz/Desktop/Thesis/GrooveTransformer")

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


model = Density1D(hparams)

test_dataset = GrooveDataSet_Density(
        dataset_setting_json_path="data/dataset_json_settings/4_4_BeatsAndFills_gmd.json",
        subset_tag="test",
        max_len=32,
        tapped_voice_idx=2,)

piano_rolls = get_piano_rolls_for_control_model_wandb(vae_model=model, device='cpu',
                                                      test_dataset=test_dataset)

print("lol")