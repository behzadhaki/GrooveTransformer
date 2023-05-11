import os
import json

import torch

import model.BaseTokenize.shared_model_components
from model.BaseTokenize import shared_model_components
from model.BaseTokenize.utils import *



class TokenizedTransformerEncoder(torch.nn.Module):
    """
    An encoder-only transformer
    """
    def __init__(self, config):
        super(TokenizedTransformerEncoder, self).__init__()

        self.d_model = config['d_model']
        self.embedding_size = config['embedding_size']
        self.nhead = config['nhead']
        self.dim_feedforward = config['dim_feedforward']
        self.num_encoder_layers = config['num_encoder_layers']
        self.dropout = config['dropout']
        self.max_len = config['max_len']
        self.device = config['device']

        # New parameters from tokenization
        self.n_token_types = config['n_token_types']
        self.token_embedding_ratio = config['token_embedding_ratio']
        self.token_type_loc = config['token_type_loc']
        self.padding_idx = config['padding_idx']
        self.n_voices = config['n_voices']


    # Layers
    # ---------------------------------------------------
        self.InputLayer = model.BaseTokenize.shared_model_components.InputLayer(
            embedding_size=self.embedding_size,
            d_model=self.d_model,
            n_token_types=self.n_token_types,
            token_embedding_ratio=self.token_embedding_ratio,
            token_type_loc=self.token_type_loc,
            padding_idx=self.padding_idx
        )

        self.Encoder = model.BaseTokenize.shared_model_components.Encoder(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            num_encoder_layers=self.num_encoder_layers,
            max_len=self.max_len
        )

        self.OutputLayer = model.BaseTokenize.shared_model_components.OutputLayer(
            n_token_types=self.n_token_types,
            n_voices=self.n_voices,
            d_model=self.d_model
        )

        # Initialize weights
        self.InputLayer.init_weights()
        self.OutputLayer.init_weights()

    def forward(self, input_tokens, input_hv, mask=None):
        # model Nx32xembedding_size_src
        x = self.InputLayer(input_tokens, input_hv)  # Nx32xd_model
        x = self.Encoder(x, src_key_padding_mask=mask)
        token_type_logits, h_logits, v_logits = self.OutputLayer(x)  # Nx32xd_model

        return token_type_logits, h_logits, v_logits

    def predict(self, input_tokens, input_hv, mask=None, threshold=0.5, return_concatenated=False):

        def encode_decode(input_tokens_, input_hv_, masks_):
            x = self.InputLayer(input_tokens_, input_hv_)
            x = self.Encoder(x, src_key_padding_mask=masks_)
            token_type, h, v = self.OutputLayer.decode(x, threshold=threshold)
            return token_type, h, v

        if not self.training:
            with torch.no_grad():
                return encode_decode(input_tokens, input_hv, mask)
        else:
            return encode_decode(input_tokens, input_hv, mask)





    def save(self, save_path, additional_info=None):
        if not save_path.endswith('.pth'):
            save_path += '.pth'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        params_dict = {
            'd_model': self.d_model,
            'embedding_size': self.embedding_size,
            'nhead': self.nhead,
            'dim_feedforward': self.dim_feedforward,
            'num_encoder_layers': self.num_encoder_layers,
            'dropout': self.dropout,
            'max_len': self.max_len,
            'device': self.device,
            'n_token_types': self.n_token_types,
            'token_embedding_ratio': self.token_embedding_ratio,
            'token_type_loc': self.token_type_loc,
            'padding_idx': self.padding_idx,
            'n_voices': self.n_voices
        }

        json.dump(params_dict, open(save_path.replace('.pth', '.json'), 'w'))
        torch.save({'model_state_dict': self.state_dict(), 'params': params_dict,
                    'additional_info': additional_info}, save_path)