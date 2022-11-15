#  Copyright (c) 2022. \n Created by Behzad Haki. behzad.haki@upf.edu

import torch
import json
import os

from model import VAE_components


class GrooveTransformerEncoderVAE(torch.nn.Module):
    """
    An encoder-encoder VAE transformer
    """
    def __init__(self, config):
        """
        This is a VAE transformer which for encoder and decoder uses the same transformer architecture
        (that is, uses the Vanilla Transformer Encoder)
        :param config: a dictionary containing the following keys:
            d_model_enc: the dimension of the model for the encoder
            d_model_dec: the dimension of the model for the decoder
            embedding_size_src: the dimension of the input embedding
            embedding_size_tgt: the dimension of the output embedding
            nhead_enc: the number of heads for the encoder
            nhead_dec: the number of heads for the decoder
            dim_feedforward_enc: the dimension of the feedforward network in the encoder
            dim_feedforward_dec: the dimension of the feedforward network in the decoder
            num_encoder_layers: the number of encoder layers
            num_decoder_layers: the number of decoder layers
            dropout: the dropout rate
            latent_dim: the dimension of the latent space
            max_len_enc: the maximum length of the input sequence
            max_len_dec: the maximum length of the output sequence
            device: the device to use
            o_activation: the activation function to use for the output
        """

        super(GrooveTransformerEncoderVAE, self).__init__()

        assert config['o_activation'] in ['sigmoid', 'tanh'], 'offset_activation must be sigmoid or tanh'
        assert config['embedding_size_src'] % 3 == 0, 'embedding_size_src must be divisible by 3'
        assert config['embedding_size_tgt'] % 3 == 0, 'embedding_size_tgt must be divisible by 3'

        # HParams
        # ---------------------------------------------------
        self.d_model_enc = config['d_model_enc']
        self.d_model_dec = config['d_model_dec']
        self.embedding_size_src = config['embedding_size_src']
        self.embedding_size_tgt = config['embedding_size_tgt']
        self.nhead_enc = config['nhead_enc']
        self.nhead_dec = config['nhead_dec']
        self.dim_feedforward_enc = config['dim_feedforward_enc']
        self.dim_feedforward_dec = config['dim_feedforward_dec']
        self.num_encoder_layers = config['num_encoder_layers']
        self.num_decoder_layers = config['num_decoder_layers']
        self.dropout = config['dropout']
        self.latent_dim = config['latent_dim']
        self.max_len_enc = config['max_len_enc']
        self.max_len_dec = config['max_len_dec']
        self.device = config['device']
        self.o_activation = config['o_activation']

        # Layers
        # ---------------------------------------------------
        self.InputLayerEncoder = VAE_components.InputLayer(
            embedding_size=self.embedding_size_src,
            d_model=self.d_model_enc,
            dropout=self.dropout,
            max_len=self.max_len_enc)

        self.Encoder = VAE_components.Encoder(
            d_model=self.d_model_enc,
            nhead=self.nhead_enc,
            dim_feedforward=self.dim_feedforward_enc,
            num_encoder_layers=self.num_encoder_layers,
            dropout=self.dropout)

        self.LatentEncoder = VAE_components.reparameterize(
            max_len=self.max_len_dec,
            d_model=self.d_model_enc,
            latent_dim=self.latent_dim)

        self.Decoder = VAE_components.VAE_Decoder(
            latent_dim=self.latent_dim,
            d_model=self.d_model_dec,
            num_decoder_layers=self.num_decoder_layers,
            nhead=self.nhead_dec,
            dim_feedforward=self.dim_feedforward_dec,
            output_max_len=self.max_len_dec,
            output_embedding_size=self.embedding_size_tgt,
            dropout=self.dropout,
            o_activation=self.o_activation)

        self.InputLayerEncoder.init_weights()
        self.Decoder.OutputLayer.init_weights()

    def encode(self, src):
        x = self.InputLayerEncoder(src)  # Nx32xd_model
        print("x", x.shape)
        memory = self.Encoder(x)  # Nx32xd_model
        print("memory", memory.shape)
        mu, log_var, latent_z = self.LatentEncoder(memory)
        print(f"mu: {mu.shape}, log_var: {log_var.shape}, latent_z: {latent_z.shape}")
        return mu, log_var, latent_z

    def forward(self, src):
        print("src", src.shape)
        mu, log_var, latent_z = self.encode(src)
        print(f"mu: {mu.shape}, log_var: {log_var.shape}, latent_z: {latent_z.shape}")
        h_logits, v_logits, o_logits = self.Decoder(latent_z)

        return (h_logits, v_logits, o_logits), mu, log_var

    def predict(self, src, use_thres=True, thres=0.5, use_pd=False):
        """
        Predicts the actual hvo array from the input Base
        :param src:
        :param use_thres:
        :param thres:
        :param use_pd:
        :return: (full_hvo_array, mu, log_var)
        """
        self.eval()
        with torch.no_grad():
            mu, log_var, latent_z = self.encode(src)
            h, v, o = self.Decoder.predict(latent_z, use_thres=use_thres, threshold=thres, use_pd=use_pd)
            hvo = torch.cat((h, v, o), dim=-1).detach().cpu().numpy()
        return hvo, mu, log_var

    def save(self, save_path, additional_info=None):
        if not save_path.endswith('.pth'):
            save_path += '.pth'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        params_dict = {
            'd_model_enc': self.d_model_enc,
            'd_model_dec': self.d_model_dec,
            'embedding_size_src': self.embedding_size_src,
            'embedding_size_tgt': self.embedding_size_tgt,
            'nhead_enc': self.nhead_enc,
            'nhead_dec': self.nhead_dec,
            'dim_feedforward_enc': self.dim_feedforward_enc,
            'dim_feedforward_dec': self.dim_feedforward_dec,
            'num_encoder_layers': self.num_encoder_layers,
            'num_decoder_layers': self.num_decoder_layers,
            'dropout': self.dropout,
            'latent_dim': self.latent_dim,
            'max_len_enc': self.max_len_enc,
            'max_len_dec': self.max_len_dec,
            'device': self.device.type if isinstance(self.device, torch.device) else self.device,
            'o_activation': self.o_activation,
        }

        json.dump(params_dict, open(save_path.replace('.pth', '.json'), 'w'))
        torch.save({'model_state_dict': self.state_dict(), 'params': params_dict,
                    'additional_info': additional_info}, save_path)
