#  Copyright (c) 2022. \n Created by Behzad Haki. behzad.haki@upf.edu

import torch
import json
import os

from model import VAE_components


class Density2D(torch.nn.Module):
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

        super(Density2D, self).__init__()

        assert config['o_activation'] in ['sigmoid', 'tanh'], 'offset_activation must be sigmoid or tanh'
        #assert config['embedding_size_src'] % 3 == 0, 'embedding_size_src must be divisible by 3'
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

        # New control params
        self.n_params = config['n_params']
        self.add_params = False
        #self.add_params = config['add_params'] # only to be used in 1D concatenation

        # Layers
        # ---------------------------------------------------

        self.InputLayerEncoder = VAE_components.InputLayer2DParam(
            embedding_size=self.embedding_size_src,
            n_params=self.n_params,
            d_model=self.d_model_enc,
            dropout=self.dropout,
            max_len=self.max_len_enc
        )

        self.Encoder = VAE_components.Encoder(
            d_model=self.d_model_enc,
            nhead=self.nhead_enc,
            dim_feedforward=self.dim_feedforward_enc,
            num_encoder_layers=self.num_encoder_layers,
            dropout=self.dropout)

        self.LatentEncoder = VAE_components.LatentLayer(
            max_len=self.max_len_enc, #changed dec to enc
            d_model=self.d_model_enc,
            latent_dim=self.latent_dim,
            add_params=self.add_params,
            n_params=self.n_params)

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

        # Initialize weights and biases
        # If tanh is used for offset activation, initialize the output layer's bias to 0.5
        self.InputLayerEncoder.init_weights()
        self.LatentEncoder.init_weights()
        self.Decoder.DecoderInput.init_weights()
        self.Decoder.OutputLayer.init_weights(offset_activation=self.o_activation)

    def get_latent_probs_and_reparametrize_to_z(self, src_, params_):
        x = self.InputLayerEncoder(src_, params_)  # Nx32xd_model
        memory = self.Encoder(x)  # Nx32xd_model
        mu, log_var, latent_z = self.LatentEncoder(memory)
        return mu, log_var, latent_z

    def encode(self, src, params):
        """ Encodes a given input sequence of shape (batch_size, seq_len, embedding_size_src) into a latent space
        of shape (batch_size, latent_dim)

        :param src: the input sequence
        :return: mu, log_var, latent_z (each of shape [batch_size, latent_dim])
        """

        if not self.training:
            with torch.no_grad():
                return self.get_latent_probs_and_reparametrize_to_z(src, params)
        else:
            return self.get_latent_probs_and_reparametrize_to_z(src, params)

    # def get_mu_var(self, src_):
    #     x = self.InputLayerEncoder(src_)  # Nx32xd_model
    #     memory = self.Encoder(x)  # Nx32xd_model
    #     mu, log_var, _ = self.LatentEncoder(memory)
    #     return mu, log_var
    #
    # def encode_to_mu_logvar(self, src):
    #
    #     if not self.training:
    #         with torch.no_grad():
    #             return get_mu_var(src)
    #     else:
    #         self.train()
    #         return get_mu_var(src)

    def reparametrize(self, mu, log_var):
        return self.LatentEncoder.reparametrize(mu, log_var)


    def decode(self, latent_z, threshold: float = 0.5):
        """ Decodes a given latent space of shape (batch_size, latent_dim) into a sequence of shape
        (batch_size, seq_len, embedding_size_tgt)

        :param latent_z: the latent space of shape (batch_size, latent_dim)
        :param threshold: (default 0.5) the threshold to use for hits
        :return: the output sequence of shape (batch_size, seq_len, embedding_size_tgt)
        """
        if not self.training:
            with torch.no_grad():
                return self.Decoder.decode(latent_z, threshold=threshold)
        else:
            return self.Decoder.decode(latent_z, threshold=threshold)

    @torch.jit.export
    def sample(self, latent_z, voice_thresholds, voice_max_count_allowed, sampling_mode: int = 0,
               temperature: float = 1.0):
        """Converts the latent vector into hit, vel, offset values

        :param latent_z: (Tensor) [N x latent_dim]
        :param voice_thresholds: (floatTensor) Thresholds for hit prediction
        :param voice_max_count_allowed: (floatTensor) Maximum number of hits to allow for each voice
        :param sampling_mode: (int) 0 for top-k sampling,
                                    1 for bernoulli sampling
        :param temperature: (float) temperature for sampling

        Returns:
        h, v, o, _h
        """
        return self.Decoder.sample(
            latent_z=latent_z,
            voice_thresholds=voice_thresholds,
            voice_max_count_allowed=voice_max_count_allowed,
            sampling_mode=sampling_mode,
            temperature=temperature)

    @torch.jit.export
    def sample_and_return_concatenated(self, latent_z, voice_thresholds, voice_max_count_allowed,
              sampling_mode: int = 0, temperature: float = 1.0):
        """Converts the latent vector into hit, vel, offset values and returns the concatenated tensor

        :param latent_z: (Tensor) [N x latent_dim]
        :param voice_thresholds: (list) Thresholds for hit prediction
        :param voice_max_count_allowed: (list) Maximum number of hits to allow for each voice
        :param sampling_mode: (int) 0 for thresholding,
                                    1 for bernoulli sampling
        :Returns:
        hvo, _h
        """
        return self.Decoder.sample_and_return_concatenated(
            latent_z=latent_z,
            voice_thresholds=voice_thresholds,
            voice_max_count_allowed=voice_max_count_allowed,
            sampling_mode=sampling_mode,
            temperature=temperature)

    def forward(self, src, params):
        """ Converts a given input sequence of shape (batch_size, seq_len, embedding_size_src) into a
        **pre-activation** output sequence of shape (batch_size, seq_len, embedding_size_tgt)

        :param src: the input sequence [batch_size, seq_len, embedding_size_src]
        :return: (h_logits, v_logits, o_logits), mu, log_var, latent_z
        """
        mu, log_var, latent_z = self.encode(src, params)
        h_logits, v_logits, o_logits = self.Decoder(latent_z)

        return (h_logits, v_logits, o_logits), mu, log_var, latent_z

    def encode_decode(self, src_, params_, thres: float):
        mu, log_var, latent_z = self.encode(src_, params_)
        h, v, o = self.Decoder.decode(latent_z, threshold=thres, use_thres=True)
        return (h, v, o), mu, log_var, latent_z

    @torch.jit.export
    def encode_decode_and_return_concatenated(self, src_, params_, thres: float):
        mu, log_var, latent_z = self.encode(src_, params_)
        hvo = self.Decoder.decode_and_return_concatenated(latent_z, threshold=thres, use_thres=True)
        return hvo, mu, log_var, latent_z
    def predict(self, src, params,  thres: float = 0.5):
        """
        Predicts the actual hvo array from the input Base
        :param src: the input sequence [batch_size, seq_len, embedding_size_src]
        :param thres: (default=0.5) the threshold to use for the output
        :return: (full_hvo_array, mu, log_var, latent_z) if return_concatenated is False, else
        ((h, v, o), mu, log_var, latent_z)
        """


        if not self.training:
            with torch.no_grad():
                return self.encode_decode(src, params, thres=thres)
        else:
            return self.encode_decode(src, params, thres=thres)

    @torch.jit.export
    def predict_and_return_concatenated(self, src, params,  thres: float = 0.5):
        """
        Predicts the actual hvo array from the input Base
        :param src: the input sequence [batch_size, seq_len, embedding_size_src]
        :param thres: (default=0.5) the threshold to use for the output
        :return: (full_hvo_array, mu, log_var, latent_z) if return_concatenated is False, else
        ((h, v, o), mu, log_var, latent_z)
        """
        if not self.training:
            with torch.no_grad():
                return self.encode_decode_and_return_concatenated(src, params, thres)
        else:
            return self.encode_decode_and_return_concatenated(src, params, thres)

    def save(self, save_path, additional_info=None):
        """ Saves the model to the given path. The Saved pickle has all the parameters ('params' field) as well as
        the state_dict ('state_dict' field) """
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
            'n_params': self.n_params,
            'add_params': self.add_params
        }

        json.dump(params_dict, open(save_path.replace('.pth', '.json'), 'w'))
        torch.save({'model_state_dict': self.state_dict(), 'params': params_dict,
                    'additional_info': additional_info}, save_path)

    # serializes to a torchscript model
    def serialize(self, save_folder):
        os.makedirs(save_folder, exist_ok=True)

        # save InputLayerEncoder
        input_layer_encoder = self.InputLayerEncoder
        torch.jit.save(torch.jit.script(input_layer_encoder), os.path.join(save_folder, 'InputLayerEncoder.pt'))

        # save Encoder
        encoder = self.Encoder
        torch.jit.save(torch.jit.script(encoder), os.path.join(save_folder, 'Encoder.pt'))

        # save LatentLayer
        latent_layer = self.LatentEncoder
        torch.jit.save(torch.jit.script(latent_layer), os.path.join(save_folder, 'LatentEncoder.pt'))

        # save Decoder
        decoder = self.Decoder
        torch.jit.save(torch.jit.script(decoder), os.path.join(save_folder, 'Decoder.pt'))

    def serialize_v2(self, filename_with_path):
        assert filename_with_path.endswith('.pt'), 'filename must end with .pt'
        os.makedirs(os.path.dirname(filename_with_path), exist_ok=True)
        # save model in single file
        with open(filename_with_path, 'wb') as f:
            torch.jit.save(torch.jit.script(self), f)
            print("Saved model to {}".format(filename_with_path))
