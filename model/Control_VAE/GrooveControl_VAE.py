import torch
import os
import json
from model import VAE_components, Control_components


class GrooveControl_VAE(torch.nn.Module):

    def __init__(self, config):
        super(GrooveControl_VAE, self).__init__()
        assert config['embedding_size_tgt'] % 3 == 0, 'embedding_size_tgt must be divisible by 3'

        # Model Hyperparameters
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
        self.max_len_enc = config['max_len_enc']
        self.max_len_dec = config['max_len_dec']
        self.latent_dim = config['latent_dim']

        # Dropouts
        self.dropout = config['dropout']
        self.velocity_dropout = config['velocity_dropout']
        self.offset_dropout = config['offset_dropout']

        # Misc
        self.device = config['device']
        self.o_activation = config['o_activation']

        # Control parameters
        self.use_in_attention = config['use_in_attention']
        self.n_continuous_params = config['n_continuous_params']
        self.genre_dict = config['genre_dict']
        self.n_genres = len(self.genre_dict)
        self.n_params = self.n_continuous_params + self.n_genres
        print(f"Using In-Attention: {self.use_in_attention}")

        self.InputLayerEncoder = Control_components.ControlEncoderInputLayer(
            embedding_size=self.embedding_size_src,
            d_model=self.d_model_enc,
            dropout=self.dropout,
            velocity_dropout=self.velocity_dropout,
            offset_dropout=self.offset_dropout,
            max_len=self.max_len_enc)

        self.Encoder = VAE_components.Encoder(
            d_model=self.d_model_enc,
            nhead=self.nhead_enc,
            dim_feedforward=self.dim_feedforward_enc,
            num_encoder_layers=self.num_encoder_layers,
            dropout=self.dropout)

        self.LatentEncoder = VAE_components.LatentLayer(
            max_len=self.max_len_enc,  # changed dec to enc
            d_model=self.d_model_enc,
            latent_dim=self.latent_dim,
            add_params=False,
            n_params=self.n_continuous_params)

        self.Decoder = Control_components.Control_Decoder(
            latent_dim=self.latent_dim,
            d_model=self.d_model_dec,
            num_decoder_layers=self.num_decoder_layers,
            nhead=self.nhead_dec,
            dim_feedforward=self.dim_feedforward_dec,
            output_max_len=self.max_len_dec,
            output_embedding_size=self.embedding_size_tgt,
            dropout=self.dropout,
            o_activation=self.o_activation,
            n_genres=self.n_genres,
            in_attention=self.use_in_attention
        )

    def forward(self, hvo, density, intensity, genre):
        mu, log_var, latent_z = self.encode(hvo)
        h_logits, v_logits, o_logits = self.Decoder(latent_z, density, intensity, genre)

        return (h_logits, v_logits, o_logits), mu, log_var, latent_z

    def predict(self, hvo, density, intensity, genre, thresh=0.5, return_concatenated=False):
        if not self.training:
            with torch.no_grad():
                return self.encode_decode(hvo, density, intensity, genre,
                                          threshold=thresh, use_thresh=True, use_pd=False,
                                          return_concatenated=return_concatenated)
        else:
            return self.encode_decode(hvo, density, intensity, genre,
                                      threshold=thresh, use_thresh=True,
                                      use_pd=False, return_concatenated=return_concatenated)

    def encode_decode(self, hvo, density, intensity, genre,
                      threshold=0.5, use_thresh=True, use_pd=False, return_concatenated=False):

        mu, log_var, latent_z = self.encode(hvo)
        h, v, o = self.Decoder.decode(latent_z, density, intensity, genre,
                                      threshold=threshold, use_thresh=use_thresh,
                                      use_pd=use_pd, return_concatenated=False)

        if return_concatenated:
            hvo_out = torch.cat((h, v, o), dim=-1)
            return hvo_out, mu, log_var, latent_z
        else:
            return (h, v, o), mu, log_var, latent_z


    # Todo: Check with Behzad if the below works

    def encode(self, hvo):
        if self.training:
            return self.get_latent_probs_and_reparametrize_to_z(hvo)
        else:
            with torch.no_grad():
                return self.get_latent_probs_and_reparametrize_to_z(hvo)

    def decode(self, latent_z, density, intensity, genre,
               threshold=0.5, use_thresh=True, use_pd=False, return_concatenated=False):
        if not self.training:
            with torch.no_grad():
                return self.Decoder.decode(latent_z, density, intensity, genre,
                                           threshold, use_thresh, use_pd, return_concatenated)
        else:
            self.train()
            return self.Decoder.decode(latent_z, density, intensity, genre,
                                       threshold, use_thresh, use_pd, return_concatenated)

    # ---Utility Functions--- #
    def get_latent_probs_and_reparametrize_to_z(self, hvo):
        # Input Layer -> Self Attention -> Latent FFN
        x = self.InputLayerEncoder(hvo)
        memory = self.Encoder(x)
        mu, log_var, latent_z = self.LatentEncoder(memory)
        return mu, log_var, latent_z

    def reparametrize(self, mu, log_var):
        return self.LatentEncoder.reparametrize(mu, log_var)

    def sample(self, latent_z, density, intensity, genre,
               voice_thresholds, voice_max_count_allowed,
               return_concatenated=False, sampling_mode=0):

        return self.Decoder.sample(latent_z, density, intensity,
                                   genre, voice_thresholds, voice_max_count_allowed,
                                   return_concatenated, sampling_mode)

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
            'max_len_enc': self.max_len_enc,
            'max_len_dec': self.max_len_dec,
            'latent_dim': self.latent_dim,

            'dropout': self.dropout,
            'velocity_dropout': self.velocity_dropout,
            'offset_dropout': self.offset_dropout,

            'device': self.device.type if isinstance(self.device, torch.device) else self.device,
            'o_activation': self.o_activation,

            'use_in_attention': self.use_in_attention,
            'n_continuous_params': self.n_continuous_params,
            'n_genres': self.n_genres,
        }

        json.dump(params_dict, open(save_path.replace('.pth', '.json'), 'w'))
        torch.save({'model_state_dict': self.state_dict(), 'params': params_dict,
                    'additional_info': additional_info}, save_path)


    def serialize_whole_model(self, model_name, save_folder):
        """
        New method for serializing; instead of individual components, serializes the entire model
        as a single entity
        @param model_name: (str) name of the model
        @param save_folder: path to save the model in
        @return:
        """
        os.makedirs(save_folder, exist_ok=True)
        serialized_model = torch.jit.script(self)
        torch.jit.save(serialized_model, os.path.join(save_folder, (model_name + '.pt')))

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

    def get_genre_dict(self):
        return self.genre_dict
