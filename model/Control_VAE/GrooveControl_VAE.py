import torch
import json

from model import VAE_components, Control_components

class GrooveControl_VAE(torch.nn.Module):

    def __init__(self, config):
        super(GrooveControl_VAE, self).__init__()
        assert config['embedding_size_tgt'] % 3 == 0, 'embedding_size_tgt must be divisible by 3'

        # Hyperparameters
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

        # Control parameters
        self.use_in_attention = config['use_in_attention']
        self.n_continuous_params = config['n_continuous_params']
        self.n_genres = config['n_genres']
        self.n_params = self.n_continuous_params + self.n_genres

        self.InputLayerEncoder = Control_components.ControlEncoderInputLayer(
            embedding_size=self.embedding_size_src,
            n_params=self.n_continuous_params,
            d_model=self.d_model_enc,
            dropout=self.dropout,
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

        mu, log_var, latent_z = self.encode(hvo, density, intensity)
        h_logits, v_logits, o_logits = self.Decoder(latent_z, density, intensity, genre)

        return (h_logits, v_logits, o_logits), mu, log_var, latent_z

    def encode(self, hvo, density, intensity):
        if not self.training:
            with torch.no_grad():
                return self.get_latent_probs_and_reparametrize_to_z(hvo, density, intensity)
        else:
            self.train()
            return self.get_latent_probs_and_reparametrize_to_z(hvo, density, intensity)

    def decode(self, latent_z, density, intensity, genre, threshold=0.5):
        if not self.training:
            with torch.no_grad():
                return self.Decoder.decode(latent_z, density, intensity, genre, threshold=0.5)
        else:
            self.train()
            return self.Decoder.decode(latent_z, density, intensity, genre, threshold=0.5)



    # ---Utility Functions--- #
    def get_latent_probs_and_reparametrize_to_z(self, hvo, density, intensity):
        # Input Layer -> Self Attention -> Latent FFN
        x = self.InputLayerEncoder(hvo, density, intensity)
        memory = self.Encoder(x)
        mu, log_var, latent_z = self.LatentEncoder(memory)
        return mu, log_var, latent_z

    def reparametrize(self, mu, log_var):
        return self.LatentEncoder.reparametrize(mu, log_var)

    def encode_decode(self, hvo, density, intensity, genre):
        mu, log_var, latent_z = self.encode(hvo, density, intensity)
        h, v, o = self.Decoder.decode(latent_z, )

