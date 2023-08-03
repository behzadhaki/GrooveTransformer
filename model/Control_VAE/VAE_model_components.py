import torch
from copy import deepcopy
from model.VAE.shared_model_components import PositionalEncoding, Encoder, OutputLayer
from model import get_hits_activation


# --- Encoder Class and Components --- #
class ControlEncoderInputLayer(torch.nn.Module):
    """
    Input layer for the Encoder portion of the VAE control model.
    It takes a single, collapsed HVO input of size [batch, 32, 3] and returns a
    tensor of shape [batch, 32, d_model].
    """

    def __init__(self, embedding_size, d_model,
                 dropout, velocity_dropout, offset_dropout, max_len):
        super(ControlEncoderInputLayer, self).__init__()

        self.velocity_dropout = torch.nn.Dropout(velocity_dropout)
        self.offset_dropout = torch.nn.Dropout(offset_dropout)
        self.Linear = torch.nn.Linear(embedding_size, d_model, bias=True)
        self.ReLU = torch.nn.ReLU()
        self.PositionalEncoding = PositionalEncoding(d_model, max_len, dropout)

    def init_weights(self, initrange=0.1):
        self.Linear.bias.data.zero_()
        self.Linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, hvo):
        """
        Apply dropout to V and O tensors, as well as a FFN and positional encoding
        @param hvo: (tensor) of shape [batch, max_len, 3]
        @return: tensor of shape [batch, max_len, d_model]
        """
        # Apply dropouts to velocity and offset
        h, v, o = hvo.split(1, dim=-1)
        v = self.velocity_dropout(v)
        o = self.velocity_dropout(o)
        src = torch.cat((h, v, o), dim=-1)

        # FFN and positional encoding
        x = self.Linear(src)
        x = self.ReLU(x)
        out = self.PositionalEncoding(x)

        return out


# --- Decoder Class and Components --- #
class Control_Decoder(torch.nn.Module):

    def __init__(self, latent_dim, d_model, num_decoder_layers, nhead, dim_feedforward,
                 output_max_len, output_embedding_size, dropout, o_activation, n_genres, in_attention=False):
        super(Control_Decoder, self).__init__()

        assert o_activation in ["sigmoid", "tanh"]

        self.latent_dim = latent_dim
        self.d_model = d_model
        self.num_decoder_layers = num_decoder_layers
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.output_max_len = output_max_len
        self.output_embedding_size = output_embedding_size
        self.dropout = dropout
        self.o_activation = o_activation
        self.n_genres = n_genres
        self.n_params = n_genres + 2  # genre and intensity
        self.in_attention = in_attention


        self.DecoderInputLayer = ControlDecoderInputLayer(
            max_len=self.output_max_len,
            latent_dim=self.latent_dim,
            d_model=self.d_model,
            n_genres=self.n_genres)


        self.InAttentionDecoder = InAttentionEncoder(
            d_model=self.d_model,
            n_head=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            num_encoder_layers=self.num_decoder_layers,
            n_params=self.n_params)


        self.Decoder = Encoder(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            num_encoder_layers=self.num_decoder_layers,
            dropout=self.dropout)

        self.OutputLayer = OutputLayer(
            embedding_size=self.output_embedding_size,
            d_model=self.d_model)

    def forward(self, latent_z, density, intensity, genre):
        """
        Takes the latent vector z, as well as ground truth (or predicted) control values of
        density, intensity and genre and outputs prediction logits for hits, velocities and offsets
        @param latent_z: (Tensor) [N x latent_dim] latent vector from encoder
        @param density: (Tensor) [batch] single float values of ground-truth densities
        @param intensity: (Tensor) [batch] single float values of ground-truth intensities
        @param genre: (tensor) [batch, n_genres] one-hot encoded ground-truth genre id
        @return: h, v, o (tensors) [N, output_max_len, output_embedding_size/3]
        """
        pre_out = self.DecoderInputLayer(latent_z, density, intensity, genre)


        if self.in_attention:
            decoder_ = self.InAttentionDecoder(pre_out, density, intensity, genre)

        else:
            decoder_ = self.Decoder(pre_out)

        h_logits, v_logits, o_logits = self.OutputLayer(decoder_)

        return h_logits, v_logits, o_logits

    def decode(self, latent_z, density, intensity, genre,
               threshold=0.5, use_thresh=True, use_pd=False, return_concatenated=False):
        """Converts the latent vector into hit, vel, offset values

        :param latent_z: (Tensor) [N x latent_dim]
        :param threshold: (float) Threshold for hit prediction
        :param use_thresh: (bool) Whether to use thresholding for hit prediction
        :param use_pd: (bool) Whether to use a pd for hit prediction
        :param return_concatenated: (bool) Whether to return the concatenated tensor or the individual tensors
        **For now only thresholding is supported**

        :return: (Tensor) h, v, o (each with dimension [N x max_len x num_voices])"""

        self.eval()
        with torch.no_grad():
            h_logits, v_logits, o_logits = self.forward(latent_z, density, intensity, genre)
            h = get_hits_activation(h_logits, use_thres=use_thresh, thres=threshold, use_pd=use_pd)
            v = torch.sigmoid(v_logits)

            if self.o_activation == "tanh":
                o = torch.tanh(o_logits) * 0.5
            elif self.o_activation == "sigmoid":
                o = torch.sigmoid(o_logits) - 0.5
            else:
                raise ValueError(f"{self.o_activation} for offsets is not supported")

        return h, v, o if not return_concatenated else torch.cat([h, v, o], dim=-1)

    def sample(self, latent_z, density, intensity, genre, voice_thresholds,
               voice_max_count_allowed, return_concatenated=False, sampling_mode=0):
        """Converts the latent vector into hit, vel, offset values

        :param latent_z: (Tensor) [N x latent_dim]
        :param voice_thresholds: (list) Thresholds for hit prediction
        :param voice_max_count_allowed: (list) Maximum number of hits to allow for each voice
        :param return_concatenated: (bool) Whether to return the concatenated tensor or the individual tensors
        :param sampling_mode: (int) 0 for top-k sampling,
                                    1 for bernoulli sampling
        """
        self.eval()
        with torch.no_grad():
            h_logits, v_logits, o_logits = self.forward(latent_z, density, intensity, genre)
            _h = torch.sigmoid(h_logits)
            h = torch.zeros_like(_h)

            v = torch.sigmoid(v_logits)

            if self.o_activation == "tanh":
                o = torch.tanh(o_logits) * 0.5
            elif self.o_activation == "sigmoid":
                o = torch.sigmoid(o_logits) - 0.5
            else:
                raise ValueError(f"{self.o_activation} for offsets is not supported")

            if sampling_mode == 0:
                for ix, (thres, max_count) in enumerate(zip(voice_thresholds, voice_max_count_allowed)):
                    max_indices = torch.topk(_h[:, :, ix], max_count).indices[0]
                    h[:, max_indices, ix] = _h[:, max_indices, ix]
                    h[:, :, ix] = torch.where(h[:, :, ix] > thres, 1, 0)
            elif sampling_mode == 1:
                for ix, (thres, max_count) in enumerate(zip(voice_thresholds, voice_max_count_allowed)):
                    # sample using probability distribution of hits (_h)
                    voice_probs = _h[:, :, ix]
                    sampled_indices = torch.bernoulli(voice_probs)
                    max_indices = torch.topk(sampled_indices * voice_probs, max_count).indices[0]
                    h[:, max_indices, ix] = 1

            # sample using probability distribution of velocities (v)
            if return_concatenated:
                return torch.concat((h, v, o), -1)
            else:
                return h, v, o


class ControlDecoderInputLayer(torch.nn.Module):
    """
    Takes a latent layer, as well as 3 control parameters (density, intensity, genre)
    and transforms into a single [Batch_size, max_len, d_model] tensor
    The initial z is fed through an FFN, to a size of (d_model - 3).
    The density and intensity continuous values are repeated and concatenated on final dimension
    Genre (one-hot encoding) is fed through a separate FFN and concatenated on final dimension
    output = concat(FFN(z) + repeat(density) + repeat(intensity) + FFN(genre))
    """
    def __init__(self, max_len, latent_dim, d_model, n_genres):
        super(ControlDecoderInputLayer, self).__init__()

        self.max_len = max_len
        self.d_model = d_model

        latent_projection_size = int(max_len * (d_model - 3))
        self.latent_linear = torch.nn.Linear(latent_dim, latent_projection_size)
        self.genre_linear = torch.nn.Linear(n_genres, max_len)

    def init_weights(self, initrange=0.1):
        self.latent_linear.bias.data.zero_()
        self.latent_linear.weight.data.uniform_(-initrange, initrange)
        self.genre_linear.bias.data.zero_()
        self.genre_linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, latent_z, density, intensity, genre):
        """
        @param latent_z: (tensor) [batch, latent_dim] the latent vector produced by the VAE encoder
        @param density: (tensor) [batch] single float values of ground-truth densities
        @param intensity: (tensor) [batch] single float values of ground-truth intensities
        @param genre: (tensor) [batch, n_genres] one-hot encoded ground-truth genre id
        @return: (tensor) [batch, max_len, d_model]
        """
        latent_z = self.latent_linear.forward(latent_z)
        latent_z = latent_z.view(-1, self.max_len, (self.d_model - 3))
        density = density.view(density.shape[0], 1).repeat(1, self.max_len).unsqueeze(dim=-1)
        intensity = intensity.view(intensity.shape[0], 1).repeat(1, self.max_len).unsqueeze(dim=-1)
        genre = self.genre_linear.forward(genre).unsqueeze(dim=-1)
        concat = torch.cat((latent_z, density, intensity, genre), dim=-1)


        return concat


# --- In Attention Decoding --- #
class InAttentionEncoder(torch.nn.Module):
    """
    This function is a modified version of the In-Attention Transformer from Musemorphose paper
    https://arxiv.org/abs/2105.04090
    We create a stack of num_encoder_layers transformer encoders. Separately, we expect control parameters to be
    a vector of num_params length. This vector is expanded to d_model with a learnable Linear layer.
    Then, at each Layer, it is summed with the source (which initially comes from sampling VAE latent space)
    and fed through the next iteration of multi-head self-attention.
    """

    def __init__(self, d_model, n_head, dim_feedforward, dropout, num_encoder_layers, n_params):
        super(InAttentionEncoder, self).__init__()
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model, n_head, dim_feedforward, dropout)
        self.layers = create_layers(encoder_layer, num_encoder_layers)
        self.parameter_projection = torch.nn.Linear(n_params, d_model)

    def forward(self, x, density, intensity, genre):
        # Concatenate the three parameters into a single [batch_size, total_params] tensor
        parameters = torch.concat((density.unsqueeze(dim=-1), intensity.unsqueeze(dim=-1), genre), dim=-1)
        parameters = self.parameter_projection(parameters).unsqueeze(1)

        # Add the parameters projection prior to each self-attention step
        for mod in self.layers:
            x += parameters
            x = mod(x)
        return x


def create_layers(module, N):
    # User to stack multiple Encoder layers for In-Attention mechanism
    return torch.nn.ModuleList([deepcopy(module) for i in range(N)])



