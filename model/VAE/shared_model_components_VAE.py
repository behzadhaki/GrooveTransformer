#  Copyright (c) 2022. \n Created by Behzad Haki. behzad.haki@upf.edu

import torch
import math
from model import get_hits_activation



# --------------------------------------------------------------------------------
# ------------       Positinal Encoding BLOCK                ---------------------
# --------------------------------------------------------------------------------
class PositionalEncoding(torch.nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, max_len, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # shape (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float)  # Shape (max_len)
        position = position.unsqueeze(1)  # Shape (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # Shape (d_model/2)
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 != 0:
            pe[:, 1::2] = torch.cos(position * div_term)[:, :-1]
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        # Insert a new dimension for batch size
        pe = pe.unsqueeze(0)  # Shape (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


# --------------------------------------------------------------------------------
# ------------                  ENCODER BLOCK                ---------------------
# --------------------------------------------------------------------------------
class Encoder(torch.nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward, num_encoder_layers, dropout, ):
        super(Encoder, self).__init__()
        norm_encoder = torch.nn.LayerNorm(d_model)
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        self.Encoder = torch.nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_encoder_layers,
            norm=norm_encoder)

    def forward(self, src):
        """
        input and output both have shape (batch, seq_len, embed_dim)
        :param src:
        :return:
        """
        out = self.Encoder(src)
        return out

# # --------------------------------------------------------------------------------
# # ------------                  DECODER BLOCK                ---------------------
# # --------------------------------------------------------------------------------
# def get_tgt_mask(d_model):
#     mask = (torch.triu(torch.ones(d_model, d_model)) == 1).transpose(0, 1).float()
#     mask = mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
#     return mask
#
# class Decoder(torch.nn.Module):
#
#     def __init__(self, d_model, nhead, dim_feedforward, dropout, num_decoder_layers):
#         super(Decoder, self).__init__()
#         norm_decoder = torch.nn.LayerNorm(d_model)
#         decoder_layer = torch.nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
#         self.Decoder = torch.nn.TransformerDecoder(decoder_layer, num_decoder_layers, norm_decoder)
#
#     def forward(self, tgt, memory, tgt_mask):
#         # tgt    Nx32xd_model
#         # memory Nx32xd_model
#
#         tgt = tgt.permute(1, 0, 2)  # 32xNxd_model
#         memory = memory.permute(1, 0, 2)  # 32xNxd_model
#
#         out = self.Decoder(tgt, memory, tgt_mask)  # 32xNxd_model
#
#         out = out.permute(1, 0, 2)  # Nx32xd_model
#
#         return out

# --------------------------------------------------------------------------------
# ------------                     I/O Layers                ---------------------
# --------------------------------------------------------------------------------
class InputLayer(torch.nn.Module):
    def __init__(self, embedding_size, d_model, dropout, max_len):
        super(InputLayer, self).__init__()

        self.Linear = torch.nn.Linear(embedding_size, d_model, bias=True)
        self.ReLU = torch.nn.ReLU()
        self.PositionalEncoding = PositionalEncoding(d_model, max_len, dropout)

    def init_weights(self, initrange=0.1):
        self.Linear.bias.data.zero_()
        self.Linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        x = self.Linear(src)
        x = self.ReLU(x)
        out = self.PositionalEncoding(x)

        return out

class OutputLayer(torch.nn.Module):
    def __init__(self, embedding_size, d_model):
        """
        Output layer of the transformer model
        :param embedding_size: size of the embedding (output dim at each time step)
        :param d_model:     size of the model         (input dim at each time step)
        :param offset_activation:   activation function for the offset (default: tanh) (options: sigmoid, tanh)
        """
        super(OutputLayer, self).__init__()

        self.embedding_size = embedding_size
        self.Linear = torch.nn.Linear(d_model, embedding_size, bias=True)

    def init_weights(self, initrange=0.1):
        self.Linear.bias.data.zero_()
        self.Linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, decoder_out):
        y = self.Linear(decoder_out)
        y = torch.reshape(y, (decoder_out.shape[0], decoder_out.shape[1], 3, self.embedding_size // 3))
        h_logits = y[:, :, 0, :]
        v_logits = y[:, :, 1, :]
        o_logits = y[:, :, 2, :]

        # h_logits = _h
        # v = torch.sigmoid(_v)
        # if self.offset_activation == "tanh":
        #     o = torch.tanh(_o) * 0.5
        # else:
        #     o = torch.sigmoid(_o) - 0.5

        return h_logits, v_logits, o_logits

# --------------------------------------------------------------------------------
# ------------         VARIAIONAL REPARAMETERIZE BLOCK       ---------------------
# --------------------------------------------------------------------------------
class reparameterize(torch.nn.Module):
    """
   :param input: (Tensor) Input tensor to REPARAMETERIZE [Nx32xd_model]
   :return: (Tensor) [B x D]
   """

    def __init__(self, max_len, d_model, latent_dim):
        super(reparameterize, self).__init__()

        self.fc_mu = torch.nn.Linear(int(max_len*d_model), latent_dim)
        self.fc_var = torch.nn.Linear(int(max_len*d_model), latent_dim)

    def forward(self, src):
        result = torch.flatten(src, start_dim=1)
        print(f"result shape: {result.shape}")
        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        print(f"mu shape: {mu.shape}")
        log_var = self.fc_var(result)
        print(f"log_var shape: {log_var.shape}")

        std = torch.exp(0.5 * log_var)
        print(f"std shape: {std.shape}")
        eps = torch.randn_like(std)
        print(f"eps shape: {eps.shape}")
        z = eps * std + mu
        print(f"z shape: {z.shape}")
        return mu, log_var, z

# --------------------------------------------------------------------------------
# ------------       RE-CONSTRUCTION DECODER INPUT             -------------------
# --------------------------------------------------------------------------------
class DecoderInput(torch.nn.Module):
    """
    reshape the input tensor to fix dimensions with decoder

   :param input: (Tensor) Input tensor distribution [Nx(latent_dim)]
   :return: (Tensor) [N x max_len x d_model]
   """

    def __init__(self, max_len, latent_dim, d_model):
        super(DecoderInput, self).__init__()

        self.max_len = max_len
        self.d_model = d_model

        self.updims = torch.nn.Linear(latent_dim, int(max_len * d_model))

    def forward(self, src):

        uptensor = self.updims(src)

        result = uptensor.view(-1, self.max_len, self.d_model)

        return result


class VAE_Decoder(torch.nn.Module):
    def __init__(self, latent_dim, d_model, num_decoder_layers, nhead, dim_feedforward,
                 output_max_len, output_embedding_size, dropout, o_activation):
        super(VAE_Decoder, self).__init__()

        assert o_activation in ["sigmoid", "tanh"]

        self.latent_dim = latent_dim
        self.d_model = d_model
        self.num_decoder_layers =    num_decoder_layers
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.output_max_len = output_max_len
        self.output_embedding_size = output_embedding_size
        self.dropout = dropout

        self.o_activation = torch.sigmoid if o_activation == "sigmoid" else torch.tanh

        self.DecoderInput = DecoderInput(
            max_len=self.output_max_len,
            latent_dim=self.latent_dim,
            d_model=self.d_model)

        self.Decoder = Encoder(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            num_encoder_layers=self.num_decoder_layers,
            dropout=self.dropout)

        self.OutputLayer = OutputLayer(
            embedding_size=self.output_embedding_size,
            d_model=self.d_model)

    def forward(self, latent_z):
        pre_out = self.DecoderInput(latent_z)
        decoder_ = self.Decoder(pre_out)
        h_logits, v_logits, o_logits = self.OutputLayer(decoder_)

        return h_logits, v_logits, o_logits

    def predict(self, latent_z, threshold=0.5, use_thres=True, use_pd=False):
        self.eval()
        with torch.no_grad():
            h_logits, v_logits, o_logits = self.forward(latent_z)
            h = get_hits_activation(h_logits, use_thres=use_thres, thres=threshold, use_pd=use_pd)
            v = torch.sigmoid(v_logits)

            if self.o_activation == "tanh":
                o = torch.tanh(o_logits) * 0.5
            else:
                o = torch.sigmoid(o_logits) - 0.5

        return h, v, o