import torch
from src.utils import PositionalEncoding

# --------------------------------------------------------------------------------
# ------------                  ENCODER BLOCK                ---------------------
# --------------------------------------------------------------------------------
class Encoder(torch.nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward, dropout, num_encoder_layers):
        super(Encoder, self).__init__()
        norm_encoder = torch.nn.LayerNorm(d_model)
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.Encoder = torch.nn.TransformerEncoder(encoder_layer, num_encoder_layers, norm_encoder)

    def forward(self, src):
        src = src.permute(1, 0, 2)  # 32xNxd_model
        out = self.Encoder(src)  # 32xNxd_model
        out = out.permute(1, 0, 2)  # Nx32xd_model
        return out

# --------------------------------------------------------------------------------
# ------------                  DECODER BLOCK                ---------------------
# --------------------------------------------------------------------------------
class Decoder(torch.nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward, dropout, num_decoder_layers):
        super(Decoder, self).__init__()
        norm_decoder = torch.nn.LayerNorm(d_model)
        decoder_layer = torch.nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.Decoder = torch.nn.TransformerDecoder(decoder_layer, num_decoder_layers, norm_decoder)

    def forward(self, tgt, memory, tgt_mask):
        # tgt    Nx32xd_model
        # memory Nx32xd_model

        tgt = tgt.permute(1, 0, 2)  # 32xNxd_model
        memory = memory.permute(1, 0, 2)  # 32xNxd_model

        out = self.Decoder(tgt, memory, tgt_mask)  # 32xNxd_model

        out = out.permute(1, 0, 2)  # Nx32xd_model

        return out

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
        super(OutputLayer, self).__init__()

        self.embedding_size = embedding_size
        self.Linear = torch.nn.Linear(d_model, embedding_size, bias=True)

    def init_weights(self, initrange=0.1):
        self.Linear.bias.data.zero_()
        self.Linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, decoder_out):
        y = self.Linear(decoder_out)
        y = torch.reshape(y, (decoder_out.shape[0], decoder_out.shape[1], 3, self.embedding_size // 3))

        _h = y[:, :, 0, :]
        _v = y[:, :, 1, :]
        _o = y[:, :, 2, :]

        h = _h
        v = torch.sigmoid(_v)
        o = torch.tanh(_o) * 0.5

        return h, v, o

