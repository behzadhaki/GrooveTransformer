import torch
import math
import numpy as np

class CustomToken:
    def __init__(self, token_type, hvo):
        self.token_type = token_type
        self.hvo = hvo

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

class Encoder(torch.nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward, dropout, num_encoder_layers):
        super(Encoder, self).__init__()
        norm_encoder = torch.nn.LayerNorm(d_model)
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.Encoder = torch.nn.TransformerEncoder(encoder_layer, num_encoder_layers, norm_encoder)

    def forward(self, src):
        src = src.permute(1, 0, 2)  # 32xNxd_model
        out = self.Encoder(src)  # 32xNxd_model ()
        out = out.permute(1, 0, 2)  # Nx32xd_model
        return out

class CustomEmbeddingLayer(torch.nn.Module):

    def __init__(self, token_type_dict, token_type_vocab_size, token_type_embedding_dim, hvo_embedding_size, hvo_projection_size, max_len, dropout):
        super(CustomEmbeddingLayer, self).__init__()
        self.token_type_dict = token_type_dict
        self.token_embedding = torch.nn.Embedding(token_type_vocab_size, token_type_embedding_dim)
        self.Linear = torch.nn.Linear(hvo_embedding_size, hvo_projection_size)
        self.ReLU = torch.nn.ReLU()
        self.d_model = hvo_projection_size
        self.PositionalEncoding = PositionalEncoding(self.d_model, max_len, dropout)

    def forward(self, token):
        token_type_tensor = torch.tensor([self.token_type_dict[token.token_type]], dtype=torch.long)
        token_type_embedding = self.token_embedding(token_type_tensor)
        print(f"token type embed size: {token_type_embedding.shape}")
        hvo_projection = self.Linear(torch.from_numpy(token.hvo).float())
        hvo_projection = self.ReLU(hvo_projection)
        print(f"hvo projection size: {hvo_projection.shape}")
        embedding = torch.cat((token_type_embedding, hvo_projection), 0)
        print(f"Concat shape: {embedding.shape}")
        out = self.PositionalEncoding(embedding)


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


if __name__ == "__main__":
    tokens = [CustomToken('measure', np.random.rand(9,3))]


    custom_dict = {'measure': 0, 'beat': 1, 'delta_30': 2}

    token = tokens[0]
    idx_token = np.append(custom_dict[token.token_type], token.hvo)
    # print(idx_token)
    # print(idx_token.shape)

    embedding = CustomEmbeddingLayer(token_type_dict=custom_dict,
                                     token_type_vocab_size=3,
                                     token_type_embedding_dim=16,
                                     hvo_embedding_size=3,
                                     hvo_projection_size=16,
                                     max_len=16,
                                     dropout=0)

    x = embedding.forward(token)

    # randn = torch.from_numpy(np.random.rand(128,20)).float()
    # linear = torch.nn.Linear(20, 30)
    # y = linear(randn)
    # torchrandn = torch.randn(128,20)
    # x = linear(torchrandn)
