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

    def __init__(self, embedding_size, d_model, dropout, max_len, n_token_types, token_type_loc):
        super(CustomEmbeddingLayer, self).__init__()
        self.token_embedding = torch.nn.Embedding(n_token_types, d_model, dtype=torch.float32)
        self.token_type_loc = token_type_loc
        self.Linear = torch.nn.Linear((embedding_size-1), d_model, bias=True)
        self.ReLU = torch.nn.ReLU()
        self.PositionalEncoding = PositionalEncoding((d_model*2), max_len, dropout)

    def init_weights(self, initrange=0.1):
        self.Linear.bias.data.zero_()
        self.Linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, token):
        # [max_len, d_model]
        token_type = token[:, self.token_type_loc].long()
        token_type_embedding = self.token_embedding(token_type)
        hvo_projection = self.Linear(token[:, (self.token_type_loc + 1):])
        hvo_projection = self.ReLU(hvo_projection)
        x = torch.cat((token_type_embedding, hvo_projection), 1)
        out = self.PositionalEncoding(x)

        return out


class InputLayer(torch.nn.Module):
    def __init__(self, embedding_size, d_model, dropout, max_len, n_token_types, token_type_loc):
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


    embed_size = 10  # columns
    d_model = 32  # columns after processing
    max_len = 32

    embedding = CustomEmbeddingLayer(embedding_size=embed_size,
                                     d_model=d_model,
                                     dropout=0.1,
                                     max_len=max_len,
                                     n_token_types=5,
                                     token_type_loc=0)

    # for testing purposes, the input tensor needs to be shape:
    # [max_len, embed_size]
    # embed size = hv size + 1(token)

    input_hvo= torch.rand(max_len, (embed_size-1))
    input_token = torch.full((max_len, 1), 3, dtype=torch.long)
    data = torch.cat((input_token, input_hvo), dim=1)
    print(f"Data shape: {data.shape}")
    output = embedding(data)
    print(f"Data shape after input layer: {output.shape}")

    """
    
    Definitions:
    Embedding size
    (input) The # of columns in a single row - in this case, how many voices in the HVO + token type
    
    d_model
    The # of columns AFTER going through linear layer, i.e. layer layer will 
    expand 9 columns to 128 for each row
    
    
    
    Discoveries:
    - Linear will change the second dimension of tensor, but leave first untouched.
    I.e.
    Input: (32, 9)
    Linear: (9, 128)
    Output: (32, 128)
    
    
    
    - The 2nd dimension (columns) of the -token embedding- and -hvo linear layer- (d_model) 
    must be *identical* (in order to pass to cat)..
    
    - ..therefor, the output dimension variable of linear layer for hvo must == d_model
    
    - First dimension of cat will be the first dimensions of token + hvo added together. 
    
    - input args to Positional Encoder must be padded, and the reverse of the concat
    i.e. if concat is [32, 8] then pos encoder must be [8, 32]
    
    - input data will be in batches, so the shape will be:
    (batch size, max_sequence_length, input_dim)
    
    
    
    
    """



    # linear = torch.nn.Linear(20, 60)
    # tensor = linear(tensor)
    # print(tensor.shape)

