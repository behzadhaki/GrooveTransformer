import torch
import math
import numpy as np
from model import get_hits_activation

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

    def __init__(self, d_model, nhead, dim_feedforward, dropout, num_encoder_layers, max_len):
        super(Encoder, self).__init__()
        norm_encoder = torch.nn.LayerNorm(d_model)
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.Encoder = torch.nn.TransformerEncoder(encoder_layer, num_encoder_layers, norm_encoder)
        self.PositionalEncoder = PositionalEncoding(d_model, max_len, dropout=dropout)

    def forward(self, src):
        src = self.PositionalEncoder(src)
        src = src.permute(1, 0, 2)  # 32xNxd_model
        out = self.Encoder(src)  # 32xNxd_model ()
        out = out.permute(1, 0, 2)  # Nx32xd_model
        return out

class EmbeddingLayer(torch.nn.Module):

    def __init__(self, embedding_size, d_model, dropout, max_len, n_token_types, token_type_loc):
        super(EmbeddingLayer, self).__init__()
        assert d_model % 2 == 0, "d_model must be divisible by 2"

        self.token_embedding = torch.nn.Embedding(n_token_types, d_model//2, dtype=torch.float32)
        self.token_type_loc = token_type_loc
        self.Linear = torch.nn.Linear((embedding_size-1), d_model//2, bias=True)
        self.ReLU = torch.nn.ReLU()

    def init_weights(self, initrange=0.1):
        self.Linear.bias.data.zero_()
        self.Linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, token):
        # [max_len, d_model]
        token_type = token[:, self.token_type_loc].long()
        token_type_embedding = self.token_embedding(token_type)
        hvo_projection = self.Linear(token[:, (self.token_type_loc + 1):])
        hvo_projection = self.ReLU(hvo_projection)
        out = torch.cat((token_type_embedding, hvo_projection), 1)
        #out = self.PositionalEncoding(x) #  move this to encoder

        return out

class OutputLayer(torch.nn.Module):
    def __init__(self, n_token_types, n_voices, d_model):
        super(OutputLayer, self).__init__()

        self.n_token_types = n_token_types
        self.n_voices = n_voices
        self.d_model = d_model

        self.tokenLinear = torch.nn.Linear(d_model, n_token_types, bias=True)
        self.hitsLinear = torch.nn.Linear(d_model, n_voices, bias=True)
        self.velocitiesLinear = torch.nn.Linear(d_model, n_voices, bias=True)

    def init_weights(self, initrange=0.1):
        self.Linear.bias.data.zero_()
        self.Linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, decoder_out):
        """
        Predicts the *pre-activation* logits
        @param decoder_out:
        @return:
        """

        token_type_logits = self.tokenLinear(decoder_out)
        h_logits = self.hitsLinear(decoder_out)
        v_logits = self.velocitiesLinear(decoder_out)

        return token_type_logits, h_logits, v_logits

    def decode(self, decoder_out, threshold=0.5, use_thres=True, use_pd=False):

        self.eval()
        with torch.no_grad():
            token_type_logits, h_logits, v_logits = self.forward(decoder_out)
            token_type = torch.nn.Softmax(token_type_logits)
            h = get_hits_activation(h_logits, use_thres=use_thres, thres=threshold, use_pd=use_pd)
            v = torch.sigmoid(v_logits)

        return token_type, h, v




if __name__ == "__main__":



    d_model = 32  # columns after processing
    max_len = 4
    n_token_types = 5
    n_voices = 7
    embed_size = (n_voices * 2) + 1  # columns

    embedding = EmbeddingLayer(embedding_size=embed_size,
                               d_model=d_model,
                               dropout=0.1,
                               max_len=max_len,
                               n_token_types=n_token_types,
                               token_type_loc=0)

    encoder = Encoder(d_model=d_model,
                      nhead=4,
                      dim_feedforward=128,
                      dropout=0.1,
                      num_encoder_layers=3,
                      max_len=max_len)

    outputlayer = OutputLayer(n_token_types=n_token_types,
                              n_voices=n_voices,
                              d_model=d_model)

    # for testing purposes, the input tensor needs to be shape:
    # [max_len, embed_size]
    # embed size = hv size + 1(token)

    input_hvo= torch.rand(max_len, (embed_size-1))
    input_token = torch.full((max_len, 1), 3, dtype=torch.long)
    data = torch.cat((input_token, input_hvo), dim=1)
    print(f"Data shape: {data.shape}")

    input = embedding(data)
    print(f"embedding layer output: {input.shape}")

    encoder_output = encoder(input)
    print(f"encoder output: {encoder_output.shape}")

    token, h, v = outputlayer(encoder_output)
    print(token)


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

