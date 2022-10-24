import torch

from model.src.shared_model_components_VAE import *
from model.src.utils import *

class GrooveTransformerEncoderVAE(torch.nn.Module):
    """
    An encoder-encoder VAE transformer
    """
    def __init__(self, d_model_enc, d_model_dec, embedding_size_src, embedding_size_tgt,
                 nhead_enc, nhead_dec, dim_feedforward, dropout, num_encoder_layers, latent_dim,
                 num_decoder_layers, max_len, device):
        super(GrooveTransformerEncoderVAE, self).__init__()

        #input/output dims
        self.d_model_enc = d_model_enc
        self.d_model_dec = d_model_dec
        self.embedding_size_src = embedding_size_src
        self.embedding_size_tgt = embedding_size_tgt
        #trasnformers dims and params
        self.nhead_enc = nhead_enc
        self.nhead_dec = nhead_dec
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.max_len = max_len
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.latent_dim = latent_dim
        self.device = device

        self.InputLayerEncoder = InputLayer(embedding_size_src, d_model_enc, dropout, max_len)
        self.Encoder = Encoder(d_model_enc, nhead_enc, dim_feedforward, dropout, num_encoder_layers)
        self.vaeEncode = reparameterize(max_len, d_model_enc, latent_dim)
        self.deco_in = deco_imput(max_len, d_model_dec, latent_dim)
        self.Decoder = Encoder(d_model_dec, nhead_dec, dim_feedforward, dropout, num_decoder_layers)

        self.OutputLayer = OutputLayer(embedding_size_tgt, d_model_dec)

        self.InputLayerEncoder.init_weights()
        self.OutputLayer.init_weights()

    def forward(self, src):
        # model Nx32xembedding_size_src
        x = self.InputLayerEncoder(src)  # Nx32xd_model
        memory = self.Encoder(x)  # Nx32xd_model
        mu,log_var,Z = self.vaeEncode(memory)

        pre_out = self.deco_in(Z)
        decoder_ = self.Decoder(pre_out)
        out = self.OutputLayer(
            decoder_)  # (Nx32xembedding_size_tgt/3,Nx32xembedding_size_tgt/3,Nx32xembedding_size_tgt/3)


        return out, mu, log_var

    def predict(self, src, use_thres=True, thres=0.5, use_pd=False):
        self.eval()
        with torch.no_grad():
            _h, v, o = self.forward(
                src)  # Nx32xembedding_size_src/3,Nx32xembedding_size_src/3,Nx32xembedding_size_src/3

            h = get_hits_activation(_h, use_thres=use_thres, thres=thres, use_pd=use_pd)

        return h, v, o
