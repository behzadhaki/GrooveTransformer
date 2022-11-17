import torch

if __name__ == '__main__':

    # 2.i BasicGrooveTransformer.GrooveTransformer
    params = {
        'd_model_enc': 128,
        'd_model_dec': 512,
        'embedding_size_src': 9,
        'embedding_size_tgt': 27,
        'nhead_enc': 2,
        'nhead_dec': 4,
        'dim_feedforward_enc': 16,
        'dim_feedforward_dec': 32,
        'num_encoder_layers': 3,
        'num_decoder_layers': 5,
        'dropout': 0.1,
        'latent_dim': 32,
        'max_len_enc': 32,
        'max_len_dec': 32,
        'device': 'cpu',
        'o_activation': 'sigmoid',
        'batch_size': 8 }

    # test transformer

    from model import GrooveTransformerEncoderVAE
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config.update({'device': device})

    TM = GrooveTransformer(config)

    # feed forward
    src = torch.rand(params["batch_size"], params["max_len_enc"], params["embedding_size_src"])
    tgt = torch.rand(params["batch_size"], params["nhead_dec"], params["embedding_size_tgt"])

    (h_logits, v_logits, o_logits), mu, log_var, latent_z = TM.forward(src, tgt)
    print(h_logits.shape, v_logits.shape, o_logits.shape)
    print(h_logits[0, 0, :], v_logits[0, 0, :], o_logits[0, 0, :])

    # test predict
    print("pred")
    (h_pred, v_pred, o:pred), mu, log_var, latent_z = TM.predict(src)
    print(h.shape)
