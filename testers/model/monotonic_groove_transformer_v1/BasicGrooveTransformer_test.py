import torch

if __name__ == '__main__':
    # Instantiating a model
    # 2.i BasicGrooveTransformer.GrooveTransformer
    params = {
        "d_model": 128,
        "nhead": 4,
        "dim_forward": 256,
        "dropout": 0.1,
        "num_encoder_layers": 6,
        "num_decoder_layers": 6,
        "max_len": 32,
        "N": 64,             # batch size
        "embedding_size_src": 16,   # input dimensionality at each timestep
        "embedding_size_tgt": 27    # output dimensionality at each timestep
    }

    # test transformer
    from model import GrooveTransformer

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    TM = GrooveTransformer(params["d_model"], params["embedding_size_src"], params["embedding_size_tgt"],
                           params["nhead"], params["dim_forward"], params["dropout"],
                           params["num_encoder_layers"], params["num_decoder_layers"], params["max_len"], device)


    # feed forward
    src = torch.rand(params["N"], params["max_len"], params["embedding_size_src"])
    tgt = torch.rand(params["N"], params["max_len"], params["embedding_size_tgt"])
    h, v, o = TM(src, tgt)
    print(h.shape, v.shape, o.shape)
    print(h[0, 0, :], v[0, 0, :], o[0, 0, :])

    # test predict
    print("pred")
    pred_h,pred_v,pred_o = TM.predict(src)
    print(pred_h.shape)

    # 2.i BasicGrooveTransformer.GrooveTransformer
    import torch

    params = {
        'd_model': 512,
        'embedding_size_src': 27,
        'embedding_size_tgt': 27,
        'nhead': 1,
        'dim_feedforward': 64,
        'dropout': 0.25542373735391866,
        'num_encoder_layers': 10,
        'max_len': 32,
        'device': 'gpu'
    }

    from model import GrooveTransformerEncoder

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    TEM = GrooveTransformerEncoder(params["d_model"], params["embedding_size_src"], params["embedding_size_tgt"],
                                   params["nhead"], params["dim_feedforward"], params["dropout"],
                                   params["num_encoder_layers"], params["max_len"], device)

    src = torch.rand(params["N"], params["max_len"], params["embedding_size"])

    mem_h, mem_v, mem_o = TEM(src)
    print(mem_h.shape, mem_v.shape, mem_o.shape)


    pred_h,pred_v,pred_o  = TEM.predict(src)
    print(pred_h.shape)

    TEM.save("model/misc/rand_model.pth")

    # test input layer
    # from model.src.BasicGrooveTransformer import InputLayer
    #
    # src = torch.rand(params["N"], params["max_len"], params["embedding_size"])
    # print(src.shape)
    # InputLayer = InputLayer(params["embedding_size_src"], params["d_model"], params["dropout"], params["max_len"])
    # y = InputLayer(src)
    # print(y.shape, y)
    #
    # # test output layer
    # from model.src.BasicGrooveTransformer import OutputLayer
    #
    # OutputLayer = OutputLayer(params["embedding_size_tgt"], params["d_model"])
    # h, v, o = OutputLayer(y)
    # print(h, v, o)
    #TEM()