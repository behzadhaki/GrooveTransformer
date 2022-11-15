model_params = {
    'robust_sweep_29':
        {
            'd_model': 512,
            'embedding_size_src': 27,
            'embedding_size_tgt': 27,
            'nhead': 1,
            'dim_feedforward': 64,
            'dropout': 0.25542373735391866,
            'num_encoder_layers': 10,
            'max_len': 32,
            'device': 'gpu'},

    'colorful_sweep_41':
        {
            'd_model': 64,
            'embedding_size_src': 27,
            'embedding_size_tgt': 27,
            'nhead': 2,
            'dim_feedforward': 16,
            'dropout': 0.1410222621621131,
            'num_encoder_layers': 11,
            'max_len': 32,
            'device': 'gpu'
        },
    'misunderstood_bush_246':
        {
            # This is the Light version in MonotonicGrooveTransformer camomile vst
            # https://wandb.ai/mmil_tap2drum/transformer_groove_tap2drum/runs/vxuuth1y
            # 'pyModelPath': "pyTorch_models/misunderstood_bush_246-epoch_26.Model",
            'd_model': 128,
            'dim_feedforward': 128,
            'dropout': 0.1038,
            'nhead': 4,
            'num_encoder_layers': 11,
            'embedding_size_src': 27,
            'embedding_size_tgt': 27,
            'max_len': 32,
            'device': 'cpu'
        },
    'rosy_durian_248':
        {
            # This is the Heavy version in MonotonicGrooveTransformer camomile vst
            # https://wandb.ai/mmil_tap2drum/transformer_groove_tap2drum/runs/2cgu9h6u
            # 'pyModelPath': "pyTorch_models/rosy_durian_248-epoch_26.Model",
            'd_model': 512,
            'dim_feedforward': 16,
            'dropout': 0.1093,
            'nhead': 4,
            'num_encoder_layers': 6,
            'embedding_size_src': 27,
            'embedding_size_tgt': 27,
            'max_len': 32,
            'device': 'cpu'
        },
    'hopeful_gorge_252':
        {   # https://wandb.ai/mmil_tap2drum/transformer_groove_tap2drum/runs/v7p0se6e
            # 'pyModelPath': "pyTorch_models/hopeful_gorge_252-epoch_90.Model",
            'd_model': 512,
            'dim_feedforward': 64,
            'dropout': 0.1093,
            'nhead': 4,
            'num_encoder_layers': 8,
            'embedding_size_src': 27,
            'embedding_size_tgt': 27,
            'max_len': 32,
            'device': 'cpu'
        },
    'solar_shadow_247':
        {
            # https://wandb.ai/mmil_tap2drum/transformer_groove_tap2drum/runs/35c9fysk
            #'pyModelPath': "pyTorch_models/solar_shadow_247-epoch_41.Model",
            'd_model': 128,
            'dim_feedforward': 16,
            'dropout': 0.159,
            'nhead': 1,
            'num_encoder_layers': 7,
            'embedding_size_src': 27,
            'embedding_size_tgt': 27,
            'max_len': 32,
            'device': 'cpu'
        }
}

# https://wandb.ai/mmil_tap2drum/transformer_groove_tap2drum/runs/3vqjemc0?workspace=user-anonmmi
# https://wandb.ai/mmil_tap2drum/transformer_groove_tap2drum/runs/0o2kg42r?workspace=user-anonmmi