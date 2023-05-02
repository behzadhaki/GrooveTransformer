import torch
import os
import numpy as np
from functools import partial

from model.BaseTokenize.shared_model_components import *
from data.src.dataLoaders import MonotonicGrooveTokenizedDataset, load_gmd_hvo_sequences
from torch.utils.data import DataLoader




if __name__ == "__main__":

    os.chdir("../../")

    d_model = 32
    n_voices = 9
    embed_size = n_voices * 2
    max_len = 500

    # load our data

    """
    When loading the dataset class, it will load GMD data as HVOs, tokenize it,
    convert the token types to integers (with a retrievable dictionary), and return in 
    a format designed for pytorch dataloader (list of tuples of tensors)
    """
    subset = load_gmd_hvo_sequences(dataset_setting_json_path="data/dataset_json_settings/BeatsAndFills_gmd_96.json",
                                    subset_tag="train",
                                    force_regenerate=False)
    tokenized_dataset = MonotonicGrooveTokenizedDataset(subset=subset)


    dictionary = tokenized_dataset.get_vocab_dictionary()
    print(dictionary)
    n_token_types = len(dictionary)
    print(f"num token types: {n_token_types}")
    data_loader = DataLoader(tokenized_dataset, batch_size=16, shuffle=True)
    # Load model componenets individually for testing
    embedding = InputLayer(embedding_size=embed_size,
                           d_model=d_model,
                           n_token_types=n_token_types,
                           token_type_loc=0,
                           token_embedding_ratio=0.8,
                           padding_idx=0)

    encoder = Encoder(d_model=d_model,
                      nhead=4,
                      dim_feedforward=128,
                      dropout=0.1,
                      num_encoder_layers=3,
                      max_len=max_len)

    outputlayer = OutputLayer(n_token_types=n_token_types,
                              n_voices=n_voices,
                              d_model=d_model)


    # Training script

    for data in data_loader:
        single_batch = data
        break

    idx, in_token, in_hv, out_token, out_hv, masks = single_batch

    x = embedding(in_token, in_hv)
    print(f"after embedding: {x.shape}")
    x = encoder(x, masks)
    print(f"after encoder: {x.shape}")

    token_type_logits, hits, velocities = outputlayer.decode(x)
    print(f"token type: {token_type_logits.shape}")
    print(f"h: {hits.shape}")
    print(f"v: {velocities.shape}")

    print(token_type_logits[0, :5, :])
    print(hits[0, :5, :])
    print(velocities[0, :5, :])

    final = torch.concat((token_type_logits, hits, velocities), dim=2)
    print(final.shape)

    # sliced = data[:, :, 1:]
    #
    # print(sliced.shape)
    # print(sliced.dtype)
    #
    # linear_layer = torch.nn.Linear(18, 16, bias=True)
    #
    # x = linear_layer(sliced)
    # print(f"output: {x.shape}")




    # epochs = 3
    #
    # for idx, (input_batch, output_batch) in enumerate(data_loader):
    #     if idx >= epochs:
    #         break
    #
    #     x = embedding(input_batch)
    #
    #     print(x.shape)



