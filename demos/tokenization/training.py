import torch
import os
import numpy as np
from functools import partial

from model.BaseTokenize.TokenizationModelTester import *
from data.src.dataLoaders import MonotonicGrooveTokenizedDataset
from torch.utils.data import DataLoader
from data.src.dataLoaders import custom_collate_fn



if __name__ == "__main__":

    os.chdir("../../")

    d_model = 32  # columns after processing
    max_len = 4
    n_voices = 9
    embed_size = (n_voices * 2) + 1  # columns
    max_len = 1000



    # load our data

    tokenized_dataset = MonotonicGrooveTokenizedDataset(
        dataset_setting_json_path="data/dataset_json_settings/BeatsAndFills_gmd_96.json",
        subset_tag="test")


    dictionary = tokenized_dataset.get_vocab_dictionary()
    print(dictionary)
    n_token_types = len(dictionary) + 1
    padding_token = float(n_token_types)
    print(f"num token types: {n_token_types}")

    collate_with_args = partial(custom_collate_fn, max_len=max_len, padding_token=padding_token, num_voices=n_voices)

    data_loader = DataLoader(tokenized_dataset, batch_size=16, shuffle=True, collate_fn=collate_with_args)

    #for batch_idx, batch in enumerate(data_loader):
        # print(f"\nBatch {batch_idx + 1}:")
        # print(f"Batch shape:")
        # print(batch[0].shape)
        # if batch_idx >= 1:
        #     break


    # Load our model componenets individually

    embedding = EmbeddingLayer(embedding_size=embed_size,
                               d_model=d_model,
                               n_token_types=20,
                               token_type_loc=0,
                               padding_idx=int(padding_token))

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

    data = single_batch[0]
    print(f"input dim: {data.shape}")

    x = embedding(data)
    print(f"after embedding: {x.shape}")

    x = encoder(x)
    print(f"after encoder: {x.shape}")

    token_type, h_logits, v_logits = outputlayer.decode(x)

    token_type_logits, h_logits, v_logits = outputlayer(x)
    print(f"token type: {token_type_logits.shape}")
    print(f"h: {h_logits.shape}")
    print(f"v: {v_logits.shape}")

    print(token_type_logits[0, :5, :])

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



