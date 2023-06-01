import torch
import numpy as np
import wandb
from hvo_sequence.tokenization import *
from hvo_sequence import HVO_Sequence
from hvo_sequence import ROLAND_REDUCED_MAPPING
from data.src.dataLoaders import load_gmd_hvo_sequences, MonotonicGrooveTokenizedDataset
from helpers.BaseTokenize.modelLoader import load_tokenized_model
import os


class TokenizerEvaluator:
    def __init__(self, model, hvo_sequences, delta_grains=[1, 2, 5, 10, 15, 20], tpb=96):

        self.model = model
        self.vocab = {'PAD': 0,
                      'beat': 1,
                      'delta_1': 2,
                      'delta_10': 3,
                      'delta_15': 4,
                      'delta_2': 5,
                      'delta_20': 6,
                      'delta_5': 7,
                      'hit': 8}
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}

        self.in_hvo_seq = list()
        self.out_hvo_seq = list()
        self.in_tokens = list()
        self.out_tokens = list()
        self.out_hits = list()



        for hvo_seq in hvo_sequences:

            tokenised_array = tokenizeConsistentSequence(hvo_seq, delta_grains, tpb)
            tokenised_array = flattenTokenizedSequence(tokenised_array, num_voices=9)
            tokens = np.zeros(400)
            hv = np.zeros((400, 18))

            for idx, token in enumerate(tokenised_array):
                tokens[idx] = np.array([self.vocab[token[0]]])
                hv[idx] = token[1][0]

            self.in_tokens.append(tokens[:len(tokenised_array)])

            tokens = torch.unsqueeze(torch.from_numpy(tokens).long(), dim=0)
            hv = torch.unsqueeze(torch.from_numpy(hv).float(), dim=0)

            token_type, h, v = model.predict(input_tokens=tokens, input_hv=hv)
            token_type = torch.squeeze(token_type, dim=0)
            h = torch.squeeze(h, dim=0)
            v = torch.squeeze(v, dim=0)


            np_tokens = token_type.numpy()
            np_hits = h.numpy()

            self.out_tokens.append(np_tokens)
            self.out_hits.append(np_hits)

            out_tokenized = convert_model_output_to_tokenized_sequence(token_type, h, v, reverse_vocab=self.reverse_vocab)
            out_hvo = convert_tokenized_sequence_to_hvo_array(out_tokenized)
            out_seq = HVO_Sequence(beat_division_factors=[96], drum_mapping=ROLAND_REDUCED_MAPPING)
            out_seq.add_time_signature(hvo_seq.time_signatures[0].time_step, hvo_seq.time_signatures[0].numerator,
                                       hvo_seq.time_signatures[0].denominator)
            out_seq.add_tempo(time_step=0, qpm=120)
            out_seq.hvo = out_hvo

            self.in_hvo_seq.append(hvo_seq)
            self.out_hvo_seq.append(out_seq)


    def __getitem__(self, idx):
        return self.in_hvo_seq[idx], self.out_hvo_seq[idx]
    def __len__(self):
        return len(self.in_hvo_seq)

    def get_tokens_and_hits(self, idx):
        return self.in_tokens[idx], self.out_tokens[idx], self.out_hits[idx]

