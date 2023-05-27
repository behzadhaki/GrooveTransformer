import numpy as np


def calculate_density(hits, num_voices=9, seq_length=32):
    return np.sum(hits[:, :num_voices]) / (seq_length * num_voices)


