import numpy as np
import torch

def calculate_density(hits, num_voices=9, seq_length=32):
    return np.sum(hits[:, :num_voices]) / (seq_length * num_voices)


def calculate_intensity(velocities):
    """Return the average of elements in a numpy array that are greater than zero.

    Args:
    velocities (np.ndarray): Input numpy array.

    Returns:
    float: The average of all elements greater than zero. Returns 0.0 if no elements are greater than zero.
    """
    if isinstance(velocities, torch.Tensor):
        velocities = velocities.detach().cpu().numpy()
    active_hits = velocities[velocities > 0]
    if active_hits.size > 0:  # Check if there are any elements greater than zero
        return active_hits.mean()
    else:
        return 0.0

