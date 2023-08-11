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

def calculate_continuous_value_weights(values, num_bins=10):
    """
    Given a vector of continuous values ranged 0.0 - 1.0, round them to the nearest bin and count the number of
    occurrences in each bin. Also compute the weights for each bin.

    @param values: (Tensor) vector of continuous values
    @param num_bins: Number of bins to divide into
    @return: (Tensor) counts of occurrences in each bin, (Tensor) weights for each bin
    """
    # Ensure values are a PyTorch tensor
    if not isinstance(values, torch.Tensor):
        values = torch.tensor(values)

    # Round values to the nearest bin
    rounded_values = (values * num_bins).round()
    rounded_values = torch.clamp(rounded_values, max=num_bins - 1)

    # Count the number of occurrences in each bin
    bin_counts = torch.bincount(rounded_values.int(), minlength=num_bins)

    # Calculate weights
    inv_counts = 1.0 / (bin_counts.float() + 1e-9)  # adding a small value to prevent division by zero
    weights = inv_counts / inv_counts.sum()

    return weights


def calculate_genre_weights(genres):
    """
    Calculate the inverse weight distribution of genres, to balance the cross entropy loss functions
    @param genres: List of np arrays with one-hot encoding of genres
    @return: (tensor) the inverse distribution weight tensor
    """
    genres = np.stack(genres)  # Convert list of arrays to 2D array
    genre_counts = genres.sum(axis=0)  # Count occurrence of each genre
    total_count = len(genres)  # Total number of samples
    weights = total_count / genre_counts  # Calculate weights

    return torch.tensor(weights, dtype=torch.float)

