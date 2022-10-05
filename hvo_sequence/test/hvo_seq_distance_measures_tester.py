from hvo_sequence.hvo_seq import HVO_Sequence
from hvo_sequence.drum_mappings import ROLAND_REDUCED_MAPPING, Groove_Toolbox_5Part_keymap, Groove_Toolbox_3Part_keymap

import numpy as np

from tqdm import tqdm

if __name__ == "__main__":

    # Create an instance of a HVO_Sequence
    hvo_seq = HVO_Sequence(drum_mapping=ROLAND_REDUCED_MAPPING)
    hvo_seq_b = HVO_Sequence(drum_mapping=ROLAND_REDUCED_MAPPING)

    # Add a time_signature
    hvo_seq.add_time_signature(0, 4, 4, [4])
    hvo_seq_b.add_time_signature(0, 4, 4, [4])

    # Add two tempos
    hvo_seq.add_tempo(0, 50)
    hvo_seq_b.add_tempo(0, 50)

    # hvo_seq.add_tempo(12, 20)  # Tempo Change at the beginning of second bar

    n_steps = 32
    n_voices = 9

    # Create a random hvo
    hits = np.random.randint(0, 2, (n_steps, n_voices))
    # vels = hits * np.random.rand(n_steps, n_voices)
    vels = hits * np.random.rand(n_steps, n_voices)
    offs = hits * (np.random.rand(n_steps, n_voices) - 0.5)

    vels_b = hits * np.random.rand(n_steps, n_voices)
    offs_b = hits * (np.random.rand(n_steps, n_voices) - 0.5)

    # Add hvo score to hvo_seq instance
    hvo_seq.hvo = np.concatenate((hits, vels, offs), axis=1)
    hvo_seq_b.hvo = np.concatenate((hits, vels_b, offs_b), axis=1)

    # Calculate distance metrics
    for i in tqdm(range(2048*2)):
        hvo_seq.calculate_all_distances_with(hvo_seq_b)

    print(hvo_seq.calculate_all_distances_with(hvo_seq_b))
    print(hvo_seq.calculate_all_distances_with(hvo_seq))

hvo_seq.to_html_plot(show_figure=True)

hits = hvo_seq.hvo[:, :9]
vels = hvo_seq.hvo[:, 9:18]
offs = hvo_seq.hvo[:, 18:]

np.argwhere(hits>0) - np.argwhere(vels!=0)