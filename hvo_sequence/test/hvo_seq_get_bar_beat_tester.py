from hvo_sequence.hvo_seq import HVO_Sequence
from hvo_sequence.drum_mappings import ROLAND_REDUCED_MAPPING

import numpy as np


if __name__ == "__main__":

    # Create an instance of a HVO_Sequence
    hvo_seq = HVO_Sequence(drum_mapping=ROLAND_REDUCED_MAPPING)

    # Add a time_signature
    hvo_seq.add_time_signature(0, 4, 4, [3, 4])

    # Add two tempos
    hvo_seq.add_tempo(0, 50)

    hvo_seq.add_tempo(12, 20)  # Tempo Change at the beginning of second bar

    # Create a random hvo
    hits = np.random.randint(0, 2, (36, 9))
    vels = hits * np.random.rand(36, 9)
    offs = hits * (np.random.rand(36, 9) - 0.5)

    # Add hvo score to hvo_seq instance
    hvo_seq.hvo = np.concatenate((hits, vels, offs), axis=1)
    # hvo_seq.to_html_plot(show_figure=True)
    print(hvo_seq.get_bar_beat_hvo(hvo_str="hvo"))

    print(hvo_seq.get_offsets_in_ms())