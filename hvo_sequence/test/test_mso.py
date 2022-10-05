from hvo_sequence.hvo_seq import HVO_Sequence, empty_like
from hvo_sequence.drum_mappings import ROLAND_REDUCED_MAPPING, Groove_Toolbox_5Part_keymap, GM1_FULL_MAP

import numpy as np


if __name__ == "__main__":

    # Create an instance of a HVO_Sequence
    hvo_seq = HVO_Sequence(drum_mapping=ROLAND_REDUCED_MAPPING)

    # Add two time_signatures
    hvo_seq.add_time_signature(0, 4, 4, [4])
    # hvo_seq.add_time_signature(13, 6, 8, [3,2])

    # Add two tempos
    hvo_seq.add_tempo(0, 50)

    #hvo_seq.add_tempo(12, 20)  # Tempo Change at the beginning of second bar

    # Create a random hvo
    hits = np.random.randint(0, 2, (16, 9))
    vels = hits * np.random.rand(16, 9)
    offs = hits * (np.random.rand(16, 9) -0.5)

    #hvo_seq.get('0')
    #print(hvo_seq.hvo)

    # Add hvo score to hvo_seq instance
    hvo_bar = np.concatenate((hits, vels, offs), axis=1)
    hvo_seq.hvo = np.concatenate((hvo_bar, hvo_bar), axis=0)

    # Reset voices
    hvo_reset, hvo_out_voices = hvo_seq.reset_voices([0])
    print(hvo_seq.hvo[:10,0],hvo_seq.hvo[:10,9],hvo_seq.hvo[:10,2*9])
    print(hvo_reset.hvo[:10,0],hvo_reset.hvo[:10,9],hvo_reset.hvo[:10,2*9])
    print(hvo_out_voices.hvo[:10,0],hvo_out_voices.hvo[:10,9],hvo_out_voices.hvo[:10,2*9])

    #mso
    mso = hvo_reset.mso()
    print(mso.shape,hvo_reset.hvo.shape)
    spec, f = hvo_reset.get_onset_strength_spec()

    import matplotlib.pyplot as plt
    #plt.pcolormesh(spec.T)
    #plt.show()

    X, f = hvo_reset.get_logf_stft()
    plt.pcolormesh(X)
    plt.show()
