from hvo_sequence.hvo_seq import HVO_Sequence, empty_like, zero_like
from hvo_sequence.drum_mappings import ROLAND_REDUCED_MAPPING, Groove_Toolbox_5Part_keymap, GM1_FULL_MAP
import numpy as np

if __name__ == "__main__":
    # Create an instance of a HVO_Sequence
    hvo_seq = HVO_Sequence(drum_mapping=ROLAND_REDUCED_MAPPING)

    # Add two time_signatures
    hvo_seq.add_time_signature(0, 4, 4, [4])
    # hvo_seq.add_time_signature(13, 6, 8, [3, 2])

    # Add two tempos
    hvo_seq.add_tempo(0, 50)
    hvo_seq.add_tempo(12, 20)  # Tempo Change at the beginning of second bar

    # Create a random hvo
    hvo_seq.hvo = np.random.rand(91, 27)

    #######################################
    #
    #           START TESTING
    #
    #######################################

    # 1. Test HVO_Sequence.copy() method
    hvo_seq_copied = hvo_seq.copy()

    # 2.Test HVO_Sequence.copy_empty() method
    # Create empty HVO_Sequence like hvo_seq
    empty_hvo_seq = hvo_seq.copy_empty()
    # print(empty_hvo_seq, hvo_seq)             # addresses should be different
    # print(empty_hvo_seq.__dict__)             # hvo should be empty
    # print(hvo_seq.__dict__)                   # hvo should not be empty

    # 3. Test empty_like()
    empty_hvo_seq2 = empty_like(hvo_seq)
    # print(empty_hvo_seq2, hvo_seq)            # addresses should be different
    # print(empty_hvo_seq2.__dict__)            # hvo should be empty
    # print(hvo_seq.__dict__)                   # hvo should not be empty

    # 4. Test HVO_Sequence.copy_zero() method
    # Create empty HVO_Sequence like hvo_seq
    zero_hvo_seq = hvo_seq.copy_zero()
    # print(zero_hvo_seq, hvo_seq)              # addresses should be different
    # print(zero_hvo_seq.__dict__)              # hvo should be zero
    # print(hvo_seq.__dict__)                   # hvo should not be zero

    # 5. Test zero_like() method
    # Create empty HVO_Sequence like hvo_seq
    zero_hvo_seq2 = zero_like(hvo_seq)
    # print(zero_hvo_seq, hvo_seq)              # addresses should be different
    # print(zero_hvo_seq2.__dict__)             # hvo should be zero
    # print(hvo_seq.__dict__)                   # hvo should not be empty

