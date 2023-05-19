import numpy as np
from copy import deepcopy

from hvo_sequence import HVO_Sequence
from hvo_sequence import ROLAND_REDUCED_MAPPING

def tokenizeDeltas(delta, delta_grains, n_voices):
    """
    Tokenize an integer using the provided step sizes.
    Args:
        delta (int): The integer to be tokenized.
        steps (list[int]): The list of allowed step sizes.
    Returns:
        list[int]: The tokens representing the integer using the provided step sizes.
    """
    tokens = []
    while delta > 0:
        for step in sorted(delta_grains, reverse=True):
            if delta >= step:
                tokens.append((f"delta_{step}", np.zeros((1, int(2 * n_voices)))))
                delta -= step
                break

    return tokens


def getContextualData(hvo_seq, input_list=[]):
    """
    Returns a list of strings for the given sub-region's data.
    If the string is in the data_dict below, it turns the value of that function
    Otherwise it simply attaches the string itself.
    Example: ["tempo", "custom_data"] -> ["120", "custom_data"]
    """
    # time_signature = (_3_4_seq.time_signatures[0].numerator, _3_4_seq.time_signatures[0].denominator)
    data_dict = {
        # Todo: Extract specific timesig/tempo info (instead of object) ("4_4") ("120"(rounded to int))
        "time_signature": (hvo_seq.time_signatures[0].numerator, hvo_seq.time_signatures[0].denominator),
        "tempo": hvo_seq.tempos[0].qpm,
        "beat": "beat",
        "measure": "measure"
    }

    output_list = []

    for string in input_list:
        if string in data_dict:
            output_list.append(data_dict[string])
        else:
            output_list.append(string)

    return output_list


def tokenizeSingleBeatRegion(hvo_seq,
                             start_position,
                             n_voices,
                             delta_grains,
                             tpb,
                             ignore_last_silence=False,
                             contextual_data=[]):
    """
    Given an hvo sequence representing a single beat region, tokenize all relevant data.
    'Beat' is defined by the denominator of the clip's time signature.
    Args:
        hvo_seq: HVO class
        start_position (int): The absolute tick position of this beat
        n_voices: number of voices in the sequence
        delta_grains: (list) available delta tokenizations, i.e. [30, 10, 5, 1]
        tpb: (int) ticks per beat
        ignore_last_silence: (bool) whether to include information on remaining time delta to next beat
    Returns:
        [list]: the tokenized sequence
    """

    # Slice to the hvo seq that is local to this individual beat
    local_hvo_seq = hvo_seq.hvo[start_position:(start_position + tpb), :]

    tokenized_seq = []

    prepend_data = getContextualData(hvo_seq, contextual_data)
    for item in prepend_data:
        tokenized_seq.append((item, np.zeros((1, int(2 * n_voices)))))

        # Identify indexes with hits
    hits = local_hvo_seq[:, :n_voices]
    (hit_locs, _) = np.nonzero(hits)
    locs = np.append([0], hit_locs)
    locs = np.append(locs, [tpb])
    deltas = locs[1:] - locs[:-1]
    is_last_silence = False

    for ix, note in enumerate(deltas):
        if hit_locs is not []:
            if ix >= len(hit_locs):
                is_last_silence = True
                if (ignore_last_silence):
                    break
        delta_tokens = tokenizeDeltas(note, delta_grains, n_voices)
        if delta_tokens is not []:
            tokenized_seq.extend(delta_tokens)
        if not is_last_silence:
            tokenized_seq.append(("hit", local_hvo_seq[hit_locs[ix]:hit_locs[ix] + 1, :(2 * n_voices)]))

    return tokenized_seq


def tokenizeMeasure(hvo_seq,
                    start_position_ticks,
                    n_voices,
                    ticks_per_measure,
                    delta_grains,
                    tpb,
                    ignore_last_silence=False,
                    measure_data=[],
                    beat_data=[]):
    """
    Given an hvo sequence, tokenize data in one measure of time.
    Args:
        hvo_seq: HVO class
        start_position_ticks (int): The absolute tick position of this measure
        n_voices: number of voices in the sequence
        ticks_per_measure: total ticks per measure (numerator * tpb)
        delta_grains: (list) available delta tokenizations, i.e. [30, 10, 5, 1]
        tpb: (int) ticks per beat
        ignore_last_silence: (bool) whether to include information on remaining time delta to next beat
        measure_data (list): Measure-level control data
        beat_data (list): Beat-level control data
    Returns:
        [list]: the tokenized sequence
    """

    # Append measure-level data to list
    tokenized_seq = []
    prepend_data = getContextualData(hvo_seq, measure_data)
    for item in prepend_data:
        tokenized_seq.append((item, np.zeros((1, int(2 * n_voices)))))

    position = start_position_ticks
    end_position = position + ticks_per_measure

    while (position < end_position):
        tokenized_seq.extend(tokenizeSingleBeatRegion(hvo_seq,
                                                      position,
                                                      n_voices,
                                                      delta_grains,
                                                      tpb,
                                                      ignore_last_silence,
                                                      beat_data))
        position += tpb

    return tokenized_seq


# Assumes no tempo or meter change within HVO

def tokenizeConsistentSequence(hvo_seq,
                               delta_grains,
                               tpb,
                               clip_data=[],
                               measure_data=[],
                               beat_data=["beat"],
                               ignore_last_silence=False):
    """
    Given an HVO sequence (of any length), returns a tokenized sequence. The original HVO
    must be in a single time-signature and tempo.
    Args:
        hvo_seq: HVO class sequence
        delta_grains (list): integers that can be used to sub-divide the delta shifts for tokenization
        tpb (int): "ticks for beat" - the granularity between each beat, as defined by denominator
        clip_data (list of strings): Items to be appended at the beginning of the clip data
        measure_data (list of strings): Items to be appended at the start of each new measure
        beat_data (list of strings): Items to be appended at the start of each beat
        ignore_last_silence (bool): Whether to add remaining deltas between final note and the next beat
    Returns:
        tokenized_sequence (numpy array): The tokenized sequence
    """



    assert len(hvo_seq.tempos) == 1, "No support for multi-tempo scores"
    assert len(hvo_seq.time_signatures) == 1, "No support for multi-TimeSig scores"

    num_voices = hvo_seq.number_of_voices
    total_length = hvo_seq.number_of_steps

    # Global timing functions
    numer = hvo_seq.time_signatures[0].numerator
    denom = hvo_seq.time_signatures[0].denominator

    ticks_per_measure = int(numer * tpb)

    # Append clip-level data to list
    tokenized_seq = []
    prepend_data = getContextualData(hvo_seq, clip_data)
    for item in prepend_data:
        tokenized_seq.append((item, np.zeros((1, int(2 * num_voices)))))

    position = 0
    # Iterate through each measure, using quarter note (floats) as time-value
    while (position < total_length):
        tokenized_seq.extend(tokenizeMeasure(hvo_seq,
                                             position,
                                             num_voices,
                                             ticks_per_measure,
                                             delta_grains,
                                             tpb,
                                             ignore_last_silence,
                                             measure_data,
                                             beat_data))
        position += ticks_per_measure

    return tokenized_seq



def flattenTokenizedSequence(tokenized_sequence, num_voices, flattened_voice_idx=2, flatten_velocities=False):

    flattened_sequence = []
    for sequence in tokenized_sequence:

        token_type = str(sequence[0])
        if token_type == "hit":
            original_array = sequence[1][0]
            flat_array = np.zeros((1, int(2 * num_voices)))
            flat_array[0][flattened_voice_idx] = 1.

            if not flatten_velocities:
                max_velocity = np.amax(original_array[num_voices:])
                flat_array[0][flattened_voice_idx + num_voices] = max_velocity
            else:
                flat_array[0][flattened_voice_idx + num_voices] = 0.8

            flat_tuple = ("hit", flat_array)
            flattened_sequence.append(flat_tuple)

        else:
            flattened_sequence.append(sequence)

    return flattened_sequence



def convertConsistentTokenizedSequenceToHVO(token_seq, tpb):
    """
    Given a tokenized sequence of data, return an HVO array
    Args:
        token_seq [list]: 2D tuples with (token_type, array)
        tpb [int]: Resolution of ticks per beat
    Returns:
        hvo array: an array of (tpq, 3*n_voices) dim
    """

    total_delta = 0
    sequence_length = calculateSequenceLength(token_seq)

    hvo_size = token_seq[0][1].shape[1]  # 27 (n_voices * 3)
    hvo = np.zeros((sequence_length, hvo_size))

    for idx, token in enumerate(token_seq):
        token_type = str(token[0])

        if token_type == "hit":
            hvo[total_delta] = token_seq[idx][1]
            pass
        if "delta" in token_type:
            delta = int(token_type[6:])
            total_delta += delta

    return hvo


"""
Reverse tokenization:
These functions are designed to work with the outputs of a model, and convert
them back into various data formats for evaluation and inference. Currently there are several
'layers' of data representation:

- MIDI
- HVO (sequence)
- HVO (array)
- Tokenized Sequence ([string, HV_array])
- Numerically tokenized sequence ([tokens], [hits], [velocities])

A dictionary provides the link between the tokenized sequence and numerically tokenized sequence,
i.e. identifying that {"hit": 0}



"""
# ---------------------------------
# Reverse Tokenization

def calculateSequenceLength(token_seq):
    """
    This utility function is designed to calculate the total length of a new HVO (numpy array) when
    provided with a tokenized sequence.
    """

    length = 1

    for token in token_seq:
        token_type = str(token[0])
        if "delta" in token_type:
            length += int(token_type[6:])

    return length

def convert_model_output_to_tokenized_sequence(t, h, v, reverse_vocab):

    assert t.size(dim=0) == h.size(dim=0) == v.size(dim=0), "All sequences must have same length"

    tokenized_arrays = list()

    for idx, token_value in enumerate(t):
        if token_value != 0:
            token = reverse_vocab[int(token_value)]
            hv = np.concatenate((h[idx], v[idx]), axis=None)
            tokenized_arrays.append((token, hv))

    return tokenized_arrays

def convert_tokenized_sequence_to_hvo_array(tokenized_arrays, num_voices=9, return_as_hvo=True):
    """
        Given a tokenized sequence of data, return an HVO array
        Args:
            tokenized_arrays [list]: 2D tuples with (token_type, array)
            num_voices [int]: Number of voices in the sequence
            return_as_hvo [bool]: if true, will convert HV to HVO with offsets as 0
        Returns:
            hv(o) array: a 2d array identical to HVO format

        """

    total_delta = 0
    sequence_length = calculateSequenceLength(tokenized_arrays)

    hvo_size = num_voices * 2  # 18 (n_voices * 2)
    hv = np.zeros((sequence_length, hvo_size))

    for idx, token in enumerate(tokenized_arrays):
        token_type = str(token[0])

        if token_type == "hit":
            try:
                #hv[total_delta] = tokenized_arrays[idx][1]
                hv[total_delta] = token[1]
                pass
            except:
                print(f"failed on idx {idx} out of {len(tokenized_arrays)}, token is:")
                print(token)

        if "delta" in token_type:
            delta = int(token_type[6:])
            total_delta += delta

    if not return_as_hvo:
        return hv

    else:
        # Create offsets, thus mimicking the original (x, 27) size of HVO
        hvo = np.pad(hv, pad_width=((0, 0), (0, 9)), mode='constant', constant_values=0)
        return hvo


def batch_create_comparative_HVO_sequences(input_hvo_sequences, hvo_arrays, beat_div_factor=96):
    """
    #Todo: Is this function necessary? Currently unused
    @param input_hvo_sequences: [list] of original HVO sequences
    @param hvo_arrays: [list] of newly generated HVO arrays
    It is up to you to ensure that they are matched!
    hvo_sequences[0] should be the same pattern as hvo_arrays[0]
    It is not possible to assert this automatically

    @return: hvo_sequences: tuple of (input, output) hvo sequences
    """
    assert len(input_hvo_sequences) == len(hvo_arrays)

    hvo_sequences = list()

    for hvo_seq, hvo_array in zip(input_hvo_sequences, hvo_arrays):

        output_hvo_seq = HVO_Sequence(beat_division_factors=[beat_div_factor],
                                      drum_mapping=ROLAND_REDUCED_MAPPING)

        output_hvo_seq.add_time_signature(time_step=0,
                                          numerator=hvo_seq.time_signatures[0].numerator,
                                          denominator=hvo_seq.time_signatures[0].denominator)
        output_hvo_seq.add_tempo(time_step=0, qpm=float(hvo_seq.tempos[0].qpm))

        output_hvo_seq.hvo = hvo_array

        hvo_sequences.append((hvo_seq, output_hvo_seq))

    return hvo_sequences