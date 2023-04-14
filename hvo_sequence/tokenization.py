import numpy as np


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
                tokens.append((f"delta_{step}", np.zeros((1, int(3 * n_voices)))))
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
        tokenized_seq.append((item, np.zeros((1, int(3 * n_voices)))))

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
            tokenized_seq.append(("hit", local_hvo_seq[hit_locs[ix]:hit_locs[ix] + 1, :]))

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
        tokenized_seq.append((item, np.zeros((1, int(3 * n_voices)))))

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
        tokenized_seq.append((item, np.zeros((1, int(3 * num_voices)))))

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


# ---------------------------------
# Reverse Tokenization

def calculateSequenceLength(token_seq):
    """
    This utility function is designed to calculate the total length of a new HVO (numpy array) when
    provided with a tokenized sequence.
    """

    length = 0

    for token in token_seq:
        token_type = str(token[0])
        if "delta" in token_type:
            length += int(token_type[6:])

    return length

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