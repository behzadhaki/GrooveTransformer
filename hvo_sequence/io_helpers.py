import pickle

try:
    import note_seq
    import pretty_midi
    _HAS_NOTE_SEQ_Pretty_Midi = True
except ImportError:
    _HAS_NOTE_SEQ_Pretty_Midi = False
    print("note_seq and/or pretty MIDI missing --> synthesis and MIDI conversion not possible")

from hvo_sequence.utils import find_nearest, find_pitch_and_tag
from hvo_sequence.hvo_seq import HVO_Sequence

try:
    import soundfile as sf
    import fluidsynth
    _CAN_SYNTHESIZE = True
except ImportError:
    _CAN_SYNTHESIZE = False

def note_sequence_to_hvo_sequence(ns, drum_mapping, beat_division_factors, num_steps=None, only_drums=False):
    """ Converts a note sequence to an HVO sequence
    :param ns:                note sequence object
    :param drum_mapping:        dict of {'Drum Voice Tag': [midi numbers]}
    :param beat_division_factors:   list of beat division factors
    :param num_steps:           number of steps in the HVO sequence
    :param only_drums:          if True, only ns.note.is_drum==True are considered
    :return:
    """

    # Create an empty HVO_Sequence instance
    hvo_seq = HVO_Sequence(drum_mapping=drum_mapping, beat_division_factors=beat_division_factors)

    # sort time_sigs and tempos by time
    time_sigs = sorted(ns.time_signatures, key=lambda x: x.time)
    tempos = sorted(ns.tempos, key=lambda x: x.time)

    # add first tempo and time_sig at index 0 (even if they are stamped later), this
    # is necessary for the hvo_seq to be able to interpolate between time stamps in
    # seconds and time stamps as grid steps
    hvo_seq.add_time_signature(0, time_sigs[0].numerator, time_sigs[0].denominator)
    hvo_seq.add_tempo(0, tempos[0].qpm)

    # add the rest of time sigs
    for time_sig in time_sigs[1:]:
        t_index, _ = hvo_seq.grid_maker.get_index_and_offset_at_sec(time_sig.time)
        hvo_seq.add_time_signature(t_index, time_sig.numerator, time_sig.denominator)

    # add the rest of tempos
    for tempo in tempos[1:]:
        t_index, _ = hvo_seq.grid_maker.get_index_and_offset_at_sec(tempo.time)
        hvo_seq.add_tempo(t_index, tempo.qpm)

    # add notes from latest to earliest
    for nsn in sorted(ns.notes, key=lambda x: x.start_time, reverse=True):
        note2add = None
        if only_drums:
            if nsn.is_drum:
                note2add = nsn
        else:
            note2add = nsn

        if note2add:
            hvo_seq.add_note(
                start_sec=note2add.start_time,
                pitch=note2add.pitch,
                velocity=note2add.velocity / 127.,
                overdub_with_louder_only=False)

    # if length is specified, pad the sequence with zeros
    if num_steps:
        hvo_seq.adjust_length(num_steps)

    return hvo_seq


def midi_to_note_seq(filename):
    if not _HAS_NOTE_SEQ_Pretty_Midi:
        print("note_seq not found. Please install it using `pip install note_seq`.")
        return None

    midi_data = pretty_midi.PrettyMIDI(filename)
    ns = note_seq.midi_io.midi_to_note_sequence(midi_data)
    return ns


def midi_to_hvo_sequence(filename, drum_mapping, beat_division_factors):
    ns = midi_to_note_seq(filename)
    if ns is None:
        return None

    return note_sequence_to_hvo_sequence(ns, drum_mapping=drum_mapping, beat_division_factors=beat_division_factors)


def place_note_in_hvo(ns_note, hvo, grid, drum_mapping):
    """
    updates the entries in hvo corresponding to features available in ns_note

    @param ns_note:                 note_sequence.note to place in hvo matrix
    @param hvo:                     hvo matrix created for the note_sequence score
    @param grid:                    grid corresponding to hvo matrix (list of time stamps in seconds)
    @param drum_mapping:            dict of {'Drum Voice Tag': [midi numbers]}
    @return hvo:                    hvo matrix containing information from ns_note
    """

    grid_index, utiming = get_grid_position_and_utiming_in_hvo(ns_note.start_time, grid)

    _, _, pitch_group_ix = find_pitch_and_tag(ns_note.pitch, drum_mapping)

    n_drum_voices = len(drum_mapping.keys())  # Get the number of reduced drum classes

    # if pitch was in the pitch_class_list
    # also check if the corresponding note is already filled, only update if the velocity is louder
    if (pitch_group_ix is not None) and ((ns_note.velocity / 127.0) > hvo[grid_index, n_drum_voices]):
        hvo[grid_index, pitch_group_ix] = 1  # Set hit  to 1
        hvo[grid_index, pitch_group_ix + n_drum_voices] = ns_note.velocity / 127.0  # Set velocity (0, 1)
        hvo[grid_index, pitch_group_ix + n_drum_voices * 2] = utiming  # Set utiming (-.5, 0.5)

    return hvo


def get_grid_position_and_utiming_in_hvo(start_time, grid):
    """
    Finds closes grid line and the utiming deviation from the grid for a queried onset time in sec

    @param start_time:                  Starting position of a note
    @param grid:                        Grid lines (list of time stamps in sec)
    @return tuple of grid_index,        the index of the grid line closes to note
            and utiming:                utiming ratio in (-0.5, 0.5) range
    """
    grid_index, grid_sec = find_nearest(grid, start_time)

    utiming = start_time - grid_sec                         # utiming in sec

    if utiming < 0:                                         # Convert to a ratio between (-0.5, 0.5)
        if grid_index == 0:
            utiming = 0
        else:
            utiming = utiming / (grid[grid_index] - grid[grid_index-1])
    else:
        if grid_index == (grid.shape[0]-1):
            utiming = utiming / (grid[grid_index] - grid[grid_index-1])
        else:
            utiming = utiming / (grid[grid_index+1] - grid[grid_index])

    return grid_index, utiming


#   ------------------- Pickle Loader --------------------

def get_pickled_note_sequences(pickle_path, item_list=None):
    """
    loads a pickled file of note_sequences, also allows for grabbing only specific items from the pickled set
    @param pickle_path:                     # path to the pickled set
    @param item_list:                       # list of items to grab, leave as None to get all
    @return note_sequences:                 # either all the items in set (when item_list == None)
                                            # or a list of sequences (when item_list = [item indices]
                                            # or a single item (when item_list is an int)
    """
    if not _HAS_NOTE_SEQ_Pretty_Midi:
        print("note_seq missing! as a result, can't load the pickled instances. Please install it using `pip install note_seq`.")
        return None

    # load pickled items
    note_sequence_pickle_file = open(pickle_path, 'rb')
    note_sequences = pickle.load(note_sequence_pickle_file)

    # get queried items or all
    if item_list:                            # check if specific items are queried
        if isinstance(item_list, list):      # Grab queried items
            note_sequences = [note_sequences[item] for item in item_list]
        else:                                # in case a single item (as integer) requested
            note_sequences = note_sequences[item_list]

    return note_sequences


def get_pickled_hvos(pickle_path, item_list=None):
    """
    loads a pickled file of hvos, also allows for grabbing only specific items from the pickled set
    @param pickle_path:                     # path to the pickled set
    @param item_list:                       # list of items to grab, leave as None to get all
    @return hvos:                           # either all the items in set (when item_list == None)
                                            # or a list of hvos (when item_list = [item indices]
                                            # or a single item (when item_list is an int)
    """
    # load pickled items
    hvo_pickle_file = open(pickle_path, 'rb')
    hvos = pickle.load(hvo_pickle_file)

    # get queried items or all
    if item_list:                            # check if specific items are queried
        if isinstance(item_list, list):      # Grab queried items
            hvos = [hvos[item] for item in item_list]
        else:                                # in case a single item (as integer) requested
            hvos = hvos[item_list]

    return hvos


def load_HVO_Sequence_from_file(pickle_path):
    """
    Loads a pickled HVO_Sequence object
    """
    with open(pickle_path, 'rb') as f:
        hvo_seq = pickle.load(f)

    return hvo_seq


#   --------------- Data type Convertors --------------------


def get_reduced_pitch(pitch_query, pitch_class_list):
    """
    checks to which drum group the pitch belongs,
    then returns the index for group and the first pitch in group

    @param pitch_query:                 pitch_query for which the corresponding drum voice group is found
    @param pitch_class_list:            list of grouped pitches sharing same tags      [..., [50, 48], ...]
    @return tuple of voice_group_ix,    index of the drum voice group the pitch belongs to
            and pitch_group[0]:         the first pitch in the corresponding drum group
                                        Note: Returns (None, None) if no matching group is found
    """

    for voice_group_ix, pitch_group in enumerate(pitch_class_list):
        if pitch_query in pitch_group:
            return voice_group_ix, pitch_group[0]

    # If pitch_query isn't in the pitch_class_list, return None, None
    return None, None


def unique_pitches_in_note_sequence(ns):
    """
    Returns unique pitches existing in a note sequence score
    @param ns: note sequence object
    @return: list of unique pitches
    """
    if not _HAS_NOTE_SEQ_Pretty_Midi:
        print("note_seq missing! Please install it using `pip install note_seq`.")
        return None

    unique_pitches = set([note.pitch for note in ns.notes])
    return unique_pitches


#   -------------------- Midi Converters -----------------------

def save_note_sequence_to_midi(ns, filename="temp.mid"):
    if not _HAS_NOTE_SEQ_Pretty_Midi:
        print("note_seq missing! Please install it using `pip install note_seq`.")
        return None

    pm = note_seq.note_sequence_to_pretty_midi(ns)
    pm.write(filename)


#   -------------------- Audio Synthesizers --------------------

def note_sequence_to_audio(ns, sr=44100, sf_path="../test/soundfonts/Standard_Drum_Kit.sf2"):
    """
    Synthesize a note_sequence score to an audio vector using a soundfont
    if you want to save the audio, use save_note_sequence_to_audio()
    @param ns:                  note_sequence score
    @param sr:                  sample_rate for generating audio
    @param sf_path:             soundfont for synthesizing to audio
    @return audio:              the generated audio (if fluidsynth is installed, otherwise 1 second of silence)
    """
    if _CAN_SYNTHESIZE and _HAS_NOTE_SEQ_Pretty_Midi:
        pm = note_seq.note_sequence_to_pretty_midi(ns)
        audio = pm.fluidsynth(fs=sr, sf2_path=sf_path)
    else:
        audio = None
        print("FluidSynth and/or note_seq are not installed. Please install it to use this feature.")
    return audio


def save_note_sequence_to_audio(ns, filename="temp.wav", sr=44100,
                                sf_path="../test/soundfonts/Standard_Drum_Kit.sf2"):
    """
    Synthesize and save a note_sequence score to an audio vector using a soundfont
    @param ns:                  note_sequence score
    @param filename:            filename/path for saving the synthesized audio
    @param sr:                  sample_rate for generating audio
    @param sf_path:             soundfont for synthesizing to audio
    @return audio:              returns audio, in addition to saving it
    """
    """
        Synthesize a note_sequence score to an audio vector using a soundfont

        @param ns:                  note_sequence score
        @param sr:                  sample_rate for generating audio
        @param sf_path:             soundfont for synthesizing to audio
        @return audio:              the generated audio
        """
    if _CAN_SYNTHESIZE and _HAS_NOTE_SEQ_Pretty_Midi:
        pm = note_seq.note_sequence_to_pretty_midi(ns)
        audio = pm.fluidsynth(sf2_path=sf_path)
        sf.write(filename, audio, sr, 'PCM_24')
    else:
        audio = None
        print("FluidSynth and/or note_seq are not installed. Please install it to use this feature.")

    return audio

