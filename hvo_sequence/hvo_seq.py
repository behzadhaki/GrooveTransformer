import os.path
import numpy as np
try:
    import note_seq
    from note_seq.protobuf import music_pb2
    _HAS_NOTE_SEQ = True

except ImportError:
    print("Note Sequence not found. Please install it with `pip install note-seq`.")
    _HAS_NOTE_SEQ = False

try:
    import librosa
    import librosa.display
    _HAS_LIBROSA = True
except ImportError:
    _HAS_LIBROSA = False

try:
    import matplotlib.pyplot as plt
    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False

from bokeh.plotting import output_file, show, save
from bokeh.models import Span
from scipy.stats import skew
from scipy.signal import find_peaks
import math
import copy
import random
import pickle

from hvo_sequence.utils import cosine_similarity, cosine_distance
from hvo_sequence.utils import _weight_groove, _reduce_part, fuzzy_Hamming_distance
from hvo_sequence.utils import _get_kick_and_snare_syncopations, get_monophonic_syncopation
from hvo_sequence.utils import get_weak_to_strong_ratio, _getmicrotiming_event_profile_1bar
from hvo_sequence.utils import onset_strength_spec, reduce_f_bands_in_spec, detect_onset, map_onsets_to_grid, logf_stft
from hvo_sequence.utils import get_hvo_idxs_for_voice

from hvo_sequence.custom_dtypes import Metadata, GridMaker
from hvo_sequence.drum_mappings import Groove_Toolbox_5Part_keymap, Groove_Toolbox_3Part_keymap

from hvo_sequence.metrical_profiles import WITEK_SYNCOPATION_METRICAL_PROFILE_4_4_16th_NOTE
from hvo_sequence.metrical_profiles import Longuet_Higgins_METRICAL_PROFILE_4_4_16th_NOTE

from bokeh.plotting import figure
from bokeh.models import HoverTool
from bokeh.palettes import viridis


import logging
logger = logging.getLogger("HVO_Sequence.hvo_seq.py")

try:
    import fluidsynth
    _CAN_SYNTHESIZE = True
    from scipy.io import wavfile
except ImportError:
    _CAN_SYNTHESIZE = False
    logger.warning("Could not import fluidsynth. AUDIO rendering will not work.")

# --------------------- #
Version = "0.8.0"
# --------------------- #


class HVO_Sequence(object):

    def __init__(self, beat_division_factors, drum_mapping):
        """
        A piano roll representation of a drum sequence.
        :param beat_division_factors:       list of integers specifying how many subdivisions of a beat are used.
                                            beat is always considered to be the denominator of the time signature.
        :param drum_mapping:                dict mapping drum names to integers.
                                            exp:
                                            {
                                                "kick": [36, 38],
                                                "snare": [37, 40],
                                                ...
                                            }
        """

        self.__version = Version

        self.__metadata = Metadata()

        self.__grid_maker = GridMaker(beat_division_factors)

        self.__drum_mapping = None
        self.__hvo = None

        # Use property setters to initiate properties (DON"T ASSIGN ABOVE so that the correct datatype is checked)
        self.drum_mapping = drum_mapping

    #   ----------------------------------------------------------------------
    #           Pickling Strategies
    #   ----------------------------------------------------------------------
    def __getstate__(self):
        state_dict = {
            "created_with_version": self.__version,
            "metadata": self.__metadata,
            "grid_maker": self.__grid_maker,
            "drum_mapping": self.__drum_mapping,
        }

        # find_non_zero_items in hvo, no need to store non-hits
        if self.__hvo is not None:
            event_idx = np.nonzero(self.__hvo)
            event_vals = self.__hvo[event_idx]
            shape = self.__hvo.shape
            state_dict.update(
                {
                    "hvo":
                        {
                            "event_idx": event_idx,
                            "event_vals": event_vals,
                            "shape": shape
                        }
                }
            )

        return state_dict

    def __setstate__(self, state):
        # if "created_with_version" in state:
        #     logger.info("Pickled HVO_Sequence was created using version: {}.".format(state["created_with_version"]))
        #     if state["created_with_version"] != Version:
        #         logger.warning("Version Mismatch. Loaded with Version: {}.".format(Version))

        if "_HVO_Sequence__version" in state:
            # old version before "0.6.0"
            self.__hvo = state["_HVO_Sequence__hvo"]
            self.__metadata = Metadata(state["_HVO_Sequence__metadata"])
            self.__time_signatures = state["_HVO_Sequence__time_signatures"]
            self.__tempos = state["_HVO_Sequence__tempos"]
            self.__drum_mapping = state["_HVO_Sequence__drum_mapping"]
            self.__version = Version

        else:
            self.__version = Version
            self.__metadata = state["metadata"] if "metadata" in state else Metadata()
            self.__grid_maker = state["grid_maker"]
            self.__drum_mapping = state["drum_mapping"]

            if "hvo" in state:
                self.__hvo = np.zeros(state["hvo"]["shape"])
                self.__hvo[state["hvo"]["event_idx"]] = state["hvo"]["event_vals"]
            else:
                self.__hvo = None

    def save(self, path):
        # make sure the path ends with .hvo
        if not path.endswith(".hvo"):
            path += ".hvo"
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump(self, f)
            logging.info("HVO_Sequence saved to: {}".format(path))

    def load(self, path):
        with open(path, "rb") as f:
            hvo_seq = pickle.load(f)
            logging.info("HVO_Sequence loaded from: {}".format(path))
        self.__dict__ = hvo_seq.__dict__
        return self

    #   ----------------------------------------------------------------------
    #          Overridden Operators for ==, !=, +
    #   ----------------------------------------------------------------------
    def __add__(self, other_):
        """
        Overridden operator for adding two HVO_Sequence objects together

        if any of the HVO_Sequences are empty, the length is set to 1 bar after the beginning of the last segment

        :param other_:      another HVO_Sequence object
        :return:            a new HVO_Sequence object
        """
        assert (self.drum_mapping == other_.drum_mapping), "Drum mappings are not the same"
        assert self.time_signatures, \
            "The time signature of the object on the Left side of the + operator can't be empty"
        assert self.tempos, "The tempo of the object on the Left side of the + operator can't be empty"

        first_part = self.copy()
        other = other_.copy()

        if first_part.hvo is None:
            last_time_sig = sorted(first_part.grid_maker.time_signatures, key=lambda x: x.time_step)[-1]
            last_tempo = sorted(first_part.grid_maker.tempos, key=lambda x: x.time_step)[-1]
            time = max(last_time_sig.time_step, last_tempo.time_step)
            n_steps = time + last_time_sig.numerator * first_part.grid_maker.n_steps_per_beat
            first_part.zeros(n_steps)
        if other.hvo is None:
            last_time_sig = sorted(other.grid_maker.time_signatures, key=lambda x: x.time_step)[-1]
            last_tempo = sorted(other.grid_maker.tempos, key=lambda x: x.time_step)[-1]
            time = max(last_time_sig.time_step, last_tempo.time_step)
            n_steps = time + last_time_sig.numerator * other.grid_maker.n_steps_per_beat
            other.zeros(n_steps)

        # ensure second one starts on the next available beat
        next_t_step = first_part.hvo.shape[0]
        while next_t_step % self.grid_maker.n_steps_per_beat != 0:
            next_t_step += 1
        first_part.adjust_length(next_t_step)

        if other.tempos:
            for tempo in sorted(other.tempos, key=lambda x: x.time_step):
                first_part.add_tempo(time_step=tempo.time_step + next_t_step, qpm=tempo.qpm)

        if other.time_signatures:
            for time_sig in sorted(other.time_signatures, key=lambda x: x.time_step):
                first_part.add_time_signature(
                    time_step=time_sig.time_step + next_t_step,
                    numerator=time_sig.numerator, denominator=time_sig.denominator)

        if other.metadata.keys() is not None:
            first_part.metadata.append(other.metadata, next_t_step)

        first_part.hvo = np.concatenate((first_part.hvo, other.hvo), axis=0)

        return first_part

    def __eq__(self, other):
        checks = [self.__metadata == other.__metadata, self.__grid_maker == other.__grid_maker,
                  self.__drum_mapping == other.__drum_mapping, np.array_equal(self.__hvo, other.__hvo)]
        checks = all(checks)
        return checks

    #   ----------------------------------------------------------------------
    #   Essential properties (which require getters and setters)
    #   Property getters and setter wrappers for ESSENTIAL class variables
    #   ----------------------------------------------------------------------
    @property
    def __version__(self):
        return self.__version

    @property
    def metadata(self):
        """ Gives access to the metadata of the HVO_Sequence as a Metadata object """
        return self.__metadata

    @property
    def grid_maker(self):
        """gives access to the grid_maker object of type GridData"""
        return self.__grid_maker

    @metadata.setter
    def metadata(self, metadata_instance):
        assert isinstance(
            metadata_instance, dict), "Expected a dictionary Instance but received {}. " \
                                          "Use a Metadata instance available in hvo_sequence.custom_dtypes".format(
            type(metadata_instance))
        self.__metadata = metadata_instance

    @property
    def time_signatures(self):
        """ Adds a Time_Signature at a given time step index (not in seconds) to the grid_maker"""
        return self.__grid_maker.time_signatures

    def add_time_signature(self, time_step=None, numerator=None, denominator=None):
        self.__grid_maker.add_time_signature(time_step=time_step, numerator=numerator, denominator=denominator)

    @property
    def tempos(self):
        """ Gives access to the tempos of the HVO_Sequence stored in grid_maker"""
        return self.__grid_maker.tempos

    def add_tempo(self, time_step=None, qpm=None):
        """ Adds a Tempo at a given time step index (not in seconds) to the grid_maker"""
        self.__grid_maker.add_tempo(time_step=time_step, qpm=qpm)

    @property
    def drum_mapping(self):
        """ Gives access to the drum_mapping of the HVO_Sequence"""
        if not self.__drum_mapping:
            logger.warning("drum_mapping is not specified")
        return self.__drum_mapping

    @drum_mapping.setter
    def drum_mapping(self, drum_map):
        """ Sets the drum_mapping of the HVO_Sequence"""
        # Ensure drum map is a dictionary
        assert isinstance(drum_map, dict), "drum_mapping should be a dict" \
                                           "of {'Drum Voice Tag': [midi numbers]}"

        # Ensure the values in each key are non-empty list of ints between 0 and 127
        for key in drum_map.keys():
            assert isinstance(drum_map[key], list), "map[{}] should be a list of MIDI numbers " \
                                                    "(int between 0-127)".format(drum_map[key])
            if len(drum_map[key]) >= 1:
                assert all([isinstance(val, int) for val in drum_map[key]]), "Expected list of ints in " \
                                                                             "map[{}]".format(drum_map[key])
            else:
                assert False, "map[{}] is empty --> should be a list of MIDI numbers " \
                              "(int between 0-127)".format(drum_map[key])

        if self.hvo is not None:
            assert self.hvo.shape[1] % len(drum_map.keys()) == 0, \
                "The second dimension of hvo should be three times the number of drum voices, len(drum_mapping.keys())"

        # Now, safe to update the local drum_mapping variable
        self.__drum_mapping = drum_map

    @property
    def hvo(self):
        """returns 'synced' hvo np.array, meaning for hits == 0, velocities and offsets are set to 0"""
        if self.__hvo is not None:
            # Return 'synced' hvo array, meaning for hits == 0, velocities and offsets are set to 0
            # i.e. the actual values stored internally will be overridden by 0
            n_voices = int(self.__hvo.shape[1] / 3)
            self.__hvo[:, n_voices:2*n_voices] = self.__hvo[:, n_voices:2*n_voices]*self.__hvo[:, :n_voices]
            self.__hvo[:, 2*n_voices:] = self.__hvo[:, 2*n_voices:]*self.__hvo[:, :n_voices]

        return self.__hvo

    @hvo.setter
    def hvo(self, x):
        """sets the hvo array of the HVO_Sequence"""
        # Ensure x is a numpy.ndarray of shape (number of steps, 3*number of drum voices)
        assert isinstance(x, np.ndarray), "Expected numpy.ndarray of shape (time_steps, 3 * number of voices), " \
                                          "but received {}".format(type(x))
        assert x.shape[1] / len(self.drum_mapping.keys()) == 3, \
            f"The second dimension of hvo should be three times the number of drum voices, {len(self.drum_mapping)}"

        # Now, safe to update the local hvo score array
        self.__hvo = x
        self.number_of_steps = x.shape[0]

    @property
    def number_of_steps(self):
        """returns the length of the HVO_Sequence in number steps"""
        return 0 if self.hvo is None else self.hvo.shape[0]

    @number_of_steps.setter
    def number_of_steps(self, n_steps):
        self.adjust_length(n_steps)

    @property
    def consistent_segment_hvo_sequences(self):
        """
        Returns a list of hvo sequences, where each sequence is a time_sig and
        tempo consistent segment of the original hvo sequence.
        """
        # todo make compatible with grid_maker

        # Start Segmentation
        segments = []                   # List of HVO_Sequences

        segments_info = self.__grid_maker.get_segments_info()

        # Iterate over each segment and find correct Metadata and Create the HVO_Sequence according to the segment
        for i, (tempo, time_sig) in enumerate(zip(segments_info["tempos"], segments_info["time_signatures"])):
            if segments_info["segment_starts"][i] <= self.number_of_steps:
                # Create the HVO_Sequence templates
                segments.append(
                    HVO_Sequence(beat_division_factors=self.__grid_maker.beat_division_factors,
                                 drum_mapping=self.drum_mapping))
                segments[-1].add_time_signature(time_step=time_sig.time_step, numerator=time_sig.numerator,
                                                denominator=time_sig.denominator)
                segments[-1].add_tempo(time_step=tempo.time_step, qpm=tempo.qpm)

        # add score and metadata to each segment
        # segment_ends = np.array(segment_starts[1:] + [np.inf])-1
        for ix, segment in enumerate(segments):

            if self.hvo is not None:
                segment.hvo = self.hvo[segments_info["segment_starts"][ix]:segments_info["segment_ends"][ix], :]

            # Split the Metadata Segments
            starts_ends_metas = self.metadata.split() if self.metadata else None

            # Find the correct metadata for each segment
            metas_to_add = []
            if starts_ends_metas is not None:
                for m_seg_ix, m_seg in enumerate(starts_ends_metas):
                    m_start = m_seg[0]
                    m_end = m_seg[1]
                    meta = m_seg[2]
                    # check if start time within segment
                    if segments_info["segment_starts"][m_seg_ix] <= meta.time_steps[0]\
                            <= segments_info["segment_ends"][m_seg_ix]:
                        metas_to_add.append((m_start-segments_info["segment_starts"][m_seg_ix], meta))
                    # check if left overlapping
                    elif m_start < meta.time_steps[0] < m_end:
                        metas_to_add.append((0, meta))
                metas_to_add = sorted(metas_to_add, key=lambda x: x[0])
                for m_ix, m in enumerate(metas_to_add):
                    if m_ix == 0:
                        segment.metadata = m[1]
                    else:
                        segment.metadata.append(m[1], start_at_time_step=m[0])

        return segments, segments_info["segment_starts"]


    def split_into_segments(self, number_of_bars_per_segment, segment_shift_in_bars, adjust_length):
        segments = []  # List of HVO_Sequences
        consistent_segment_hvo_sequences, _ = self.consistent_segment_hvo_sequences

        for hvo_seq_consistent in consistent_segment_hvo_sequences:
            seg_numerator = hvo_seq_consistent.time_signatures[0].numerator
            steps_per_bar = len(hvo_seq_consistent.grid_maker.get_grid_lines_for_n_beats(seg_numerator))
            hvo_seq_ = hvo_seq_consistent.copy()

            while True:
                # grab number_of_bars_per_segment bars
                temp_hvo_seq = hvo_seq_.copy_empty()
                temp_hvo_seq.hvo = hvo_seq_.hvo[:steps_per_bar * number_of_bars_per_segment, :]
                if adjust_length:
                    temp_hvo_seq.adjust_length(steps_per_bar * number_of_bars_per_segment)
                segments.append(temp_hvo_seq)

                # shift by segment_shift_in_bars bars
                if hvo_seq_.hvo[segment_shift_in_bars:, :].shape[0] > (segment_shift_in_bars * steps_per_bar):
                    hvo_seq_.hvo = hvo_seq_.hvo[segment_shift_in_bars * steps_per_bar:, :]
                else:
                    break
        return segments

    #   --------------------------------------------------------------
    #   Utilities to modify hvo sequence
    #   --------------------------------------------------------------
    def remove_hvo(self):
        """removes hvo content and resets to None"""
        self.__hvo = None

    def get_active_voices(self):
        """
        Returns the indices of the voices that are active (i.e. have any hits) for this HVO
        """
        return np.argwhere(np.sum(self.hits, axis=0) > 0).flatten()

    def reset_voices(self, voice_idx=None):
        """
        Returns two HVO_Sequence objects, one (or more if list) with the specified voice(s) removed,
        and one with only the specified voice(s)

        **Note THIS METHOD DOES NOT MODIFY THE ORIGINAL HVO_SEQUENCE**
        """
        if voice_idx is None:
            logger.warning("Pass a voice index or a list of voice indexes to be reset")
            return None

        # for consistency, turn voice_idx int into list
        if isinstance(voice_idx, int):
            voice_idx = [voice_idx]

        # copy original hvo_seq into hvo_reset and hvo_reset_complementary
        hvo_reset = self.copy()  # copies the full hvo, each voice will be later set to 0
        hvo_reset_comp = self.copy_zero()  # copies a zero hvo, each voice will be later set to its values in hvo_seq

        n_voices = len(self.drum_mapping)  # number of instruments in the mapping

        if not np.all(np.isin(voice_idx, list(range(n_voices)))):
            logger.warning("Instrument index not in drum mapping")
            return None

        hvo_reset.hvo[:, voice_idx] = 0
        h_idx, v_idx, o_idx = get_hvo_idxs_for_voice(list(voice_idx), n_voices)
        hvo_reset_comp.hvo[:, h_idx + v_idx + o_idx] = self.hvo[:, h_idx + v_idx + o_idx]

        return hvo_reset, hvo_reset_comp

    def remove_random_events(self, threshold_range=(0.4, 0.6)):
        """
        Removes random hvo events sampling from a uniform probability distribution. A threshold is sampled from the
        threshold range. Hits with associated probability distribution value less or equal than this threshold are
        removed. Returns two HVO_Sequence objects, one with the specified event(s) removed, and one with only
        containing the removed event(s)

        **Note THIS METHOD DOES NOT MODIFY THE ORIGINAL HVO_SEQUENCE**
        """
        hvo_reset = self.copy()     # non_empty copying so that offset and velocity info is kept
        hvo_reset_comp = self.copy()

        # hvo_seq hvo hits 32x9 matrix
        n_voices = len(self.drum_mapping)
        hits = self.hvo[:, 0:n_voices]

        # uniform probability distribution over nonzero hits
        nonzero_hits_idx = np.nonzero(hits)
        pd = np.random.uniform(size=len(nonzero_hits_idx[0]))

        # get threshold from range
        threshold = random.uniform(*threshold_range)
        # sample hits from probability distribution
        hits_to_keep_idx = (nonzero_hits_idx[0][pd > threshold], nonzero_hits_idx[1][pd > threshold])
        hits_to_remove_idx = (nonzero_hits_idx[0][~(pd > threshold)], nonzero_hits_idx[1][~(pd > threshold)])

        # remove hits with associated probability distribution (pd) value lower than threshold
        hits_to_keep, hits_to_remove = np.zeros(hits.shape), np.zeros(hits.shape)

        hits_to_keep[tuple(hits_to_keep_idx)] = 1
        hvo_reset.hvo[:, 0:n_voices] = hits_to_keep

        hits_to_remove[tuple(hits_to_remove_idx)] = 1
        hvo_reset_comp.hvo[:, 0:n_voices] = hits_to_remove

        return hvo_reset, hvo_reset_comp

    def random(self, length):
        """Sets the .hvo attribute to a random sequence of the specified length"""
        h_ran = np.random.random_integers(0, 1, (length, self.number_of_voices))
        v_ran = np.random.ranf((length, self.number_of_voices)) * h_ran
        o_ran = (np.random.ranf((length, self.number_of_voices)) - 0.5) * h_ran
        self.hvo = np.concatenate([h_ran, v_ran, o_ran], axis=-1)

    def zeros(self, length):
        """Sets the .hvo attribute to a zero sequence of the specified length"""
        self.hvo = np.zeros((length, self.number_of_voices * 3))

    def adjust_length(self, max_size):
        """Adjusts the length of the hvo sequence to the specified number of steps.
        Adjustment by truncation or padding with zeros"""
        self.grid_maker.n_steps = max_size  # make sure grid is long enough
        if self.number_of_steps == 0:
            self.zeros(max_size)
        elif max_size < self.number_of_steps:
            self.hvo = self.hvo[:max_size, :]
        elif max_size > self.number_of_steps:
            self.hvo = np.concatenate(
                (self.hvo, np.zeros((max_size - self.number_of_steps, self.hvo.shape[1]))), axis=0)

    def flatten_voices(self, offset_aggregator_modes=3, velocity_aggregator_modes=1, 
                       get_velocities=True, reduce_dim=False, voice_idx=2):

        """ Flatten all voices into a single tapped sequence. If there are several voices hitting at the same
        time step, the loudest one will be selected and its offset will be kept, however velocity is discarded
        (set to the maximum). The default settings flatten the sequence into voice 2, using a max pooled velocity

        Parameters
        ----------
        offset_aggregator_modes : int
            Integer to choose which offset to keep:
                0: set to offset corresponding to max velocity value at that time step and set to 0 if multiple events
                1: set to smallest absolute offset at that time step
                2: set to largest absolute offset at that time step
                3: set to offset corresponding to max velocity value at that time step (DEFAULT)
                4: set to average of offsets at that time step
                5: set to sum of offsets at that time step
        velocity_aggregator_modes : int
            Integer to choose which velocity to keep:
                0: set to max velocity value at that time step and set to 1 if multiple events
                1: set to max velocity value at that time step (DEFAULT)
                2: set to velocity of the event with smallest offset
                3: set to average of velocities at that time step
                4: set to sum of velocities at that time step
        get_velocities: bool
            When set to True the function will return an hvo array with the hits, velocities and
            offsets of the voice with the hit that has maximum velocity at each time step, when
            set to False it will do the same operation but only return the hits and offsets, discarding
            the velocities at the end.
        reduce_dim: bool
            When set to False the hvo array returned will have the same number of voices as the original
            hvo, with the tapped sequenced in the selected voice_idx and the rest of the voices set to 0.
            When True, the hvo array returned will have only one voice.
        voice_idx : int
            The index of the voice where the tapped sequence will be stored. 0 by default.
            If reduce_dim is True, this index is disregarded as the flat hvo will only have
            one voice.
        """

        assert(0 <= offset_aggregator_modes <= 5), "invalid offset_aggregator_modes"
        assert(0 <= velocity_aggregator_modes <= 4), "invalid velocity_aggregator_modes"

        if not reduce_dim:
            # Make sure voice index is within range
            assert (self.number_of_voices > voice_idx >= 0), "invalid voice index"
        else:
            # Overwrite voice index
            voice_idx = 0

        new_hits = np.zeros_like(self.hits)
        new_velocities = np.zeros_like(self.velocities)
        new_offsets = np.zeros_like(self.offsets)

        synced_velocities = self.hits * self.velocities
        synced_offsets = self.hits * self.offsets

        all_idx = np.arange(self.number_of_steps)
        idx_max_vel = np.argmax(synced_velocities, axis=1)
        idx_multiple_hits = np.argwhere(np.sum(self.hits, axis=1) > 1)
        idx_smallest_offsets = np.argmin(np.abs(synced_offsets), axis=1)
        idx_biggest_offsets = np.argmax(np.abs(synced_offsets), axis=1)

        hit_idx = np.extract(np.any(self.hits != 0, axis=1), all_idx)
        new_hits[hit_idx, voice_idx] = 1

        if velocity_aggregator_modes == 0:
            new_velocities[all_idx, voice_idx] = synced_velocities[all_idx, idx_max_vel]
            new_velocities[idx_multiple_hits, voice_idx] = 1
        elif velocity_aggregator_modes == 1:
            new_velocities[all_idx, voice_idx] = synced_velocities[all_idx, idx_max_vel]
        elif velocity_aggregator_modes == 2:
            new_velocities[all_idx, voice_idx] = synced_velocities[all_idx, idx_smallest_offsets]
        elif velocity_aggregator_modes == 3:
            divider = np.sum(np.where(synced_velocities > 0, 1, 0), axis=1)
            new_velocities[all_idx, voice_idx] = np.sum(synced_velocities, axis=1) / np.where(divider != 0, divider, 1)
        else:
            new_velocities[all_idx, voice_idx] = np.sum(synced_velocities, axis=1)

        if offset_aggregator_modes == 0:
            new_offsets[all_idx, voice_idx] = synced_offsets[all_idx, idx_max_vel]
            new_offsets[idx_multiple_hits, voice_idx] = 0
        elif offset_aggregator_modes == 1:
            new_offsets[all_idx, voice_idx] = synced_offsets[all_idx, idx_smallest_offsets]
        elif offset_aggregator_modes == 2:
            new_offsets[all_idx, voice_idx] = synced_offsets[all_idx, idx_biggest_offsets]
        elif offset_aggregator_modes == 3:
            new_offsets[all_idx, voice_idx] = synced_offsets[all_idx, idx_max_vel]
        elif offset_aggregator_modes == 4:
            divider = np.sum(np.where(synced_offsets != 0, 1, 0), axis=1)
            new_offsets[all_idx, voice_idx] = np.sum(synced_offsets, axis=1) / np.where(divider != 0, divider, 1)
        else:
            new_offsets[all_idx, voice_idx] = np.sum(synced_offsets, axis=1)

        if reduce_dim:
            # if we want to return only 1 voice (instead of e.g. 9 with all the others to 0)
            # we remove the other dimensions and transform it into a 2-dim array so that it can be
            # concatenated after will join the arrays in the right axis
            new_hits = np.array([new_hits[:, voice_idx]]).T
            new_velocities = np.array([new_velocities[:, voice_idx]]).T
            new_offsets = np.array([new_offsets[:, voice_idx]]).T

        # concatenate arrays
        flat_hvo = np.concatenate((new_hits, new_velocities, new_offsets), axis=1)\
            if get_velocities else np.concatenate((new_hits, new_offsets), axis=1)
        return flat_hvo

    @property
    def hits(self):
        """ ndarray of dimensions m,n where m is the number of time steps and n the number of drums in the current
            drum mapping (e.g. 9 for the reduced mapping). The values of the array are 1 or 0, indicating whether a
            hit occurs at that time step for that drum (1) or not (0).
        """
        return self.__hvo[:, :self.number_of_voices] if self.is_hvo_score_available() else None

    def __is_hit_array_valid(self, hit_array):
        """checks to see if hit array is binary (0 / 1) and second dimension matches number of voices in drum mapping"""
        valid = True
        if len(self.__hvo[:, :self.number_of_voices]) != len(hit_array):
            valid = False
            logger.warning("hit array length mismatch")
        if not np.all(np.logical_or(np.asarray(hit_array) == 0, np.asarray(hit_array) == 1)):
            valid = False
            logger.warning("invalid hit values in array, they must be 0 or 1")
        return valid

    @hits.setter
    def hits(self, hit_array):
        """setter for hits array

        **hit_array: ndarray of dimensions m,n where m is the number of time steps and n the number of drums
        in the current drum mapping (e.g. 9 for the reduced mapping). The values of the array are 1 or 0, indicating whether a
        hit occurs at that time step for that drum (1) or not (0).**
        """
        assert self.__is_hit_array_valid(hit_array), "Hit array is invalid! Must be binary and second dimension " \
                                                     "must match the number of voices in drum_mapping"
        if self.hvo is None:  # if no hvo score is available set velocities to one at hit and offsets to zero
            self.hvo = np.concatenate((hit_array, hit_array, np.zeros_like(hit_array)), axis=1)
        else:
            self.hvo[:, :self.number_of_voices] = hit_array

    @property
    def velocities(self):
        """ ndarray of dimensions m,n where m is the number of time steps and n the number of drums in the current
            drum mapping (e.g. 9 for the reduced mapping). The values of the array are continuous floating point
            numbers from 0 to 1 indicating the velocity.
        """
        if not self.is_hvo_score_available():
            logger.warning("can't get velocities as there is no hvo score previously provided")
        else:
            # Note that the value returned is the internal one - even if a hit is 0 at an index,
            # velocity at that same index might not be 0
            return self.__hvo[:, self.number_of_voices: 2 * self.number_of_voices]

    def __is_vel_array_valid(self, vel_array):
        """checks to see if velocity array is continuous (0 to 1) and
        second dimension matches number of voices in drum mapping"""
        if vel_array.shape[1] != self.number_of_voices:
            logger.warning('Second dimension of vel_array must match the number of keys in drum_mapping')
            return False
        if np.min(vel_array) < 0 or np.max(vel_array) > 1:
            logger.warning("Velocity values must be between 0 and 1")
            return False
        if self.is_hvo_score_available():
            if vel_array.shape[0] != len(self.hvo[:, self.number_of_voices: 2 * self.number_of_voices]):
                logger.warning("velocity array length mismatch")
                return False
        return True

    @velocities.setter
    def velocities(self, vel_array):
        """setter for velocities array

        **vel_array: ndarray of dimensions m,n where m is the number of time steps and n the number of drums
        in the current drum mapping (e.g. 9 for the reduced mapping).
        The values of the array are continuous floating point numbers from 0 to 1 indicating the velocity.**
        """
        assert self.__is_vel_array_valid(vel_array), "velocity array is incorrect! either time step mismatch " \
                                                     "or second number of voices mismatch or values outside [0, 1]"

        if self.hvo is None:  # if hvo empty, set corresponding hits to one and offsets to zero
            self.hvo = np.concatenate((np.where(vel_array > 0, 1, 0),
                                       vel_array,
                                       np.zeros_like(vel_array)), axis=1)
        else:
            self.hvo[:, self.number_of_voices: 2 * self.number_of_voices] = vel_array

    @property
    def offsets(self):
        """ ndarray of dimensions m,n where m is the number of time steps and n the number of drums in the current
            drum mapping (e.g. 9 for the reduced mapping). The values of the array are continuous floating point
            numbers from -0.5 to 0.5 indicating the offset respect to the beat grid line that each hit is on.
        """
        if not self.is_hvo_score_available():
            logger.warning("can't get offsets/u_timings as there is no hvo score previously provided")
            return None
        else:
            # Note that the value returned is the internal one - even if a hit is 0 at an index,
            # offset at that same index might not be 0
            return self.hvo[:, 2 * self.number_of_voices:]

    def __is_offset_array_valid(self, offset_array):
        """checks to see if offset array is continuous (-0.5 to 0.5) and
        second dimension matches number of voices in drum mapping"""
        if self.is_hvo_score_available() is False:
            logger.warning("hvo field is empty: Can't set offsets without hvo field")
            return False

        if offset_array.shape[1] != self.number_of_voices:
            logger.warning("offset array length mismatch")
            return False

        if offset_array.mean() < -0.5 or offset_array.mean() > 0.5:
            logger.warning("invalid offset values in array, they must be between -0.5 and 0.5")
            return False

        return True

    @offsets.setter
    def offsets(self, offset_array):
        """setter for offsets array

        **offset_array: ndarray of dimensions m,n where m is the number of time steps and n the number of drums
        in the current drum mapping (e.g. 9 for the reduced mapping).
        The values of the array are continuous floating point numbers from -0.5 to 0.5 indicating the offset
        respect to the beat grid line that each hit is on.**
        """
        if not self.is_hvo_score_available():
            logger.warning("can't set offsets as there is no hvo score previously provided")
        else:
            if self.__is_offset_array_valid(offset_array):
                self.hvo[:, 2 * self.number_of_voices:] = offset_array

    #   ----------------------------------------------------------------------
    #   Utility methods to check whether required properties are
    #       available for carrying out a request
    #
    #   Assuming that the local variables haven't been modified directly,
    #   No need to check the validity of data types if they are available
    #       as this is already done in the property.setters
    #   ----------------------------------------------------------------------

    def is_time_signatures_available(self):
        """Checks whether time_signatures are already specified and necessary fields are filled"""

        if len(self.time_signatures) == 0:
            logger.warning("Time Signature missing")
            return False

        time_signatures_ready_to_use = list()
        if self.time_signatures is not None:
            for time_signature in self.time_signatures:
                time_signatures_ready_to_use.append(time_signature.is_ready_to_use)

        if not all(time_signatures_ready_to_use):
            for ix, ready_status in enumerate(time_signatures_ready_to_use):
                if ready_status is not True:
                    logger.warning(
                        "There are missing fields in Time_Signature {}: {}".format(ix, self.time_signatures[ix]))
            return False
        else:
            return True

    def is_tempos_available(self):
        """Checks whether tempos are already specified and necessary fields are filled"""
        tempos_ready_to_use = list()
        if self.tempos is not None:
            for tempo in self.tempos:
                tempos_ready_to_use.append(tempo.is_ready_to_use)
        else:
            logger.warning("No tempos specified")
            return False

        if not all(tempos_ready_to_use):
            for ix, ready_status in enumerate(tempos_ready_to_use):
                if ready_status is not True:
                    logger.warning(
                        "There are missing fields in Tempo {}: {}".format(ix, self.tempos[ix]))
            return False
        else:
            return True

    def is_hvo_score_available(self):
        # Checks whether hvo score array is already specified
        if self.hvo is None:
            logger.warning(".hvo field is empty: Can't get hits/velocities/offsets without hvo field")
            return False
        else:
            return True

    def is_ready_for_use(self):
        """checks if a .hvo score and time_signature info is available"""
        state = all([
            self.is_hvo_score_available(),
            self.is_time_signatures_available()
        ])
        return state

    #   -------------------------------------------------------------
    #   Method to get hvo in a flexible way
    #   -------------------------------------------------------------
    def get(self, hvo_str, offsets_in_ms=False, use_nan_for_non_hits=False):
        """
        Flexible method to get hits, velocities and offsets in the desired order, or zero arrays with the same
        dimensions as one of those vectors. The velocities and offsets are synced to the hits, so whenever a hit is 0,
        velocities and offsets will be 0 as well.

        Parameters
        ----------
        hvo_str: str
            String formed with the characters 'h', 'v', 'o' and '0' in any order. It's not necessary to use all the
            characters, they can be repeated. E.g. 'ov', will return the offsets and velocities, 'h0h' will return
            the hits, a 0-vector and the hits again, again and '000' will return a hvo-sized 0 matrix.

        offsets_in_ms: bool
            If true, the queried offsets will be provided in ms deviations from grid, otherwise, will be
            provided in terms of ratios

        use_nan_for_non_hits: bool
        If true, non-hit data will be returned as NaNs, otherwise, as 0s
        """
        assert self.is_hvo_score_available(), "No hvo score available, update this field"

        assert isinstance(hvo_str, str), 'hvo_str must be a string'
        hvo_str = hvo_str.lower()
        hvo_arr = []

        # Get h, v, o
        h = copy.deepcopy(self.hvo[:, :self.number_of_voices])
        v = copy.deepcopy(self.hvo[:, self.number_of_voices:self.number_of_voices*2])
        o = self.get_offsets_in_ms() if offsets_in_ms else self.hvo[:, self.number_of_voices*2:]
        o = copy.deepcopy(o)
        zero = np.zeros_like(h)

        # replace velocities and offsets with no associated hit to np.nan when use_nan_for_non_hits is set to True
        if use_nan_for_non_hits is not False:
            v[h == 0] = -1000000
            v = np.where(v == -1000000, np.nan, v)
            o[h == 0] = -1000000
            o = np.where(o == -1000000, np.nan, o)

        # Concatenate parts
        for c in hvo_str:
            assert (c == 'h' or c == 'v' or c == 'o' or c == '0'), 'hvo_str not valid'
            concat_arr = zero if c == '0' else h if c == 'h' else v if c == 'v' else o
            hvo_arr = concat_arr if len(hvo_arr) == 0 else np.concatenate((hvo_arr, concat_arr), axis=1)

        return hvo_arr

    def get_with_different_drum_mapping(self, hvo_str, tgt_drum_mapping,
                                        offsets_in_ms=False, use_nan_for_non_hits=False):
        """
        similar to self.get() except that it maps the extracted hvo sequence to a provided target mapping

        if multiple velocities/offsets are to be grouped together, only the position of the loudest velocity is used

        :param hvo_str: str
            String formed with the characters 'h', 'v', 'o' and '0' in any order. It's not necessary to use all the
            characters AND they can be repeated. E.g. 'ov', will return the offsets and velocities, 'h0h' will return
            the hits, a 0-vector and the hits again, again and '000' will return a hvo-sized 0 matrix.
        :param tgt_drum_mapping:        Alternative mapping to use
        :param offsets_in_ms:           True/False, specifies if offsets should be in ms
        :param use_nan_for_non_hits:    True/False, specifies if np.nan should be used instead of 0 wherever a hit is
                                        missing
        :return:
            the sequence associated with hvo_str mapped to a target drum map

        """

        def get_tgt_map_index_for_src_map(src_map, tgt_map):
            """
            Finds the corresponding voice group index for
            :param src_map:   a drum_mapping dictionary
            :param tgt_map:     a drum_mapping dictionary
            :return: list of indices in Base to be grouped together. Each element in returned list is the corresponding
            voice group to be used in
                     tgt_map for each of the voice groups in the src_map

                    Example:
                    if src_map = ROLAND_REDUCED_MAPPING and tgt_map = Groove_Toolbox_5Part_keymap
                    the return will be [[0], [1], [2, 8], [3, 7], [4, 5, 6]]
                    this means that kicks are to be mapped to kick
                                    snares are to be mapped to snares
                                    c_hat and rides are to be mapped to the same group (closed)
                                    o_hat and crash are to be mapped to the same group (open)
                                    low. mid. hi Toms are to be mapped to the same group (toms)

            """
            # Find the corresponding index in tgt mapping for each element in Base map
            src_ix_to_tgt_ix_map = np.array([])
            for src_voice_ix, src_voice_midi_list in enumerate(src_map.values()):
                corresponding_tgt_indices = []
                for tgt_voice_ix, tgt_voice_midi_list in enumerate(tgt_map.values()):
                    if src_voice_midi_list[0] in tgt_voice_midi_list:
                        corresponding_tgt_indices.append(tgt_voice_ix)
                src_ix_to_tgt_ix_map = np.append(src_ix_to_tgt_ix_map,
                                                 max(corresponding_tgt_indices, key=corresponding_tgt_indices.count))

            n_voices_tgt = len(tgt_map.keys())

            grouped_voices_ = [np.argwhere(src_ix_to_tgt_ix_map == tgt_ix).reshape(-1)
                               for tgt_ix in range(n_voices_tgt)]

            return grouped_voices_

        # Find Base indices in src_map corresponding to tgt
        grouped_voices = get_tgt_map_index_for_src_map(self.drum_mapping, tgt_drum_mapping)

        # Get non-reduced score with the existing mapping
        hvo = self.get("hvo", offsets_in_ms, use_nan_for_non_hits=False)
        h_src, v_src, o_src = np.split(hvo, 3, axis=1)

        # Create empty placeholders for hvo and zero
        h_tgt = np.zeros((h_src.shape[0], len(tgt_drum_mapping.keys())))
        v_tgt, o_tgt, zero_tgt = None, None, None
        if "v" in hvo_str or "o" in hvo_str:
            v_tgt = np.zeros((h_src.shape[0], len(tgt_drum_mapping.keys())))
            o_tgt = np.zeros((h_src.shape[0], len(tgt_drum_mapping.keys())))
        if "0" in hvo_str:
            zero_tgt = np.zeros((h_src.shape[0], len(tgt_drum_mapping.keys())))

        # use the groups of indices in grouped_voices to map the Base sequences to tgt sequence
        for ix, voice_group in enumerate(grouped_voices):
            if len(voice_group) > 0:
                h_tgt[:, ix] = np.any(h_src[:, voice_group], axis=1)
                if "v" in hvo_str or "o" in hvo_str:
                    v_max_indices = np.nanargmax(v_src[:, voice_group], axis=1)     # use loudest velocity
                    v_tgt[:, ix] = v_src[:, voice_group][range(len(v_max_indices)), v_max_indices]
                    o_tgt[:, ix] = o_src[:, voice_group][range(len(v_max_indices)), v_max_indices]

        # replace vels and offsets with no associated hit to np.nan if use_nan_for_non_hits set to True
        if use_nan_for_non_hits is not False and ("v" in hvo_str or "o" in hvo_str):
            v_tgt[h_tgt == 0] = -1000000
            v_tgt = np.where(v_tgt == -1000000, np.nan, v_tgt)
            o_tgt[h_tgt == 0] = -1000000
            o_tgt = np.where(o_tgt == -1000000, np.nan, o_tgt)

        # Concatenate parts according to hvo_str
        hvo_arr = []
        for c in hvo_str:
            assert (c == 'h' or c == 'v' or c == 'o' or c == '0'), 'hvo_str not valid'
            concat_arr = zero_tgt if c == '0' else h_tgt if c == 'h' else v_tgt if c == 'v' else o_tgt
            hvo_arr = concat_arr if len(hvo_arr) == 0 else np.concatenate((hvo_arr, concat_arr), axis=1)

        return hvo_arr

    def get_offsets_in_ms(self):
        """
        Gets the offset portion of hvo and converts the values to ms using the associated grid

        :return:    the offsets in hvo tensor in ms
        """
        convertible = all([self.is_tempos_available(),
                           self.is_time_signatures_available()])

        if not convertible:
            logger.warning("Above fields need to be provided so as to get the offsets in ms")
            return None

        # get the number of allowed drum voices
        n_voices = len(self.__drum_mapping.keys())

        # create an empty offsets array
        offsets_ratio = self.__hvo[:, 2*n_voices:]
        neg_offsets = np.where(offsets_ratio < 0, offsets_ratio, 0)
        pos_offsets = np.where(offsets_ratio > 0, offsets_ratio, 0)

        # Find negative and positive scaling factors for offset ratios
        grid = np.array(self.__grid_maker.get_grid_lines(self.number_of_steps))
        inter_grid_distances = (grid[1:] - grid[:-1]) * 1000    # 1000 for sec to ms
        neg_bar_durations = np.zeros_like(grid)
        pos_bar_durations = np.zeros_like(grid)
        # assume left of first gridline is similar to right of first gridline
        neg_bar_durations[0] = inter_grid_distances[0]
        neg_bar_durations[1:] = inter_grid_distances
        # assume right of last gridline is similar to left of last gridline
        pos_bar_durations[-1] = inter_grid_distances[-1]
        pos_bar_durations[:-1] = inter_grid_distances

        # Scale offsets by grid durations
        neg_offsets = neg_offsets*neg_bar_durations[:neg_offsets.shape[0], None]
        pos_offsets = pos_offsets * pos_bar_durations[:pos_offsets.shape[0], None]

        return neg_offsets+pos_offsets

    def get_notes(self, return_tuples=False):
        """
        Returns a dictionary containing information about the notes in the score. The dictionary has the following
        structure:

        {
            'start':    np.array of shape (n_notes, 1) containing the start time of each note in seconds
            'end':      np.array of shape (n_notes, 1) containing the end time of each note in seconds
            'instrument':   np.array of shape (n_notes, 1) containing the instrument number of each note
            'midi_nunmber': np.array of shape (n_notes, 1) containing the midi number of each note
            'velocity': np.array of shape (n_notes, 1) containing the velocity of each note
            'offset':   np.array of shape (n_notes, 1) containing the offset of each note in ratio of grid
            'offset_in_ms': np.array of shape (n_notes, 1) containing the offset of each note in ms
            'grid_line':    np.array of shape (n_notes, 1) containing the closest grid line to each note
        }

        :param return_tuples:   If True, returns a list containing tuples of the form
        (start, end, instrument, midi number, velocity, offset, offset_in_ms)

        """
        h = self.get("h")
        v = self.get("v")
        o = self.get("o")
        o_sec = self.get_offsets_in_ms()/1000
        grid_lines_sec = np.array(self.__grid_maker.get_grid_lines(self.number_of_steps))

        drum_voice_tags = [(k, v[0] if isinstance(v, list) else v) for k, v in self.drum_mapping.items()]
        note_duration = np.min(grid_lines_sec[1:] - grid_lines_sec[:-1]) / 2.0

        # for each row of h, find the indices of the nonzero elements
        # and use them to get the corresponding v and o values
        # and append them to the list of notes
        notes = {"start": [],
                 "end": [],
                 "instrument": [],
                 "voice_index": [],
                 "midi": [],
                 "velocity": [],
                 "offset": [],
                 "offset_sec": [],
                 "grid_line": []
                 }

        for i in range(self.number_of_steps):
            indices = np.nonzero(h[i, :])[0]
            for j in indices:
                notes["start"].append(grid_lines_sec[i]+o_sec[i, j])
                notes["end"].append(grid_lines_sec[i]+o_sec[i, j]+note_duration)
                notes["instrument"].append(drum_voice_tags[j][0])
                notes["voice_index"].append(j)
                notes["midi"].append(drum_voice_tags[j][1])
                notes["velocity"].append(np.round(v[i, j], 3))
                notes["offset"].append(np.round(o[i, j], 3))
                notes["offset_sec"].append(np.round(o_sec[i, j], 3))
                notes["grid_line"].append(i)

        if return_tuples:
            return list(
                zip(notes["start"], notes["end"], notes["instrument"], notes["voice_index"], notes["midi"],
                    notes["velocity"], notes["offset"], notes["offset_sec"], notes["grid_line"]))
        else:
            return notes

    #   ----------------------------------------------------------------------
    #            Calculated properties
    #   Useful properties calculated from ESSENTIAL class variables
    #   EACH SEGMENT MEANS A PART THAT TEMPO AND TIME SIGNATURE IS CONSTANT
    #   ----------------------------------------------------------------------

    @property
    def number_of_voices(self):
        """Returns the number of voices in the score"""
        return len(self.drum_mapping)

    #   --------------------------------------------------------------
    #   Utilities to Copy the object, Reset
    #   --------------------------------------------------------------
    def copy(self):
        """ Returns a copy of the object"""
        new = HVO_Sequence(beat_division_factors=self.__grid_maker.beat_division_factors,
                           drum_mapping=self.drum_mapping)
        new.__dict__ = copy.deepcopy(self.__dict__)
        return new

    def copy_empty(self):
        """returns a copy of the object with .hvo set to None"""
        new = HVO_Sequence(drum_mapping=self.drum_mapping,
                           beat_division_factors=self.__grid_maker.beat_division_factors)
        new.__dict__ = copy.deepcopy(self.__dict__)
        new.__hvo = None
        return new

    def copy_zero(self):
        """returns a copy of the object with .hvo set to zeros"""
        new = HVO_Sequence(drum_mapping=self.drum_mapping, beat_division_factors=self.__grid_maker.beat_division_factors)
        new.__dict__ = copy.deepcopy(self.__dict__)
        if new.hvo is not None:
            new.hvo = np.zeros_like(new.__hvo)
        return new

    #   --------------------------------------------------------------
    #   Utilities to Change Length and Add Notes
    #   --------------------------------------------------------------
    def find_index_for_pitch(self, pitch):
        """Finds the index of the voice that corresponds to the given pitch (using the drun mapping)"""
        for i, (k, v) in enumerate(self.drum_mapping.items()):
            if pitch in v:
                return i
        raise ValueError(f"Can't find pitch {pitch} in drum_mapping")

    def add_note(self, start_sec, pitch, velocity, overdub_with_louder_only=False):
        """Adds a note to the score using the given start time, pitch and velocity.

        :param start_sec:   onset time in seconds from the beginning of the score.
        :param pitch:       midi pitch of the note (must be available in the drum mapping)
        :param velocity:    0-1 velocity of the note
        :param overdub_with_louder_only:  If True, overdubs the note only if the velocity is higher than the existing
                                            velocity at the same time and pitch. otherwise, overrides the existing note.
        """
        assert velocity <= 1, "velocity should be between 0 and 1"

        voice_ix = self.find_index_for_pitch(pitch)

        # expand the score if necessary
        time_ix, offset = self.__grid_maker.get_index_and_offset_at_sec(start_sec)

        if time_ix >= self.number_of_steps:
            self.adjust_length(time_ix+1)

        if velocity > self.velocities[time_ix, voice_ix] or not overdub_with_louder_only:
            self.hits[time_ix, voice_ix] = 1
            self.velocities[time_ix, voice_ix] = velocity
            self.offsets[time_ix, voice_ix] = offset

    #   --------------------------------------------------------------
    #   Utilities to import/export/Convert different score formats such as
    #       1. NoteSequence, 2. HVO array, 3. Midi
    #   --------------------------------------------------------------

    def to_note_sequence(self, midi_track_n=9):
        """Exports the hvo_sequence to a note_sequence object

        :param midi_track_n:    the midi track channel used for the drum scores"""
        if not _HAS_NOTE_SEQ:
            print("Can't export to note sequence. Please install note_seq package")
            return None

        if self.is_ready_for_use() is False:
            return None

        # get grid
        grid_lines = np.array(self.__grid_maker.get_grid_lines(self.number_of_steps))

        # Create a note sequence instance
        ns = music_pb2.NoteSequence()

        # get the number of allowed drum voices
        n_voices = len(self.__drum_mapping.keys())

        # find nonzero hits tensor of [[position, drum_voice]]
        pos_instrument_tensors = np.transpose(np.nonzero(self.__hvo[:, :n_voices]))

        # Set note duration as 1/2 of the smallest grid distance
        note_duration = np.min(grid_lines[1:] - grid_lines[:-1]) / 2.0

        # Add notes to the NoteSequence object
        for drum_event in pos_instrument_tensors:  # drum_event -> [grid_position, drum_voice_class]
            grid_pos = drum_event[0]  # grid position
            drum_voice_class = drum_event[1]  # drum_voice_class in range(n_voices)

            # Grab the first note for each instrument group
            pitch = list(self.__drum_mapping.values())[drum_voice_class][0]
            velocity = self.__hvo[grid_pos, drum_voice_class + n_voices]  # Velocity of the drum event
            utiming_ratio = self.__hvo[  # exact timing of the drum event (rel. to grid)
                grid_pos, drum_voice_class + 2 * n_voices]

            utiming = 0
            if utiming_ratio < 0:
                # if utiming comes left of grid, figure out the grid resolution left of the grid line
                if grid_pos > 0:
                    utiming = (grid_lines[grid_pos] - grid_lines[grid_pos - 1]) * \
                              utiming_ratio
                else:
                    utiming = 0  # if utiming comes left of beginning,  snap it to the very first grid (loc[0]=0)
            elif utiming_ratio > 0:
                if grid_pos < (self.number_of_steps - 2):
                    utiming = (grid_lines[grid_pos + 1] -
                               grid_lines[grid_pos]) * utiming_ratio
                else:
                    utiming = (grid_lines[grid_pos] -
                               grid_lines[grid_pos - 1]) * utiming_ratio
                    # if utiming_ratio comes right of the last grid line, use the previous grid resolution for finding
                    # the utiming value in ms

            start_time = grid_lines[grid_pos] + utiming  # starting time of note in sec

            end_time = start_time + note_duration  # ending time of note in sec

            ns.notes.add(pitch=pitch, start_time=start_time.item(), end_time=end_time.item(),
                         is_drum=True, instrument=midi_track_n, velocity=int(velocity.item() * 127))

        # ns.total_time = self.total_len

        for tempo in self.tempos:
            loc_ = self.__grid_maker.get_grid_lines(tempo.time_step+1)[tempo.time_step]
            ns.tempos.add(
                time=loc_,
                qpm=tempo.qpm
            )

        for time_sig in self.time_signatures:
            loc_ = self.__grid_maker.get_grid_lines(time_sig.time_step+1)[time_sig.time_step]
            ns.time_signatures.add(
                time=loc_,
                numerator=time_sig.numerator,
                denominator=time_sig.denominator
            )

        return ns

    def save_hvo_to_midi(self, filename="misc/temp.mid", midi_track_n=9):
        """Exports to a  midi file

        :param filename:        the filename to save the midi file
        :param midi_track_n:    the midi track channel used for the drum scores
        """
        if not _HAS_NOTE_SEQ:
            print("Can't export to midi. Please install note_seq package")
            return None

        if self.is_ready_for_use() is False:
            logger.warning("The hvo_sequence is not exportable to MIDI. "
                           "Check that a Time_Signature, and a .hvo score is added.")
            return None

        if os.path.dirname(filename) != '':
            os.makedirs(os.path.dirname(filename), exist_ok=True)

        ns = self.to_note_sequence(midi_track_n=midi_track_n)
        pm = note_seq.note_sequence_to_pretty_midi(ns)
        pm.write(filename)
        return pm

    def convert_to_alternate_mapping(self, tgt_drum_mapping):
        """Returns a new hvo_sequence with the same content, but with a different drum mapping"""
        if self.is_ready_for_use() is False:
            return None

        hvo_seq_tgt = HVO_Sequence(
            beat_division_factors=self.__grid_maker.beat_division_factors, drum_mapping=tgt_drum_mapping)

        # Copy the tempo and time signature fields to new hvo
        for tempo in self.tempos:
            hvo_seq_tgt.add_tempo(tempo.time_step, tempo.qpm)
        for ts in self.time_signatures:
            hvo_seq_tgt.add_time_signature(ts.time_step, ts.numerator, ts.denominator)

        hvo_tgt = self.get_with_different_drum_mapping("hvo", tgt_drum_mapping=tgt_drum_mapping)
        hvo_seq_tgt.hvo = hvo_tgt

        return hvo_seq_tgt

    #   --------------------------------------------------------------
    #   Utilities to Synthesize the hvo score
    #   --------------------------------------------------------------

    def synthesize(self, sr=44100, sf_path="../hvo_sequence/soundfonts/Standard_Drum_Kit.sf2"):
        """
        Synthesizes the hvo_sequence to audio using a provided sound font
        @param sr:                          sample rate
        @param sf_path:                     path to the soundfont samples
        @return:                            synthesized audio sequence (if fluidsynth is installed
                                                otherwise 1 sec of silence)
        """

        if self.is_ready_for_use() is False:
            return None

        if _CAN_SYNTHESIZE and _HAS_NOTE_SEQ:
            ns = self.to_note_sequence(midi_track_n=9)
            pm = note_seq.note_sequence_to_pretty_midi(ns)
            audio = pm.fluidsynth(fs=sr, sf2_path=sf_path)
        else:
            audio = [0.0]*44100
            print("Generating 1 sec of Silence!! "
                  "Please install note_seq and fluidsynth packages to synthesize correctly")
        return audio

    def save_audio(self, filename="misc/temp.wav", sr=44100,
                   sf_path="../hvo_sequence/soundfonts/Standard_Drum_Kit.sf2"):
        """
        Synthesizes and saves the hvo_sequence to audio using a provided sound font
        @param filename:                    filename/path used for saving the audio
        @param sr:                          sample rate
        @param sf_path:                     path to the soundfont samples
        @return:                            synthesized audio sequence
        """

        if self.is_ready_for_use() is False:
            return None

        if _CAN_SYNTHESIZE and _HAS_NOTE_SEQ:
            ns = self.to_note_sequence(midi_track_n=9)
            pm = note_seq.note_sequence_to_pretty_midi(ns)
            audio = pm.fluidsynth(sf2_path=sf_path, fs=sr)
            # save audio using scipy
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            wavfile.write(filename, sr, audio)
        else:
            audio = [0.0]*44100
            print("Generating 1 sec of Silence!! "
                  "Please install note_seq and fluidsynth packages to synthesize correctly")
        return audio

    #   --------------------------------------------------------------
    #   Utilities to plot the score
    #   --------------------------------------------------------------
    def to_html_plot(self, filename="misc/temp.html", show_figure=False,
                     save_figure=False,
                     show_tempo=True, 
                     show_time_signature=True, 
                     show_metadata=True,
                     minor_grid_color="black", minor_line_width=0.1,
                     major_grid_color="black", major_line_width=0.5,
                     downbeat_color="black", downbeat_line_width=2,
                     note_color="grey",
                     width=800, height=400):
        """**DEPRECIATED** Use piano_roll to plot the hvo_sequence instead """

        return self.piano_roll(
            filename=filename, show_figure=show_figure, save_figure=save_figure,
            show_tempo=show_tempo, 
            show_time_signature=show_time_signature, 
            show_metadata=show_metadata, minor_grid_color=minor_grid_color, minor_line_width=minor_line_width,
            major_grid_color=major_grid_color, major_line_width=major_line_width,
            downbeat_color=downbeat_color, downbeat_line_width=downbeat_line_width,
            note_color=note_color,
            width=width, height=height)

    def piano_roll(self, filename="misc/temp.html", show_figure=False,
                   save_figure=False,
                   show_tempo=True,
                   show_time_signature=True,
                   show_metadata=True,
                   minor_grid_color="black", minor_line_width=0.1,
                   major_grid_color="black", major_line_width=0.5,
                   downbeat_color="black", downbeat_line_width=2,
                   note_color="grey",
                   width=1000, height=400):
        assert self.drum_mapping is not None, "Drum mapping not set"

        if self.is_ready_for_use() is False:
            logger.warning(".hvo is initialized to 1 bar after the beginning of the last tempo or time sig change.")
            last_time_sig = sorted(self.grid_maker.time_signatures, key=lambda x: x.time_step)[-1]
            last_tempo = sorted(self.grid_maker.tempos, key=lambda x: x.time_step)[-1]
            time = max(last_time_sig.time_step, last_tempo.time_step)
            n_steps = time + last_time_sig.numerator * self.grid_maker.n_steps_per_beat
            self.zeros(n_steps)

        notes = self.get_notes(return_tuples=False)
        colors = viridis(15)

        _html_fig = figure(plot_width=width, plot_height=height, y_range=(-0.5, len(self.drum_mapping)+1.75))
        _html_fig.xaxis.axis_label = 'Time (sec)'
        _html_fig.yaxis.axis_label = 'Instrument'

        legend_it = list()

        if len(notes['start']) > 0:
            notes['top'] = np.array(notes["voice_index"]) + 0.2
            notes['bottom'] = np.array(notes["voice_index"]) - 0.2
            p_roll = _html_fig.quad(top='top', bottom='bottom', left='start', right='end',
                                    line_color=note_color, fill_color=note_color,
                                    fill_alpha='velocity', source=notes, legend_label="Piano Roll")
            legend_it.append(("Notes", [p_roll]))
            notes.pop('top')
            notes.pop('bottom')
            _html_fig.add_tools(
                HoverTool(tooltips=[(f"{k}", f"@{k}") for k in notes.keys()], renderers=[p_roll]))
        else:
            legend_it.append(("Notes", []))
        _html_fig.title.text = filename.split("/")[-1]  # add title

        # Add y-labels corresponding to instrument names rather than midi note ("kick", "snare", ...)
        unique_pitches = []
        drum_tags = []
        for j, k in enumerate(self.drum_mapping.keys()):
            unique_pitches.append(j)
            drum_tags.append(k)
        # if len(unique_pitches) == 0:
        #     unique_pitches = [0]
        #     drum_tags = ["None"]

        _html_fig.xgrid.grid_line_color = None
        _html_fig.ygrid.grid_line_color = None

        _html_fig.yaxis.ticker = list(unique_pitches)
        _html_fig.yaxis.major_label_overrides = dict(zip(unique_pitches, drum_tags))

        # Add beat and beat_division grid lines
        major_grid_lines = self.__grid_maker.get_major_grid_lines(self.number_of_steps)
        minor_grid_lines = self.__grid_maker.get_minor_grid_lines(self.number_of_steps)

        minor_grid_ = []
        for t in minor_grid_lines:
            minor_grid_.append(Span(location=t, dimension='height',
                                    line_color=minor_grid_color, line_width=minor_line_width))
            _html_fig.add_layout(minor_grid_[-1])

        _html_fig.xaxis.ticker = [np.round(x, 2) for x in major_grid_lines]
        _html_fig.xaxis.major_label_orientation = 1.57

        major_grid_ = []
        for t in major_grid_lines:
            major_grid_.append(Span(location=t, dimension='height',
                                    line_color=major_grid_color, line_width=major_line_width))
            _html_fig.add_layout(major_grid_[-1])

        downbeat_grid_ = []
        for t in self.__grid_maker.get_downbeat_grid_lines(self.number_of_steps):
            downbeat_grid_.append(Span(location=t, dimension='height',
                                       line_color=downbeat_color, line_width=downbeat_line_width))
            _html_fig.add_layout(downbeat_grid_[-1])

        if self.number_of_steps > 0:
            grid = self.__grid_maker.get_grid_lines(self.number_of_steps)
        else:
            grid = self.__grid_maker.get_grid_lines_for_n_beats(2)

        if show_tempo:
            tempo_dict = {"x": [], "y": [], "tempo": [], "grid_index": []}
            for ix, tempo in enumerate(self.tempos):
                if tempo.time_step <= len(grid):
                    tempo_dict["x"].append(grid[tempo.time_step])
                    tempo_dict["y"].append(unique_pitches[-1] + 1.5)
                    tempo_dict["tempo"].append(tempo.qpm)
                    tempo_dict["grid_index"].append(tempo.time_step)
            temp = _html_fig.circle(x="x", y="y", source=tempo_dict, size=10,
                                    line_color=colors[0], fill_color=colors[0], legend_label="Tempo")

            tempo_dict.pop("x")
            tempo_dict.pop("y")
            _html_fig.add_tools(
                HoverTool(tooltips=[(f"{k}", f"@{k}") for k in tempo_dict.keys()], renderers=[temp]))

        if show_time_signature:
            time_signature_dict = {"x": [], "y": [], "Numerator": [], "Denominator": [], "grid_index": []}
            for ix, ts in enumerate(self.time_signatures):
                if ts.time_step <= len(grid):
                    time_signature_dict["x"].append(grid[ts.time_step])
                    time_signature_dict["y"].append(unique_pitches[-1] + 1)
                    time_signature_dict["Numerator"].append(ts.numerator)
                    time_signature_dict["Denominator"].append(ts.denominator)
                    time_signature_dict["grid_index"].append(ts.time_step)

            temp = _html_fig.circle(x="x", y="y", source=time_signature_dict, size=10,
                                    line_color=colors[-4], fill_color=colors[-4], legend_label="Time Signature")

            time_signature_dict.pop("x")
            time_signature_dict.pop("y")
            _html_fig.add_tools(
                HoverTool(tooltips=[(f"{k}", f"@{k}") for k in time_signature_dict.keys()], renderers=[temp]))

        if show_metadata:
            if self.metadata.keys() is not None:
                metadata = {
                    'x': [grid[ix] for ix in self.metadata.time_steps],
                    'y': [list(unique_pitches)[-1] + 0.5] * len(self.metadata.time_steps)
                }
                metadata.update({k: [] for k, v in self.metadata.items()})
                for k, v in self.metadata.items():
                    metadata[k].extend(v if isinstance(v, list) else [v])
                temp = _html_fig.circle(x="x", y="y", source=metadata, size=10,
                                        line_color=colors[8], fill_color=colors[8], legend_label="Metadata")
                _html_fig.add_tools(
                    HoverTool(tooltips=[(f"{k}", f"@{k}") for k in self.metadata.keys()], renderers=[temp]))

        h = self.__grid_maker.get_grid_lines(self.number_of_steps)[-1]

        _html_fig.x_range.start = h * -0.1
        _html_fig.x_range.end = h * 1.1

        _html_fig.legend.click_policy = "hide"
        leg = _html_fig.legend[0]
        _html_fig.add_layout(leg, 'right')

        # create a row
        # Plot the figure if requested
        if show_figure:
            show(_html_fig)

        # Save the plot
        if save_figure:
            if not filename.endswith(".html"):
                filename += ".html"
            if os.path.dirname(filename) != "":
                os.makedirs(os.path.dirname(filename), exist_ok=True)

            output_file(filename)  # Set name used for saving the figure
            save(_html_fig)  # Save to file
            logger.info(f"Saved HTML piano roll to {filename}")

        return _html_fig

    #   --------------------------------------------------------------
    #   Utilities to compute, plot and save Spectrogram
    #   --------------------------------------------------------------
    def stft(self, sr=44100, sf_path="../hvo_sequence/soundfonts/Standard_Drum_Kit.sf2", n_fft=2048, hop_length=128,
             win_length=1024, window='hamming', plot=False, plot_filename="misc/temp_spec.png", plot_title="STFT",
             width=800, height=400, font_size=12, colorbar=False):

        """
        Computes the Short-time Fourier transform.
        @param sr:                          sample rate of the audio file from which the STFT is computed
        @param sf_path:                     path to the soundfont samples
        @param n_fft:                       length of the windowed signal after padding to closest power of 2
        @param hop_length:                  number of samples between successive STFT frames
        @param win_length:                  window length in samples. must be equal or smaller than n_fft
        @param window:                      window type specification (see scipy.signal.get_window) or function
        @param plot                         if True, plots and saves plot
        @param plot_filename:               filename for saved figure
        @param plot_title:                  plot title
        @param width:                       figure width in pixels
        @param height:                      figure height in pixels
        @param font_size:                   font size in pt
        @param colorbar:                    if True, display colorbar
        @return:                            STFT ndarray
        """
        if not _HAS_LIBROSA:
            logger.warning("Librosa is not installed. Please install it to use the logf-stft feature.")
            return None

        if self.is_ready_for_use() is False:
            return None

        # Check inputs
        if not win_length <= n_fft:
            logger.warning("Window size must be equal or smaller than FFT size.")
            return None

        if not hop_length > 0:
            logger.warning("Hop size must be greater than 0.")
            return None

        # Get audio signal
        y = self.save_audio(sr=sr, sf_path=sf_path)

        # Get STFT
        sy = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window)
        stft = np.abs(sy)

        if plot and _HAS_MATPLOTLIB:
            # Plot STFT
            # Plot params
            plt.rcParams['font.size'] = font_size

            px = 1 / plt.rcParams['figure.dpi']  # pixel to inch conversion factor
            [width_i, height_i] = [width * px, height * px]  # width and height in inches

            plt.rcParams.update({'figure.autolayout': True})  # figure layout
            plt.tight_layout()

            # Plot spectogram and save
            fig, ax = plt.subplots(figsize=(width_i, height_i))
            ax.set_title(plot_title)

            spec = librosa.display.specshow(librosa.amplitude_to_db(stft, ref=np.max), 
                                            y_axis='log', x_axis='time', ax=ax)

            if colorbar:
                fig.colorbar(spec, ax=ax, format="%+2.0f dB")

            fig.savefig(plot_filename)

        return stft

    def mel_spectrogram(self, sr=44100, sf_path="../hvo_sequence/soundfonts/Standard_Drum_Kit.sf2", n_fft=2048,
                        hop_length=128, win_length=1024, window='hamming', n_mels=24, fmin=0, fmax=22050, plot=False,
                        plot_filename="misc/temp_mel_spec.png", plot_title="'Mel-frequency spectrogram'", width=800,
                        height=400, font_size=12, colorbar=False):

        """
        Computes mel spectrogram of the synthesized version of the .hvo score
        @param sr:                          sample rate of the audio file from which the STFT is computed
        @param sf_path:                     path to the soundfont samples
        @param n_fft:                       length of the windowed signal after padding to closest power of 2
        @param hop_length:                  number of samples between successive STFT frames
        @param win_length:                  window length in samples. must be equal or smaller than n_fft
        @param window:                      window type specification (see scipy.signal.get_window) or function
        @param n_mels:                      number of mel bands
        @param fmin:                        lowest frequency in Hz
        @param fmax:                        highest frequency in Hz
        @param plot                         if True, plots and saves plot
        @param plot_filename:               filename for saved figure
        @param plot_title:                  plot title
        @param width:                       figure width in pixels
        @param height:                      figure height in pixels
        @param font_size:                   font size in pt
        @param colorbar:                    if True, display colorbar
        @return:                            mel spectrogram ndarray
        """

        if not _HAS_LIBROSA:
            logger.warning("Librosa is not installed. Please install it to use the logf-stft feature.")
            return None

        if self.is_ready_for_use() is False:
            return None

        # Check inputs
        if not win_length <= n_fft:
            logger.warning("Window size must be equal or smaller than FFT size.")
            return None

        if not hop_length > 0:
            logger.warning("Hop size must be greater than 0.")
            return None

        if not n_mels > 0:
            logger.warning("Number of mel bands must be greater than 0.")
            return None

        if not fmin >= 0 or not fmax > 0:
            logger.warning("Frequency must be greater than 0.")
            return None

        # Get audio signal
        y = self.save_audio(sr=sr, sf_path=sf_path)

        # Get mel spectrogram
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
                                                  window=window, n_mels=n_mels, fmin=fmin, fmax=fmax)

        if plot:
            # Plot mel spectrogram
            # Plot specs
            plt.rcParams['font.size'] = font_size

            px = 1 / plt.rcParams['figure.dpi']  # pixel to inch conversion factor
            [width_i, height_i] = [width * px, height * px]  # width and height in inches

            plt.rcParams.update({'figure.autolayout': True})  # figure layout
            plt.tight_layout()

            # Plot spectogram and save
            fig, ax = plt.subplots(figsize=(width_i, height_i))
            ax.set_title(plot_title)

            spec = librosa.display.specshow(librosa.power_to_db(mel_spec, ref=np.max), 
                                            y_axis='mel', x_axis='time', ax=ax)

            if colorbar:
                fig.colorbar(spec, ax=ax, format="%+2.0f dB")

            fig.savefig(plot_filename)

        return mel_spec

    #   -------------------------------------------------------------
    #   MSO::Multiband Synthesized Onsets
    #   -------------------------------------------------------------
    def get_logf_stft(self, **kwargs):
        """calculates the log-frequency STFT of the synthesized version of the .hvo score"""
        sf_path = kwargs.get('sf_path', "soundfonts/Standard_Drum_Kit.sf2")
        sr = kwargs.get('sr', 44100)
        n_fft = kwargs.get('n_fft', 1024)
        win_length = kwargs.get('win_length', 1024)
        hop_length = kwargs.get('hop_length', 512)
        n_bins_per_octave = kwargs.get('n_bins_per_octave', 16)
        n_octaves = kwargs.get('n_octaves', 9)
        f_min = kwargs.get('f_min', 40)
        # mean_filter_size = kwargs.get('mean_filter_size', 22)

        # audio
        y = self.synthesize(sr=sr, sf_path=sf_path)
        y /= np.max(np.abs(y))

        mX, f_bins = logf_stft(y, n_fft, win_length, hop_length, n_bins_per_octave, n_octaves, f_min, sr)

        return mX, f_bins

    def get_onset_strength_spec(self, **kwargs):
        """calculates the onset strength spectrogram of the synthesized version of the .hvo score"""
        sf_path = kwargs.get('sf_path', "soundfonts/Standard_Drum_Kit.sf2")
        sr = kwargs.get('sr', 44100)
        n_fft = kwargs.get('n_fft', 1024)
        win_length = kwargs.get('win_length', 1024)
        hop_length = kwargs.get('hop_length', 512)
        n_bins_per_octave = kwargs.get('n_bins_per_octave', 16)
        n_octaves = kwargs.get('n_octaves', 9)
        f_min = kwargs.get('f_min', 40)
        mean_filter_size = kwargs.get('mean_filter_size', 22)

        # audio
        y = self.synthesize(sr=sr, sf_path=sf_path)
        y /= np.max(np.abs(y))

        # onset strength spectrogram
        spec, f_cq = onset_strength_spec(y, n_fft, win_length, hop_length, n_bins_per_octave, n_octaves, f_min, sr,
                                         mean_filter_size)

        return spec, f_cq

    def mso(self, **kwargs):
        """calculates the Multi-band synthesized onsets."""
        sf_path = kwargs.get('sf_path', "soundfonts/Standard_Drum_Kit.sf2")
        sr = kwargs.get('sr', 44100)
        n_fft = kwargs.get('n_fft', 1024)
        win_length = kwargs.get('win_length', 1024)
        hop_length = kwargs.get('hop_length', 512)
        n_bins_per_octave = kwargs.get('n_bins_per_octave', 16)
        n_octaves = kwargs.get('n_octaves', 9)
        f_min = kwargs.get('f_min', 40)
        mean_filter_size = kwargs.get('mean_filter_size', 22)
        c_freq = kwargs.get('c_freq', [55, 90, 138, 175, 350, 6000, 8500, 12500])

        # onset strength spectrogram
        spec, f_cq = self.get_onset_strength_spec(sf_path=sf_path, n_fft=n_fft, win_length=win_length,
                                                  hop_length=hop_length, n_bins_per_octave=n_bins_per_octave,
                                                  n_octaves=n_octaves, f_min=f_min, sr=sr,
                                                  mean_filter_size=mean_filter_size)

        # multi-band onset detection and strength
        mb_onset_strength = reduce_f_bands_in_spec(c_freq, f_cq, spec)
        mb_onset_detect = detect_onset(mb_onset_strength)

        # map to grid
        grid = np.array(self.__grid_maker.get_grid_lines(self.number_of_steps))
        strength_grid, onsets_grid = map_onsets_to_grid(grid, mb_onset_strength, mb_onset_detect, n_fft=n_fft,
                                                        hop_length=hop_length, sr=sr)

        # concatenate in one single array
        mso = np.concatenate((strength_grid, onsets_grid), axis=1)

        return mso

    # ######################################################################
    #
    #           Rhythmic Features::Statistical Features Related
    #
    # ######################################################################

    def get_number_of_active_voices(self):
        """gets total number of active instruments in patter"""
        h = self.hits
        h = np.where(h == 0.0, np.nan, h)
        noi = 0
        for voice_ix in range(self.number_of_voices):
            # check if voice part is empty
            if all(np.isnan(h[:, voice_ix])) is not True:
                noi += 1
        return noi

    def get_total_step_density(self):
        """ calculates the ratio of total steps in which there is at least one hit """
        hits = self.hits
        return np.clip(np.count_nonzero(hits, axis=1), 0, 1).sum()/self.number_of_steps

    def get_average_voice_density(self):
        """ average of number of instruments divided by total number of voices over all steps """
        hits = self.hits
        return np.count_nonzero(hits, axis=1).sum()/hits.size

    def get_hit_density_for_voice(self, voice_ix):
        """ calculates the ratio of steps in which there is at least one hit for a given voice """
        return np.count_nonzero(self.hits[:, voice_ix]) / self.number_of_steps

    def get_velocity_intensity_mean_stdev_for_voice(self, voice_ix):
        """Calculates mean and std of velocities for a single voice.
        first gets all non-zero hits. then divide by number of hits"""
        if self.is_ready_for_use() is False:
            return None
        v = self.get("v", use_nan_for_non_hits=True)[:, voice_ix]
        if all(np.isnan(v)) is True:
            return 0, 0
        else:
            return np.nanmean(v), np.nanstd(v)

    def get_offset_mean_stdev_for_voice(self, voice_ix, offsets_in_ms=False):
        """Calculates mean and std of offsets for a single voice.
        first gets all non-zero hits. then divide by number of hits"""
        if self.is_ready_for_use() is False:
            return None
        o = self.get("o", offsets_in_ms=offsets_in_ms, use_nan_for_non_hits=True)[:, voice_ix]
        if all(np.isnan(o)) is True:
            return 0, 0
        else:
            return np.nanmean(o), np.nanstd(o)

    def get_lowness_midness_hiness(self, low_mid_hi_drum_map=Groove_Toolbox_3Part_keymap):
        """
        "Share of the total density of patterns that belongs to each
        of the different instrument categories. Computed as the
        quotient between the densities per instrument category
        and the total density" [2]

        [2] Drum rhythm spaces: From polyphonic similarity to generative maps by Daniel Gomez Marin, 2020
        """
        lmh_hits = self.get_with_different_drum_mapping("h", tgt_drum_mapping=low_mid_hi_drum_map)
        total_hits = np.count_nonzero(self.hits)
        lowness = np.count_nonzero(lmh_hits[:, 0])/total_hits if total_hits != 0 else 0
        midness = np.count_nonzero(lmh_hits[:, 1])/total_hits if total_hits != 0 else 0
        hiness = np.count_nonzero(lmh_hits[:, 2])/total_hits if total_hits != 0 else 0
        return lowness, midness, hiness

    def get_velocity_score_symmetry(self):
        """Get total symmetry of pattern. Defined as the number of onsets that appear in the same positions in the first
        and second halves of the pattern, divided by total number of onsets in the pattern.
        symmetry is calculated using velocity section of hvo"""

        # fixme the implementation doesnt match the description

        if self.is_ready_for_use() is False:
            return None

        v = self.get("v", use_nan_for_non_hits=True)
        assert v.shape[0] % 2 == 0, "symmetry can't be calculated as the length of score needs to be a multiple of 2"

        # Find difference between splits
        part1, part2 = np.split(v, 2)
        diff = np.abs(part1 - part2)
        diff = diff[~np.isnan(diff)]    # Remove non hit locations (denoted with np.nan)
        # get symmetry level
        symmetry_level = (1 - diff)

        res = np.nanmean(symmetry_level)
        res = 0 if np.isnan(res) else res
        return res

    def get_total_weak_to_strong_ratio(self):
        """ returns the ratio of total weak onsets divided by all strong onsets.
        Strong onsets are onsets that occur on beat positions and weak onsets are the other ones
        """
        # todo easily adaptable to alternative grids if implementation is changed

        return get_weak_to_strong_ratio(self.get("v"))

    def get_polyphonic_velocity_mean_stdev(self):
        """Get average loudness for any single part or group of parts. Will return 1 for binary loop,
        otherwise calculate based on velocity mode chosen (transform or regular)"""

        # first get all non-zero hits. then divide by number of hits
        if self.is_ready_for_use() is False:
            return None
        v = self.get("v", use_nan_for_non_hits=True)
        if all(np.isnan(v.flatten())) is True:
            return 0, 0
        else:
            return np.nanmean(v), np.nanstd(v)

    def get_polyphonic_offset_mean_stdev(self, offsets_in_ms=False):
        """Get average loudness for any single part or group of parts. Will return 1 for binary loop, otherwise calculate
        based on velocity mode chosen (transform or regular)"""

        # first get all non-zero hits. then divide by number of hits
        if self.is_ready_for_use() is False:
            return None
        o = self.get("o", offsets_in_ms=offsets_in_ms, use_nan_for_non_hits=True)
        if all(np.isnan(o.flatten())) is True:
            return 0, 0
        else:
            return np.nanmean(o), np.nanstd(o)

    # ######################################################################
    #   Rhythmic Features::Syncopation from GrooveToolbox
    #
    #        The following code is mostly from the GrooveToolbox
    #              https://github.com/fredbru/GrooveToolbox
    #        Some additional functions have been implemented here
    #         to adapt hvo_sequence representation to the groove and
    #            utiming representations used in the GrooveToolbox
    #
    # Reference:    Yang, Li-Chia, and Alexander Lerch. "On the evaluation
    #               of generative models in music." Neural Computing
    #               and Applications 32.9 (2020): 4773-4784.
    # ######################################################################

    def get_monophonic_syncopation_for_voice(self, voice_index):
        """Using Longuet-Higgins  and  Lee 1984 metric profile, get syncopation of 1 monophonic line.
        Assumes it's a drum loop - loops round.
        Normalized against maximum syncopation: syncopation score of pattern with all pulses of lowest metrical level
        at maximum amplitude (=30 for 2 bar 4/4 loop)"""

        # todo : use get_monophonic_syncopation_for_voice() to get syncopation of tapped pattern

        if self.is_ready_for_use() is True:
            assert self.time_signatures[0].denominator == 4, \
                "currently Can't calculate syncopation for patterns with multiple time signatures and " \
                "time signature denominators other than 4"
            if len(self.time_signatures) > 1:
                logger.warning("More than one time signatures: {}".format(self.time_signatures))
        else:
            return None

        metrical_profile = Longuet_Higgins_METRICAL_PROFILE_4_4_16th_NOTE

        part = self.hits[:, voice_index]

        return get_monophonic_syncopation(part, metrical_profile)

    def get_combined_syncopation(self):
        """Calculate syncopation as summed across all kit parts."""
        # Tested - working correctly (12/3/20)
        # todo: error of size mismatch here

        combined_syncopation = 0.0

        for i in range(self.number_of_voices):
            combined_syncopation += self.get_monophonic_syncopation_for_voice(voice_index=i)

        return combined_syncopation

    def get_witek_polyphonic_syncopation(self, low_mid_hi_drum_map=Groove_Toolbox_3Part_keymap):
        """Calculate syncopation using Witek syncopation distance - modelling syncopation between instruments
        Works on semiquaver and quaver levels of syncopation
        at maximum amplitude (=30 for 2 bar 4/4 loop)"""
        # todo: Normalize...?

        metrical_profile = WITEK_SYNCOPATION_METRICAL_PROFILE_4_4_16th_NOTE

        max_syncopation = 30.0

        # Get hits score reduced to low mid high groups
        lmh_hits = self.get_with_different_drum_mapping("h", tgt_drum_mapping=low_mid_hi_drum_map)
        low = lmh_hits[:, 0]
        mid = lmh_hits[:, 1]
        high = lmh_hits[:, 2]

        total_syncopation = 0

        for i in range(len(low)):
            kick_syncopation, snare_syncopation = _get_kick_and_snare_syncopations(low, mid, high, i, metrical_profile)
            total_syncopation += kick_syncopation * low[i]
            total_syncopation += snare_syncopation * mid[i]

        return total_syncopation / max_syncopation

    def get_low_mid_hi_syncopation_info(self, low_mid_hi_drum_map=Groove_Toolbox_3Part_keymap):
        """calculates monophonic syncopation of low/mid/high voice groups
        also weighted by their number of corresponding onsets

        Details of definitions here:
        [2] Drum rhythm spaces: From polyphonic similarity to generative maps by Daniel Gomez Marin, 2020
        :return: dictionary of all values calculated
                """

        # todo: error of size mismatch here
        metrical_profile = WITEK_SYNCOPATION_METRICAL_PROFILE_4_4_16th_NOTE

        # Get hits score reduced to low mid high groups
        lmh_hits = self.get_with_different_drum_mapping("h", tgt_drum_mapping=low_mid_hi_drum_map)

        lowsync = get_monophonic_syncopation(lmh_hits[:, 0], metrical_profile)
        midsync = get_monophonic_syncopation(lmh_hits[:, 1], metrical_profile)
        hisync = get_monophonic_syncopation(lmh_hits[:, 2], metrical_profile)
        lowsyness = (lowsync / np.count_nonzero(lmh_hits[:, 0])) if np.count_nonzero(lmh_hits[:, 0]) > 0 else 0
        midsyness = (midsync / np.count_nonzero(lmh_hits[:, 1])) if np.count_nonzero(lmh_hits[:, 1]) > 0 else 0
        hisyness = (hisync / np.count_nonzero(lmh_hits[:, 2])) if np.count_nonzero(lmh_hits[:, 2]) > 0 else 0

        return {"lowsync": lowsync, "midsync": midsync, "hisync": hisync,
                "lowsyness": lowsyness, "midsyness": midsyness, "hisyness": hisyness}

    def get_complexity_for_voice(self, voice_index):
        """ Calculated following Sioros and Guedes (2011) as
        combination of density and syncopation"""

        hit_density = self.get_hit_density_for_voice(voice_index)
        sync = self.get_monophonic_syncopation_for_voice(voice_index)
        return math.sqrt(pow(sync, 2) + pow(hit_density, 2))

    def get_total_complexity(self):
        """Find Source!!!???"""
        # todo: error of size mismatch here
        hit_density = self.get_total_step_density()
        sync = self.get_combined_syncopation()
        return math.sqrt(pow(sync, 2) + pow(hit_density, 2))

    # ######################################################################
        #      Rhythmic Features::Autocorrelation Related from GrooveToolbox
    #
    #        The following code is mostly from the GrooveToolbox
    #              https://github.com/fredbru/GrooveToolbox
    #        Some additional functions have been implemented here
    #         to adapt hvo_sequence representation to the groove and
    #            utiming representations used in the GrooveToolbox
    #
    # Reference:    Yang, Li-Chia, and Alexander Lerch. "On the evaluation
    #               of generative models in music." Neural Computing
    #               and Applications 32.9 (2020): 4773-4784.
    # ######################################################################

    def get_total_autocorrelation_curve(self, hvo_str="v", offsets_in_ms=False):
        """ Returns the autocorrelation of hvo score (according to hvo_str)
        the autocorrelation is calculated per voice and then added up per step

        :param hvo_str: str
            String formed with the characters 'h', 'v', 'o' and '0' in any order. It's not necessary to use all of the
            characters and they can be repeated. E.g. 'ov', will return the offsets and velocities, 'h0h'
            set offsets_in_ms to True if 'o' should be in milliseconds
        :param offsets_in_ms: bool
            If True, the offsets will be returned in milliseconds instead of steps
        :return:
            autocorrelation curve for all parts summed.
        """

        if self.is_ready_for_use() is False:
            return None

        def autocorrelation(x):
            result = np.correlate(x, x, mode='full')
            return result[result.size // 2:]

        score = self.get(hvo_str=hvo_str, offsets_in_ms=offsets_in_ms)

        total_autocorrelation_curve = 0.0
        for i in range(score.shape[1]):
            total_autocorrelation_curve = total_autocorrelation_curve + autocorrelation(score[:, i])

        return total_autocorrelation_curve

    def get_velocity_autocorrelation_features(self):
        """Calculate autocorrelation curve"""

        if self.is_ready_for_use() is False:
            return None

        acorr = self.get_total_autocorrelation_curve(hvo_str="v")
        vels = self.get("v")

        # Create an empty dict to store features
        autocorrelation_features = dict()

        # Feature 1:
        # Calculate skewness of autocorrelation curve
        autocorrelation_features["skewness"] = skew(acorr)

        # Feature 2:
        # Maximum of autocorrelation curve
        autocorrelation_features["max"] = acorr.max()

        # Feature 3:
        # Calculate acorr centroid Like spectral centroid - weighted mean of frequencies
        # in the signal, magnitude = weights.
        centroid_sum = 0
        total_weights = 0

        for i in range(acorr.shape[0]):
            # half wave rectify
            addition = acorr[i] * i  # sum all periodicity's in the signal
            if addition >= 0:
                total_weights += acorr[i]
                centroid_sum += addition

        if total_weights != 0:
            autocorrelation_centroid = centroid_sum / total_weights
        else:
            autocorrelation_centroid = vels.shape[0] / 2
        autocorrelation_features["centroid"] = autocorrelation_centroid

        # Feature 4:
        # Autocorrelation Harmonicity adapted from Lartillot et al. 2008
        total_autocorrelation_curve = acorr

        alpha = 0.15
        rectified_autocorrelation = acorr
        for i in range(total_autocorrelation_curve.shape[0]):
            if total_autocorrelation_curve[i] < 0:
                rectified_autocorrelation[i] = 0

        # weird syntax due to 2.x/3.x compatibility issues here todo: rewrite for 3.x
        peaks = np.asarray(find_peaks(rectified_autocorrelation))
        peaks = peaks[0] + 1  # peaks = lags

        inharmonic_sum = 0.0
        inharmonic_peaks = []
        for i in range(len(peaks)):
            remainder1 = 16 % peaks[i]
            if (16 * alpha) < remainder1 < (16 * (1 - alpha)):
                inharmonic_sum += rectified_autocorrelation[peaks[i] - 1]  # add magnitude of inharmonic peaks
                inharmonic_peaks.append(rectified_autocorrelation[i])

        if float(rectified_autocorrelation.max()) != 0:
            harmonicity = math.exp((-0.25 * len(peaks) * inharmonic_sum / float(rectified_autocorrelation.max())))
        else:
            harmonicity = np.nan
        autocorrelation_features["harmonicity"] = harmonicity
        return autocorrelation_features

    # ######################################################################
    #               Micro-timing Features from GrooveToolbox
    #
    #        The following code is mostly from the GrooveToolbox
    #              https://github.com/fredbru/GrooveToolbox
    #        Some additional functions have been implemented here
    #         to adapt hvo_sequence representation to the groove and
    #            utiming representations used in the GrooveToolbox
    #
    # Reference:    Yang, Li-Chia, and Alexander Lerch. "On the evaluation
    #               of generative models in music." Neural Computing
    #               and Applications 32.9 (2020): 4773-4784.
    # ######################################################################

    def swingness(self, mode=1):
        """ Two modes here for calculating Swing implemented

        algorithm 1: same as groovetoolbox
            "Swung onsets are detected as significantly delayed second eighth-notes, approximating the typically
            understood 2:1 eighth-note swing ratio. Although musically these are considered as eighth notes,
            they fall into sixteenth note positions when quantized,
            with !!!significant negative (ahead of the position) deviations.!!!
            The ‘swingness’ feature first records whether these timing deviations occur or not,
            returning 0 for no swing or 1 for swing. This is then weighted by the number of swung onsets to model
            perceptual salience of the swing." [1]


        [1] Bruford, Fred, et al. "Multidimensional similarity modelling of complex drum loops
                                using the GrooveToolbox." (2020): 263-270.

        algorithm 2: Based on Ableton 16th note swing definitions
            The implementation here is slightly different!
            here we look out the utiming_ratio (not in ms) at each second and fourth time step within each
            quantization time_grid. (similar to 16th note swings in ableton and many other DAWs/Drum Machines)
            In this context, a +.5 swing at 2nd or 4th time steps means that the note is 100 percent swung
            while a value of <=0 means no swing
            In other words, we measure average of delayed timings on 2and 4th time steps in each beat (i.e. grid(1::2))

            maximum swing in this method is

        :param mode: int
                         0 --> groovetoolbox method
                         1 --> similar to DAW Swing

        :return: a tuple of swingness

        """

        if self.is_ready_for_use() is False:
            return None

        if len(self.time_signatures) > 1:
            logger.warning("Currently doesn't support multiple time signatures. Received: {}".format(
                self.time_signatures))

        assert self.grid_maker.beat_division_factors == [4] and self.time_signatures[0].denominator == 4, \
            "Currently Swing calculation can only be done for binary grids with time signature denominator of 4"

        if mode == 0:
            # get micro-timings in ms and use np.nan whenever no note is played
            microtiming_matrix = self.get("o", offsets_in_ms=True, use_nan_for_non_hits=True)
            n_steps = microtiming_matrix.shape[0]

            # Calculate average_timing_matrix
            average_timings_per_step = np.array([])

            # Get the mean of micro-timings per step
            for timings_at_step in microtiming_matrix:
                average_timings_per_step = np.append(
                    average_timings_per_step,
                    np.nanmean(timings_at_step) if not np.all(np.isnan(timings_at_step)) else 0
                )

            # get indices for swing positions = delayed 8th notes (timing on [fourth step, 8th, ..] 16th note grid)
            swung_note_positions = list(range(n_steps))[3::4]

            swing_count = 0.0
            j = 0
            for i in swung_note_positions:
                if average_timings_per_step[i] < -25.0:
                    swing_count += 1
                j += 1

            swing_count = np.clip(swing_count, 0, len(swung_note_positions))

            if swing_count > 0:
                swingness = (1 + (swing_count / len(swung_note_positions) / 10))  # todo: weight swing count
            else:
                swingness = 0.0

            return swingness

        elif mode == 1:
            # Get offsets at required swing steps
            microtiming_matrix = self.get("o", offsets_in_ms=False, use_nan_for_non_hits=False)

            # look at offsets at 2nd, 4th steps in each beat (corresponding to  grid line indices 1, 3, 5, 7, ... )
            offset_at_swing_steps = microtiming_matrix[1::2, :]

            # get average of positive offsets at swing steps
            offset_at_swing_steps = offset_at_swing_steps[offset_at_swing_steps > 0]

            # return mean of swung offsets or zero if none
            # max swing should be 1 (but max offset is 0.5) hence mult by 2
            return offset_at_swing_steps.mean()*2 if offset_at_swing_steps.size > 0 else 0

    def laidbackness(self,
                     kick_key_in_drum_mapping="KICK",
                     snare_key_in_drum_mapping="SNARE",
                     hihat_key_in_drum_mapping="HH_CLOSED",
                     threshold=12.0):

        """ Calculates how 'pushed' (or laidback) the loop is, based on number of pushed events /
        number of possible pushed events

        pushedness or laidbackness are calculated by looking at the timing of kick/snare/hat combinations with respect
        to each other

        :param kick_key_in_drum_mapping:
        :param snare_key_in_drum_mapping:
        :param hihat_key_in_drum_mapping:
        :param threshold:
        :return: laidbackness - pushedness
        """

        if self.is_ready_for_use() is False:
            return None

        if len(self.time_signatures) > 1:
            logger.warning("Currently doesn't support multiple time signatures. Received: {}".format(
                self.time_signatures))

        assert self.grid_maker.beat_division_factors == [4] and self.time_signatures[0].denominator == 4, \
            "Currently  calculation can only be done for binary grids with time signature denominator of 4"

        # Get micro-timings in ms
        microtiming_matrix = self.get("o", offsets_in_ms=True)

        n_bars = int(np.ceil(microtiming_matrix.shape[0] / 16))

        microtiming_event_profile, microtiming_event_profile_in_bar = None, None
        for bar_n in range(n_bars):
            microtiming_event_profile_in_bar = self._getmicrotiming_event_profile_1bar(
                microtiming_matrix[bar_n:(bar_n + 1) * 16, :],
                kick_key_in_drum_mapping=kick_key_in_drum_mapping,
                snare_key_in_drum_mapping=snare_key_in_drum_mapping,
                hihat_key_in_drum_mapping=hihat_key_in_drum_mapping,
                threshold=threshold
            )
            if bar_n == 0:
                microtiming_event_profile = microtiming_event_profile_in_bar
            else:
                microtiming_event_profile = np.append(microtiming_event_profile, microtiming_event_profile_in_bar)

        # Calculate how 'pushed' the loop is, based on number of pushed events / number of possible pushed events
        push_events = microtiming_event_profile[1::2]
        push_event_count = np.count_nonzero(push_events)
        total_push_positions = push_events.shape[0]
        pushed_events = push_event_count / total_push_positions

        # Calculate how 'laid-back' the loop is,
        # based on the number of laid back events / number of possible laid back events
        laidback_events = microtiming_event_profile[0::2]
        laidback_event_count = np.count_nonzero(laidback_events)
        total_laidback_positions = laidback_events.shape[0]
        laidback_events = laidback_event_count / float(total_laidback_positions)

        return laidback_events - pushed_events

    def _getmicrotiming_event_profile_1bar(self,
                                           microtiming_matrix,
                                           kick_key_in_drum_mapping="KICK",
                                           snare_key_in_drum_mapping="SNARE",
                                           hihat_key_in_drum_mapping="HH_CLOSED",
                                           threshold=12.0):
        """ Same implementation as groovetoolbox
        :param microtiming_matrix:                  offsets matrix for maximum of 1 bar in 4/4
        :param kick_key_in_drum_mapping:
        :param snare_key_in_drum_mapping:
        :param hihat_key_in_drum_mapping:
        :return:
            microtiming_event_profile_1bar
        """

        if self.is_ready_for_use() is False:
            return None

        if len(self.time_signatures) > 1:
            logger.warning("Currently doesn't support multiple time signatures. Received: {}".format(
                self.time_signatures))

        assert self.grid_maker.beat_division_factors == [4] and self.time_signatures[0].denominator == 4, \
            "Currently Swing calculation can only be done for binary grids with time signature denominator of 4"

        # Ensure duration is 16 steps
        if microtiming_matrix.shape[0] < 16:
            pad_len = 16-microtiming_matrix.shape[0]
            for i in range(pad_len):
                microtiming_matrix = np.append(microtiming_matrix, np.zeros((1, microtiming_matrix.shape[1])))
        microtiming_matrix = microtiming_matrix[:16, :]

        # Get 2nd dimension indices corresponding to kick, snare and hi-hats using the keys provided
        kick_ix = list(self.drum_mapping.keys()).index(kick_key_in_drum_mapping)
        snare_ix = list(self.drum_mapping.keys()).index(snare_key_in_drum_mapping)
        chat_ix = list(self.drum_mapping.keys()).index(hihat_key_in_drum_mapping)

        microtiming_event_profile_1bar = _getmicrotiming_event_profile_1bar(
            microtiming_matrix, kick_ix, snare_ix, chat_ix, threshold)

        return microtiming_event_profile_1bar

    def get_timing_accuracy(self, offsets_in_ms=False):
        # Calculate timing accuracy of the loop
        # timing accuracy is defined as the sum of microtiming deviations on 8th note positions

        # Get micro-timings in ms
        microtiming_matrix = self.get("o", offsets_in_ms=offsets_in_ms, use_nan_for_non_hits=True)

        if all(np.isnan(microtiming_matrix.flatten())):
            return np.nan

        average_timing_matrix = np.nanmean(microtiming_matrix, axis=1)
        non_triplet_or_swung_positions = average_timing_matrix[0::2]
        timing_accuracy = np.nanmean(np.abs(non_triplet_or_swung_positions))

        return timing_accuracy

    # ######################################################################
    #                   Similarity/Distance Measures
    #
    #        The following code is partially from the GrooveToolbox
    #              https://github.com/fredbru/GrooveToolbox
    #        Some additional functions have been implemented here
    #         to adapt hvo_sequence representation to the groove and
    #            utiming representations used in the GrooveToolbox
    #
    # Reference:    Yang, Li-Chia, and Alexander Lerch. "On the evaluation
    #               of generative models in music." Neural Computing
    #               and Applications 32.9 (2020): 4773-4784.
    # ######################################################################

    def calculate_all_distances_with(self, hvo_seq_b):
        distances_dictionary = \
            {
                "l1_distance -hvo": self.calculate_l1_distance_with(hvo_seq_b),
                "l1_distance -h": self.calculate_l1_distance_with(hvo_seq_b, "h"),
                "l1_distance -v": self.calculate_l1_distance_with(hvo_seq_b, "v"),
                "l1_distance -o": self.calculate_l1_distance_with(hvo_seq_b, "o"),
                "l2_distance -hvo": self.calculate_l2_distance_with(hvo_seq_b),
                "l2_distance -h": self.calculate_l2_distance_with(hvo_seq_b, "h"),
                "l2_distance -v": self.calculate_l2_distance_with(hvo_seq_b, "v"),
                "l2_distance -o": self.calculate_l2_distance_with(hvo_seq_b, "o"),
                "cosine-distance": self.calculate_cosine_distance_with(hvo_seq_b),
                "cosine-similarity": self.calculate_cosine_similarity_with(hvo_seq_b),
                "hamming_distance -all_voices_not_weighted ": self.calculate_hamming_distance_with(
                    hvo_seq_b, reduction_map=None, beat_weighting=False),
                "hamming_distance -all_voices_weighted ": self.calculate_hamming_distance_with(
                    hvo_seq_b, reduction_map=None, beat_weighting=True),
                "hamming_distance -low_mid_hi_not_weighted ": self.calculate_hamming_distance_with(
                    hvo_seq_b, reduction_map=Groove_Toolbox_3Part_keymap, beat_weighting=False),
                "hamming_distance -low_mid_hi_weighted ": self.calculate_hamming_distance_with(
                    hvo_seq_b, reduction_map=Groove_Toolbox_3Part_keymap, beat_weighting=True),
                "hamming_distance -5partKit_not_weighted ": self.calculate_hamming_distance_with(
                    hvo_seq_b, reduction_map=Groove_Toolbox_5Part_keymap, beat_weighting=False),
                "hamming_distance -5partKit_weighted ": self.calculate_hamming_distance_with(
                    hvo_seq_b, reduction_map=Groove_Toolbox_5Part_keymap, beat_weighting=True),
                "fuzzy_hamming_distance-not_weighted": self.calculate_fuzzy_hamming_distance_with(
                    hvo_seq_b, beat_weighting=False),
                "fuzzy_hamming_distance-weighted": self.calculate_fuzzy_hamming_distance_with(
                    hvo_seq_b, beat_weighting=True),
                "structural_similarity-structural_similarity": self.calculate_structural_similarity_distance_with(
                    hvo_seq_b)
            }

        sorted_dict = {key: value for key, value in sorted(distances_dictionary.items())}

        return sorted_dict

    def calculate_l1_distance_with(self, hvo_seq_b, hvo_str="hvo"):
        """
        :param hvo_seq_b:   Sequence to find l1 norm of euclidean distance with
        :param hvo_str:     String formed with the characters 'h', 'v' and 'o' in any order. It's not necessary
                            to use all of the characters and they can be repeated. E.g. 'ov' or 'hvoh'
        :return:            l1 norm of euclidean distance with hvo_seq_b
        """
        if self.is_ready_for_use() is False or hvo_seq_b.is_ready_for_use() is False:
            return None

        a = self.get(hvo_str).flatten()
        b = hvo_seq_b.get(hvo_str).flatten()

        return np.linalg.norm((a - b), ord=1)

    def calculate_l2_distance_with(self, hvo_seq_b, hvo_str="hvo"):
        """
        :param hvo_seq_b:   Sequence to find l2 norm of euclidean distance with
        :param hvo_str:     String formed with the characters 'h', 'v' and 'o' in any order. It's not necessary
                            to use all of the characters and they can be repeated. E.g. 'ov' or 'hvoh'

        :return:            l2 norm of euclidean distance with hvo_seq_b
        """

        if self.is_ready_for_use() is False or hvo_seq_b.is_ready_for_use() is False:
            return None

        a = self.get(hvo_str).flatten()
        b = hvo_seq_b.get(hvo_str).flatten()

        return np.linalg.norm((a - b), ord=2)

    def calculate_cosine_similarity_with(self, hvo_seq_b):
        """
        Calculates cosine similarity with secondary sequence
        Calculates the cosine of the angle between flattened hvo scores (flatten into 1d)

        :param hvo_seq_b: a secondary hvo_sequence to measure similarity with
        :return:
            a value between -1 and 1 --> 1. 0 when sequences are equal, 0 when they are "perpendicular"
        """

        if self.is_ready_for_use() is False or hvo_seq_b.is_ready_for_use() is False:
            return None

        return cosine_similarity(self, hvo_seq_b)

    def calculate_cosine_distance_with(self, hvo_seq_b):
        # returns 1 - cosine_similarity
        # returns 0 when equal and 1 when "perpendicular"

        if self.is_ready_for_use() is False or hvo_seq_b.is_ready_for_use() is False:
            return None

        return cosine_distance(self, hvo_seq_b)

    def calculate_hamming_distance_with(self, hvo_seq_b, reduction_map=None, beat_weighting=False):
        """
        Calculates the vanilla hamming distance between the current hvo_seq and a target sequence

        :param hvo_seq_b:       target sequence from which the distance is measured
        :param reduction_map:   None:       Calculates distance as is
                                reduction_map: an alternative drum mapping to reduce the score
                                               options available in drum_mappings.py:
                                                1.      Groove_Toolbox_3Part_keymap (low, mid, hi)
                                                2.      Groove_Toolbox_5Part_keymap (kick, snare, closed, open, tom)

        :param beat_weighting:  If true, weights time steps using a 4/4 metrical awareness weights
        :return:
            distance:           abs value of distance between sequences
        """
        if self.is_ready_for_use() is False or hvo_seq_b.is_ready_for_use() is False:
            return None

        if reduction_map is not None:
            groove_a = self.get_with_different_drum_mapping("v", reduction_map)
            groove_b = hvo_seq_b.get_with_different_drum_mapping("v", reduction_map)
        else:
            groove_a = self.get("v")
            groove_b = hvo_seq_b.get("v")

        if beat_weighting is True:
            groove_a = _weight_groove(groove_a)
            groove_b = _weight_groove(groove_b)

        x = (groove_a.flatten() - groove_b.flatten())

        _weighted_Hamming_distance = math.sqrt(np.dot(x, x.T))

        return _weighted_Hamming_distance

    def calculate_fuzzy_hamming_distance_with(self, hvo_seq_b, beat_weighting=False):
        """
        Calculates the vanilla hamming distance between the current hvo_seq and a target sequence

        :param hvo_seq_b:       target sequence from which the distance is measured

        :param beat_weighting:  If true, weights time steps using a 4/4 metrical awareness weights
        :return:
            distance:           abs value of distance between sequences
        """

        if self.is_ready_for_use() is False or hvo_seq_b.is_ready_for_use() is False:
            return None

        velocity_groove_a = self.get("v")
        utiming_groove_a = self.get("o", offsets_in_ms=True, use_nan_for_non_hits=True)
        velocity_groove_b = hvo_seq_b.get("v")
        utiming_groove_b = hvo_seq_b.get("o", offsets_in_ms=True, use_nan_for_non_hits=True)

        fuzzy_dist = fuzzy_Hamming_distance(velocity_groove_a, utiming_groove_a,
                                            velocity_groove_b, utiming_groove_b,
                                            beat_weighting=beat_weighting)
        return fuzzy_dist

    def calculate_structural_similarity_distance_with(self, hvo_seq_b):
        """
        Calculates the vanilla hamming distance between the current hvo_seq and a target sequence

        :param hvo_seq_b:       target sequence from which the distance is measured

        :return:
            distance:           abs value of distance between sequences
        """

        if self.is_ready_for_use() is False or hvo_seq_b.is_ready_for_use() is False:
            return None

        groove_b = hvo_seq_b.get_reduced_velocity_groove()
        groove_a = self.get_reduced_velocity_groove()

        assert(groove_a.shape == groove_b.shape), "can't calculate structural similarity difference as " \
                                                  "the dimensions are different"

        x = (groove_b.flatten() - groove_a.flatten())

        structural_similarity_distance = math.sqrt(np.dot(x, x.T))

        return structural_similarity_distance

    def get_reduced_velocity_groove(self):
        # Remove ornamentation from a groove to return a simplified representation of the rhythm structure
        # change salience profile for different metres etc

        velocity_groove = self.get("v")

        metrical_profile_4_4 = [0, -2, -1, -2, 0, -2, -1, -2, -0, -2, -1, -2, -0, -2, -1, -2,
                                0, -2, -1, -2, 0, -2, -1, -2, 0, -2, -1, -2, 0, -2, -1, -2]

        reduced_groove = np.zeros(velocity_groove.shape)
        for i in range(velocity_groove.shape[1]):  # number of parts to reduce
            reduced_groove[:, i] = _reduce_part(velocity_groove[:, i], metrical_profile_4_4)

        rows_to_remove = [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19, 21, 22, 23, 25, 26, 27, 29, 30, 31]
        reduced_groove = np.delete(reduced_groove, rows_to_remove, axis=0)

        return reduced_groove

    def is_performance(self, velocity_threshold=0.3, offset_threshold=0.1):
        """
        By looking at the unique velocities and offsets, approximate whether the hvo_sequence comes from a
        performance MIDI file or not

        :param velocity_threshold:      threshold from 0 to 1 indicating what percentage of the velocities different
                                        from 0 and 1 must be unique to consider the sequence to be performance.

        :param offset_threshold:        threshold from 0 to 1 indicating what percentage of the offsets different
                                        from 0 must be unique to consider the sequence to be performance.
        :return:
            is_performance:             boolean value returning whether the sequence is or not from a performance
        """

        assert (0 <= velocity_threshold <= 1 and 0 <= offset_threshold <= 1), "Invalid threshold"
        is_perf = False

        nonzero_nonone_vels = self.velocities[(self.velocities != 0) & (self.velocities != 1)]
        unique_velocities = np.unique(nonzero_nonone_vels)
        unique_vel_perc = 0 if len(nonzero_nonone_vels) == 0 else len(unique_velocities) / len(nonzero_nonone_vels)

        nonzero_offsets = self.offsets[np.nonzero(self.offsets)[0]]
        unique_offsets = np.unique(nonzero_offsets)
        unique_off_perc = 0 if len(nonzero_offsets) == 0 else len(unique_offsets) / len(nonzero_offsets)

        if unique_vel_perc >= velocity_threshold and unique_off_perc >= offset_threshold:
            is_perf = True

        return is_perf


def empty_like(other_hvo_sequence):
    """
    Creates a copy of other_hvo_sequence while ignoring the hvo content
    :param other_hvo_sequence:
    :return:
    """
    new_hvo_seq = HVO_Sequence(
        beat_division_factors=other_hvo_sequence.grid_maker.beat_division_factors,
        drum_mapping=other_hvo_sequence.drum_mapping
    )
    other_dict = copy.deepcopy(other_hvo_sequence.__dict__)
    new_hvo_seq.__dict__.update(other_dict)
    new_hvo_seq.remove_hvo()
    return new_hvo_seq


def zero_like(other_hvo_sequence):
    new_hvo_seq = HVO_Sequence(
        beat_division_factors=other_hvo_sequence.grid_maker.beat_division_factors,
        drum_mapping=other_hvo_sequence.drum_mapping
    )
    other_dict = copy.deepcopy(other_hvo_sequence.__dict__)
    new_hvo_seq.__dict__.update(other_dict)
    if new_hvo_seq.hvo is not None:
        new_hvo_seq.hvo = np.zeros_like(new_hvo_seq.hvo)

    return new_hvo_seq
