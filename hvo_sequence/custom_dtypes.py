import warnings
from copy import deepcopy
import numpy as np
import note_seq
import math

# ======================================================================================================================
# === Meta Data, Time Signature, and Tempo Classes =====================================================================
# ======================================================================================================================


class Metadata(dict):
    """
    A dictionary that can be appended to.
    This means that instead of overwriting the values, it will append the values to the existing values.
    If the values are not lists, they will be converted to lists prior to appending.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.time_steps = [0]   # keeps track of the time steps for sequenece of metadata info

    def __append_single_metadata(self, other, start_at_time_step):
        assert isinstance(other, Metadata), "the object to append must be of type Metadata"
        assert start_at_time_step > max(self.time_steps), \
            "the start time step must be greater than the last time step of the current metadata" \
            " (start_at_time_step > {})".format(max(self.time_steps))

        # check the values are different from the current values
        sames = []
        for key, value in other.items():
            if key in self:
                sames.append(value == self[key])
        if sames:
            if all(sames):
                return

        other_ = deepcopy(other)
        # ensure values are both lists
        if len(self.time_steps) == 1:
            for key, value in self.items():
                self[key] = [value]
        if len(other_.time_steps) == 1:
            for key, value in other_.items():
                other_[key] = [value]
        # find union of metadata keys
        for key in set(self.keys()).union(set(other_.keys())):
            if key not in self:
                self[key] = [None]*len(self.time_steps)
            if key not in other_:
                other_[key] = [None]*len(other_.time_steps)
        # append the values
        for ix, time_step in enumerate(other_.time_steps):
            new_time_step = time_step + start_at_time_step
            self.time_steps.append(new_time_step)
            for key in set(self.keys()).union(set(other_.keys())):
                self[key].append(other_[key][ix])

    def append(self, other, start_at_time_step):
        for start, end, metadata in other.split():
            self.__append_single_metadata(metadata, start_at_time_step + start)

    def split(self):
        """
        Split the metadata into a list of metadata objects, one for each meta data consistent region.
        :return: list of tuples of (start_time_step, end_time_step, Metadata)
        """
        starts = self.time_steps
        ends = np.array(starts[1:] + [np.inf])-1
        metadatas = []
        if len(starts) == 1:
            return [(0, np.inf, self)]
        else:
            for i in range(len(starts)):
                metadatas.append(Metadata({k: v[i] for k, v in self.items()}))
        return list(zip(starts, ends, metadatas))


class Time_Signature(object):
    def __init__(self, time_step=None, numerator=None, denominator=None):
        self.__time_step = None     # index corresponding to the time_step in hvo where signature change happens
        self.__numerator = None
        self.__denominator = None

        if time_step is not None:
            self.time_step = time_step
        if numerator is not None:
            self.numerator = numerator
        if denominator is not None:
            self.denominator = denominator

    def __repr__(self):
        rep = "Time_Signature = { \n " +\
              "\t time_step: {}, \n \t numerator: {}, \n \t denominator: {}".format(
                  self.time_step, self.numerator, self.denominator)+"\n}"
        return rep

    def __eq__(self, other):
        # Make sure the types are the same
        assert isinstance(other, Time_Signature), "Expected a Time_Signature Instance but received {}".format(
            type(other))

        # ignore the start time index of time_signatures and check whether other fields are equal
        is_eq = all([
            self.numerator == other.numerator,
            self.denominator == other.denominator,
            self.time_step == other.time_step
        ])
        return is_eq

    def __ne__(self, other):
        return not self.__eq__(other)

    @property
    def time_step(self):
        return self.__time_step

    @time_step.setter
    def time_step(self, val):
        if val is None:
            self.__time_step = None
        else:
            assert isinstance(val, int), "Time signature time should be an integer " \
                                         "corresponding to the time_step in HVO sequence"
            assert val >= 0, "Make sure the numerator is greater than or equal to zero"
            self.__time_step = val

    @property
    def numerator(self):
        return self.__numerator

    @numerator.setter
    def numerator(self, val):
        if val is None:
            self.__numerator = None
        else:  # Check consistency of input with the required
            assert isinstance(val, int), "Make sure the numerator is an integer"
            assert val > 0, "Make sure the numerator is greater than zero"
            # Now, safe to update the __time_signature local variable
            self.__numerator = val

    @property
    def denominator(self):
        return self.__denominator

    @denominator.setter
    def denominator(self, val):
        if val is None:
            self.__denominator = None
        else:  # Check consistency of input with the required
            # Ensure numerator is an integer
            assert isinstance(val, int), "Make sure the denominator is an integer"
            assert is_power_of_two(val), "Denominator must be binary (i.e. a value that is a power of 2)"
            assert val > 0, "Make sure the numerator is greater than zero"
            # Now, safe to update the __time_signature local variable
            self.__denominator = val

    @property
    def is_ready_to_use(self):
        # Checks to see if all fields are filled and consequently, the Time_Signature is ready to be used externally
        fields_available = list()
        for key in self.__dict__.keys():
            fields_available.append(True) if self.__dict__[key] is not None else fields_available.append(False)
        return all(fields_available)

    def copy(self):
        return Time_Signature(self.time_step, self.numerator, self.denominator)

class Tempo(object):
    def __init__(self, time_step=None,  qpm=None):
        self.__time_step = None    # index corresponding to the time_step in hvo where signature change happens
        self.__qpm = None
        if time_step is not None:
            self.time_step = time_step
        if qpm is not None:
            self.qpm = qpm

    def __repr__(self):
        rep = "Tempo = { \n " +\
              "\t time_step: {}, \n \t qpm: {}".format(self.time_step, self.qpm)+"\n}"
        return rep

    @property
    def time_step(self):
        return self.__time_step

    @time_step.setter
    def time_step(self, val):
        if val is None:
            self.__time_step = None
        else:
            assert isinstance(val, int), "Starting time index for tempo should be an integer " \
                                         "corresponding to the time_step in HVO sequence. Received {}".format(type(val))
            assert val >= 0, "Make sure the numerator is greater than or equal to zero"
            self.__time_step = val

    @property
    def qpm(self):
        return self.__qpm

    @qpm.setter
    def qpm(self, val):
        if val is None:
            self.__qpm = None
        else:   # Check consistency of input with the required
            # Ensure numerator is an integer
            assert isinstance(val, (int, float)), "Make sure the qpm is a float"
            assert val > 0, "Make sure tempo is positive"
            # Now, safe to update the __qpm local variable
            self.__qpm = val

    @property
    def is_ready_to_use(self):
        # Checks to see if all fields are filled and consequently, the Time_Signature is ready to be used externally
        fields_available = list()
        for key in self.__dict__.keys():
            fields_available.append(True) if self.__dict__[key] is not None else fields_available.append(False)
        return all(fields_available)

    def __eq__(self, other):
        # Make sure the types are the same
        assert isinstance(other, Tempo), "Expected a Tempo Instance but received {}".format(
            type(other))

        # ignore the start time index of time_signatures and check whether qpms are equal
        return (self.qpm == other.qpm) and (self.time_step == other.time_step)

    def __ne__(self, other):
        return not self.__eq__(other)

    def copy(self):
        return Tempo(self.time_step, self.qpm)

def is_power_of_two(n):
    """
    Checks if a value is a power of two
    @param n:                               # value to check (must be int or float - otherwise assert error)
    @return:                                # True if a power of two, else false
    """
    if n is None:
        return False

    assert (isinstance(n, int) or isinstance(n, float)), "The value to check must be either int or float"

    if (isinstance(n, float) and n.is_integer()) or isinstance(n, int):
        # https://stackoverflow.com/questions/57025836/how-to-check-if-a-given-number-is-a-power-of-two
        n = int(n)
        return (n & (n - 1) == 0) and n != 0
    else:
        return False

# ======================================================================================================================
# === GridMaker Class ==================================================================================================
# Only recalculate the grid when a get_* method is called to access the values have changed back to None
# ======================================================================================================================
def are_beat_division_factors_legal(beat_division_factors: list):
    """beat_division_factors must integers and no two factors can be multiples of each other"""
    assert isinstance(beat_division_factors, list), "beat_division_factors must be a list"
    assert all([isinstance(x, int) for x in beat_division_factors]), "beat_division_factors must be a list of integers"
    assert all([x > 0 for x in beat_division_factors]), "beat_division_factors must be a list of positive integers"

    # check if any two factors are multiples of each other
    for i in range(len(beat_division_factors)):
        for j in range(i+1, len(beat_division_factors)):
            if beat_division_factors[i] % beat_division_factors[j] == 0 or \
                    beat_division_factors[j] % beat_division_factors[i] == 0:
                return False
    return True


class GridMaker:
    """
    Class to generate a grid for a given time signature and tempo and beat division factors
    """
    def __init__(self, beat_division_factors: list):
        """
        Class to generate a grid for a given time signature and tempo and beat division factors
        This class is implemented such that if any information that can modify [1] the grid is changed, the grid is
        cleared and only recalculated when a get_* method is called to access the values

        [1] The grid is cleared whenever a new time signature, tempo is set or the length of the grid is changed

        :param beat_division_factors:  list of integers that are the factors of the beat divisions.
        no two factors can be multiples of each other
        """
        # must be pickled
        # ------------------------------------------------------------------------------------------------
        assert are_beat_division_factors_legal(beat_division_factors), \
            "beat_division_factors must be a list of integers and no two factors can be multiples of each other"
        self.__beat_division_factors = beat_division_factors

        # ------------------------------------------------------------------------------------------------
        self.__tempos = None
        self.__time_signatures = None
        self.__n_steps = -1                  # total steps of grid pre-calculated / must be larger than hvo steps

        # Constructed per segment after unpickling
        # ------------------------------------------------------------------------------------------------
        self.__n_steps_per_beat = sum(beat_division_factors)-(len(beat_division_factors)-1)     # num steps per segment

        # Segment Info
        # ------------------------------------------------------------------------------------------------
        self.__segment_starts = None          # start time steps of each segment
        self.__segment_ends = None            # end time steps of each segment **exclusive final step**
        self.__segment_time_signatures = None             # time signature of each segment
        self.__segment_tempos = None                      # tempo of each segment
        self.__segment_durations_in_steps = None          # duration of each seg in num steps (multiple of beat steps)
        self.__segment_durations_in_sec = None            # tempo of each segment
        self.__segment_single_beat_grid_locations_in_sec = None            # duration of a single beat in seconds

        # Grid Info
        # ------------------------------------------------------------------------------------------------
        self.__grid_lines = []
        self.__is_minor_grid_lines = []
        self.__is_major_grid_lines = []
        self.__is_downbeat_grid_line = []
        self.__total_seconds_prepared = -1

    def __getstate__(self):
        state = {
            'beat_division_factors': self.__beat_division_factors,
            'tempos': self.__tempos,
            'time_signatures': self.__time_signatures,
            'n_steps': self.__n_steps,
        }
        return state

    def __setstate__(self, state):
        self.__beat_division_factors = state['beat_division_factors']
        self.__tempos = state['tempos']
        self.__time_signatures = state['time_signatures']
        self.__n_steps = state['n_steps']

        # Constructed per segment after unpickling
        # ------------------------------------------------------------------------------------------------
        self.__n_steps_per_beat = \
            sum(self.__beat_division_factors) - (len(self.__beat_division_factors) - 1)  # num steps per segment

        # Segment Info
        # ------------------------------------------------------------------------------------------------
        self.__segment_starts = None  # start time steps of each segment
        self.__segment_ends = None  # end time steps of each segment **exclusive final step**
        self.__segment_time_signatures = None  # time signature of each segment
        self.__segment_tempos = None  # tempo of each segment
        self.__segment_durations_in_steps = None  # duration of each seg in num steps (multiple of beat steps)
        self.__segment_durations_in_sec = None  # tempo of each segment
        self.__segment_single_beat_grid_locations_in_sec = None  # duration of a single beat in seconds

        # Grid Info
        # ------------------------------------------------------------------------------------------------
        self.__grid_lines = []
        self.__is_minor_grid_lines = []
        self.__is_major_grid_lines = []
        self.__is_downbeat_grid_line = []
        self.__total_seconds_prepared = -1

    @property
    def beat_division_factors(self):
        return self.__beat_division_factors

    @property
    def time_signatures(self):
        ts_ = sorted([ts for ts in self.__time_signatures], key=lambda x: x.time_step)
        ts_[0].time_step = 0
        return ts_

    @time_signatures.setter
    def time_signatures(self, ts_list):
        assert isinstance(ts_list, list), "time_signatures must be a list"
        assert all([isinstance(x, Time_Signature) for x in ts_list]), "time_signatures must be a list of TimeSignature instances"
        self.__time_signatures = None
        for ts in ts_list:
            self.add_time_signature(ts.time_step, ts.numerator, ts.denominator)
        self.erase_segment_info()
        self.erase_grid()

    @property
    def tempos(self):
        tempos_ = sorted([t for t in self.__tempos], key=lambda x: x.time_step)
        tempos_[0].time_step = 0
        return tempos_

    @tempos.setter
    def tempos(self, tempo_list):
        assert isinstance(tempo_list, list), "tempos must be a list"
        assert all([isinstance(x, Tempo) for x in tempo_list]), "tempos must be a list of Tempo instances"
        self.__tempos = None
        for tempo in tempo_list:
            self.add_tempo(tempo.time_step, tempo.qpm)
        self.erase_segment_info()
        self.erase_grid()

    @property
    def n_steps(self):
        return self.__n_steps

    @n_steps.setter
    def n_steps(self, n_steps):
        """adjusts the number of steps in the grid.
        Ensures that (1) snaps n_steps to nest available beat, (2) ensure resulting grid steps are at least 1 bar after
        the last time signature change
        """
        assert int(n_steps) == n_steps, "n_steps must be an integer"
        assert n_steps > 0, "n_steps must be greater than 0"
        # make sure aligns on beat
        n_steps = math.ceil(n_steps / self.__n_steps_per_beat) * self.__n_steps_per_beat

        if n_steps > self.__n_steps:
            self.erase_segment_info()
            self.erase_grid()
            # self.extract_segment_info()
            # at least one bar after las
            ts = sorted(self.time_signatures, key=lambda x: x.time_step)[-1]
            tp = sorted(self.tempos, key=lambda x: x.time_step)[-1]
            last_seg_beginning_at = max(ts.time_step, tp.time_step)
            self.__n_steps = max(last_seg_beginning_at + ts.numerator * self.__n_steps_per_beat, n_steps)

    @property
    def n_steps_per_beat(self):
        return self.__n_steps_per_beat

    def add_time_signature(self, time_step, numerator, denominator):
        """ Adds a time signature to the grid maker __time_signatures list
        If two consecutive time signatures exist with the same numerator and denominator, the second one is ignored
        Also, if two consecutive time signatures exist with the same time_step, the latest one is kept
        @param time_step:       time step of the time signature
        @param numerator:       numerator of the time signature
        @param denominator:     denominator of the time signature
        """
        old_time_signatures = deepcopy(self.__time_signatures if self.__time_signatures is not None else [])

        # Make sure time step is a multiple of n_steps_per_beat
        time_step = int(round(time_step / self.__n_steps_per_beat) * self.__n_steps_per_beat)

        for i, prev_ts in enumerate(old_time_signatures):
            if prev_ts.time_step == time_step:
                old_time_signatures[i] = Time_Signature(time_step, numerator, denominator)

        old_time_signatures.append(Time_Signature(time_step=time_step, numerator=numerator, denominator=denominator))
        old_time_signatures.sort(key=lambda x: x.time_step)

        # Ignore time signatures that have the same numerator and denominator as the previous one
        new_time_signatures = []
        for i in range(len(old_time_signatures)):
            if i == 0:
                new_time_signatures.append(old_time_signatures[i])
            else:
                if old_time_signatures[i].numerator != old_time_signatures[i-1].numerator or \
                        old_time_signatures[i].denominator != old_time_signatures[i-1].denominator:
                    new_time_signatures.append(old_time_signatures[i])
                else:
                    new_time_signatures[i-1].time_step = min(
                        old_time_signatures[i].time_step, old_time_signatures[i-1].time_step)

        # Update time signatures if any changes were made
        # if new_time_signatures != self.__time_signatures:
        self.__time_signatures = new_time_signatures
        self.erase_segment_info()
        self.erase_grid()

    def add_tempo(self, time_step, qpm):
        """ Adds a tempo to the grid maker __tempos list
        If two consecutive tempos exist with the same qpm, the second one is ignored
        @param time_step:       time step of the tempo
        @param qpm:             qpm of the tempo
        """
        old_tempos = deepcopy(self.__tempos if self.__tempos is not None else [])

        # Make sure time step is a multiple of n_steps_per_beat
        time_step = int(round(time_step / self.__n_steps_per_beat) * self.__n_steps_per_beat)

        old_tempos.append(Tempo(time_step=time_step, qpm=qpm))
        old_tempos.sort(key=lambda x: x.time_step)

        # Ignore tempos that have the same qpm as the previous one
        new_tempos = []
        for i in range(len(old_tempos)):
            if i == 0:
                new_tempos.append(old_tempos[i])
            else:
                if old_tempos[i].qpm != old_tempos[i-1].qpm:
                    new_tempos.append(old_tempos[i])
                else:
                    new_tempos[i-1].time_step = min(old_tempos[i].time_step, old_tempos[i-1].time_step)

        # Only update if the tempos have changed
        if new_tempos != self.__tempos:
            self.__tempos = new_tempos
            self.erase_grid()
            self.erase_segment_info()
            return True
        else:
            return False

    def prepare_time_signatures_and_tempos(self):
        if self.__time_signatures is not None:
            if self.__time_signatures[0].time_step != 0 or self.__tempos[0].time_step != 0:
                self.__time_signatures[0].time_step = 0
                self.__tempos[0].time_step = 0
                warnings.warn("The first time signature and tempo must start at time step 0. Forced to change.")
        else:
            raise ValueError("No time signatures have been added to create the grid")

    def erase_segment_info(self):
        self.__segment_starts = None
        self.__segment_ends = None
        self.__segment_time_signatures = None
        self.__segment_tempos = None
        self.__segment_durations_in_steps = None
        self.__segment_durations_in_sec = None
        self.__segment_single_beat_grid_locations_in_sec = None

    def extract_segment_info(self):
        if not self.is_ready():
            raise ValueError("Time signatures or Tempos are required create a segmented grid")

        # Make sure the first time signature and tempo start at time step 0
        self.prepare_time_signatures_and_tempos()

        # Make sure the total number of steps is at least 1 bar after the beginning of last segment
        last_ts = max(self.__tempos[-1].time_step, self.__time_signatures[-1].time_step)
        self.n_steps = max(
            last_ts + 2 * self.__time_signatures[-1].numerator * self.__n_steps_per_beat,
            self.n_steps)

        # Create a list of segment starts and ends
        self.__segment_starts = [x.time_step for x in self.__time_signatures] + [x.time_step for x in self.__tempos]
        self.__segment_starts = list(sorted(set(self.__segment_starts)))
        self.__segment_ends = self.__segment_starts[1:] + [self.__n_steps]

        # find which time signature and tempo each segment starts with
        self.__segment_time_signatures = []
        self.__segment_tempos = []
        for i in range(len(self.__segment_starts)):
            ts_, tmp_ = None, None
            for ts in self.__time_signatures:
                if self.__segment_starts[i] >= ts.time_step:
                    ts_ = ts.copy()
            for tmp in self.__tempos:
                if self.__segment_starts[i] >= tmp.time_step:
                    tmp_ = tmp.copy()
            t_ = max(ts_.time_step, tmp_.time_step)
            ts_.time_step = t_
            tmp_.time_step = t_
            self.__segment_time_signatures.append(ts_)
            self.__segment_tempos.append(tmp_)
            idx = max(self.__segment_time_signatures[i].time_step, self.__segment_tempos[i].time_step)
            self.__segment_time_signatures[i].time_step = idx
            self.__segment_time_signatures[i].time_step = idx

        # Create a list of segment durations
        self.__segment_durations_in_steps = [
            self.__segment_ends[i] - self.__segment_starts[i] for i in range(len(self.__segment_starts))]

        self.__segment_durations_in_sec = []
        self.__segment_single_beat_grid_locations_in_sec = []
        for i in range(len(self.__segment_starts)):
            # calculate the duration of each segment in seconds
            n_beats = (self.__segment_ends[i] - self.__segment_starts[i]) / self.__n_steps_per_beat
            secs_per_beat = (60.0 / self.__segment_tempos[i].qpm) * 4.0 / self.__segment_time_signatures[i].denominator
            secs = n_beats * secs_per_beat
            self.__segment_durations_in_sec.append(round(secs, 3))

            locs = []
            for bdf in self.__beat_division_factors:
                locs.extend([round(i*secs_per_beat/bdf, 3) for i in range(int(bdf))])
            self.__segment_single_beat_grid_locations_in_sec.append(sorted(set(locs)))



    def is_ready(self):
        return all([self.__time_signatures is not None, self.__tempos is not None])

    def erase_grid(self):
        self.__grid_lines = []
        self.__is_minor_grid_lines = []
        self.__is_major_grid_lines = []
        self.__is_downbeat_grid_line = []
        self.__total_seconds_prepared = -1

    def prepare_grid_for_n_steps(self, n_steps=None):
        if not self.is_ready():
            raise ValueError("Time signatures or Tempos are required create a segmented grid")

        # it would automatically clear the grid if steps expanded
        if n_steps is None:
            if self.__n_steps <= 0:
                # create a grid for at least two bars after the end of the last segment
                last_time_sig = self.time_signatures[-1]
                self.n_steps = int(last_time_sig.time_step + 2 * last_time_sig.numerator * self.__n_steps_per_beat)
        else:
            self.n_steps = n_steps

        # if makes it here then a new grid is required,
        # either because grid was empty or grid length needs to be extended
        if not self.__grid_lines:
            self.extract_segment_info()

            for seg_ix in range(len(self.__segment_starts)):
                n_beats = self.__segment_durations_in_steps[seg_ix] / self.__n_steps_per_beat
                t_shift = sum(self.__segment_durations_in_sec[:seg_ix])
                ix_shift = sum(self.__segment_durations_in_steps[:seg_ix])
                beat_dur_sec = self.__segment_durations_in_sec[seg_ix] / n_beats
                for beat_ix in range(int(n_beats)):
                    for ix_from_beat, secs_from_beat in enumerate(self.__segment_single_beat_grid_locations_in_sec[seg_ix]):
                        t = round(secs_from_beat + beat_dur_sec * beat_ix + t_shift, 3)
                        in_seg_index = beat_ix * self.__n_steps_per_beat + ix_from_beat
                        from_beg_index = in_seg_index + ix_shift
                        self.__grid_lines.append(t)
                        self.__is_downbeat_grid_line.append(
                            in_seg_index %
                            (self.__segment_time_signatures[seg_ix].numerator * self.__n_steps_per_beat) == 0)
                        self.__is_major_grid_lines.append(in_seg_index % self.__n_steps_per_beat == 0)
                        self.__is_minor_grid_lines.append(in_seg_index % self.__n_steps_per_beat != 0)
            if len(self.__grid_lines) > 1:
                self.__total_seconds_prepared = \
                    self.__grid_lines[-1] + (self.__grid_lines[-1] - self.__grid_lines[-2]) / 2.0
        else:
            return False

    def get_grid_lines(self, n_steps):
        self.prepare_grid_for_n_steps(n_steps)
        return self.__grid_lines[:n_steps]

    def get_grid_lines_for_n_beats(self, n_beats):
        n_steps = n_beats * self.__n_steps_per_beat
        return self.get_grid_lines(n_steps)

    def get_major_grid_lines(self, n_steps):
        self.prepare_grid_for_n_steps(n_steps)
        return [self.__grid_lines[i] for i in range(n_steps) if self.__is_major_grid_lines[i]]

    def get_minor_grid_lines(self, n_steps):
        self.prepare_grid_for_n_steps(n_steps)
        return [self.__grid_lines[i] for i in range(n_steps) if self.__is_minor_grid_lines[i]]

    def get_downbeat_grid_lines(self, n_steps):
        self.prepare_grid_for_n_steps(n_steps)
        return [self.__grid_lines[i] for i in range(n_steps) if self.__is_downbeat_grid_line[i]]

    def get_major_grid_line_indices(self, n_steps):
        self.prepare_grid_for_n_steps(n_steps)
        return [i for i in range(n_steps) if self.__is_major_grid_lines[i]]

    def get_minor_grid_line_indices(self, n_steps):
        self.prepare_grid_for_n_steps(n_steps)
        return [i for i in range(n_steps) if self.__is_minor_grid_lines[i]]

    def get_downbeat_grid_line_indices(self, n_steps):
        self.prepare_grid_for_n_steps(n_steps)
        return [i for i in range(n_steps) if self.__is_downbeat_grid_line[i]]

    def get_index_and_offset_at_sec(self, t_sec):
        if not self.__grid_lines:
            self.prepare_grid_for_n_steps()

        if t_sec > self.__grid_lines[-2]:
            last_seg_start = sum(self.__segment_durations_in_sec[:-1])
            time_diff = t_sec - last_seg_start
            ts = self.__segment_time_signatures[-1]
            tmp = self.__segment_tempos[-1]
            secs_per_beat = (60.0 / tmp.qpm) * 4.0 / ts.denominator
            min_steps_needed = math.ceil(time_diff / secs_per_beat) * self.__n_steps_per_beat
            n_steps = ts.time_step + min_steps_needed + 1
            self.prepare_grid_for_n_steps(n_steps)

        # find index and diff from grid line
        idx, grid_val = min(enumerate(self.__grid_lines), key=lambda x: abs(x[1]-t_sec))
        diff = t_sec - grid_val
        print("idx: {}, diff: {}".format(idx, diff))
        if diff != 0:
            print(f"grid line len inside grid maker {len(self.__grid_lines)}")
            print(f"self.n_steps {self.n_steps}")
            offset = diff / (self.__grid_lines[idx+1] - self.__grid_lines[idx]) if diff > 0 else \
                diff / (self.__grid_lines[idx] - self.__grid_lines[idx-1])
        else:
            offset = 0

        return idx, round(offset, 3)

    def get_segments_info(self):
        if not self.__segment_starts:
            self.extract_segment_info()
        return {
            "segment_starts": self.__segment_starts,
            "segment_ends": self.__segment_ends,
            "tempos": self.__segment_tempos,
            "time_signatures": self.__segment_time_signatures
        }

if __name__ == "__main__":

    import time

    gridMaker = GridMaker([4])
    # # add time signatures
    gridMaker.add_time_signature(0, 3, 4)
    # #gridMaker.add_time_signature(8, 2, 4)
    # gridMaker.add_time_signature(16, 3, 4)
    #
    # # Add tempos
    gridMaker.add_tempo(0, 120)
    # gridMaker.add_tempo(30, 50)
    #
    # # Make grid
    # gridMaker.prepare_grid_for_n_steps()
    #
    # # print grid
    # # gridMaker.__dict__
    # len(gridMaker.get_grid_lines(1003))

    start = time.time()
    gridMaker.get_index_and_offset_at_sec(200)
    print(time.time() - start)

    print(gridMaker.__dict__)