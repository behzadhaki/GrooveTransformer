from hvo_sequence import HVO_Sequence
from hvo_sequence import ROLAND_REDUCED_MAPPING
beat_div_factor = [4]           # divide each quarter note in 4 divisions
hvo_seq = HVO_Sequence(beat_division_factors=beat_div_factor,
                       drum_mapping=ROLAND_REDUCED_MAPPING)

# ----------------------------------------------------------------
# -----------           CREATE A SCORE              --------------
# ----------------------------------------------------------------

# Add two time_signatures
hvo_seq.add_time_signature(time_step=0, numerator=4, denominator=4)

# Add two tempos
hvo_seq.add_tempo(time_step=0, qpm=120)

# Create a random hvo for 32 time steps and 9 voices
hvo_seq.random(110)


# ----------------------------------------------------------------
# -----------           CREATE A SCORE              --------------
# ----------------------------------------------------------------

#
number_of_bars_per_segment, segment_shift_in_bars = 2, 1
adjust_length = False

segments = hvo_seq.split_into_segments(number_of_bars_per_segment=2,
                                       segment_shift_in_bars=1, adjust_length=True)

for ix, seg in enumerate(segments):
    seg.to_html_plot(save_figure=True, filename=f'misc/{ix}.html')