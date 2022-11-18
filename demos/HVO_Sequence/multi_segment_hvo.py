from hvo_sequence import HVO_Sequence
from hvo_sequence import ROLAND_REDUCED_MAPPING
from hvo_sequence import note_sequence_to_hvo_sequence, midi_to_hvo_sequence


figs = []

# ----------------------------------------------------------------
# -----------          CREATE TWO HVOs              --------------
# ----------------------------------------------------------------
# Sequence A
beat_div_factors = [3, 4]
hvo_seq_a = HVO_Sequence(drum_mapping=ROLAND_REDUCED_MAPPING, beat_division_factors=beat_div_factors)
hvo_seq_a.add_time_signature(0, 4, 4)
hvo_seq_a.add_tempo(0, 50)
hvo_seq_a.add_tempo(4, 50)
hvo_seq_a.add_time_signature(10, 4, 8)
hvo_seq_a.add_tempo(15, 100)
hvo_seq_a.metadata["performer"] = "drummer_a"
hvo_seq_a.metadata["dataset"] = "datasetA"
hvo_seq_a.metadata["instrument"] = "drums"
hvo_seq_a.number_of_steps = 32
hvo_seq_a.random(32)
#hvo_seq_a.piano_roll(filename="SEQ A", show_figure=True)
# WARNING At this time, to have multi segment HVOs, the beat_div_factors must be the same for all segments

# Get Segments
hvo_seq_a.consistent_segment_hvo_sequences
# ([<hvo_sequence.hvo_seq.HVO_Sequence at 0x1285c5af0>,
#   <hvo_sequence.hvo_seq.HVO_Sequence at 0x1285c57f0>,
#   <hvo_sequence.hvo_seq.HVO_Sequence at 0x1285c5850>],
#  [0, 12, 18])
# Observations:
#   1. While we added 4 changes at 0, 4, 10, 15 there are only 3 segments.
#       The tempo change at 4 is the same as the tempo at 0, so it is not a new event to be registered.
#   2. The start times of the segments are 0, 12, 18 while the events were added at 0, 10, 15.
#       this is because the changes are automatically snapped to the nearest beat determined by the time signature.
#       in this case, the events can only be registered at gridlines that are multiples of sum(beat_div_factors) - 1 = 6.


#hvo_seq_a.expand_length(4, "step")
hvo_seq_a.hits
hvo_seq_a.add_note(pitch=42, velocity=.1, start_sec=1)
hvo_seq_a.add_note(1.3, 42, .7)
hvo_seq_a.add_note(2.35, 42, .7)
hvo_seq_a.add_note(4.24325, 42, .7)
# hvo_seq_a.random(100)
#hvo_seq_a.grid_lines

hvo_seq_a.piano_roll(filename="SEQ A", show_figure=True)

hvo_seq_a.save_hvo_to_midi("demos/HVO_Sequence/misc/multiSeg/Segment A.mid")
hvo_seq_a.save_audio(sf_path="hvo_sequence/soundfonts/Standard_Drum_Kit.sf2", filename="demos/HVO_Sequence/misc/multiSeg/Segment A.wav")
# hvo_seq_a_from_midi = midi_to_hvo_sequence("hvo_seq_a.mid", drum_mapping=ROLAND_REDUCED_MAPPING, beat_division_factors=[3, 4])
# hvo_seq_a_from_midi.piano_roll("demos/HVO_Sequence/misc/multiSeg/Sequence A_fromMidi.html", save_figure=True)

# Sequence B
hvo_seq_b = HVO_Sequence(drum_mapping=ROLAND_REDUCED_MAPPING, beat_division_factors=beat_div_factors)
hvo_seq_b.add_time_signature(0, 6, 8)
hvo_seq_b.add_tempo(0, 50)
hvo_seq_b.add_time_signature(1, 4, 8)
hvo_seq_b.add_tempo(20, 50)
hvo_seq_b.metadata["genre"] = "rock"
hvo_seq_b.metadata["dataset"] = "datasetB"
hvo_seq_b.random(100)
hvo_seq_b.piano_roll(filename="SEQ B", show_figure=True)
hvo_seq_b.save_hvo_to_midi("demos/HVO_Sequence/misc/multiSeg/Segment B.mid")
hvo_seq_b.save_audio(sf_path="hvo_sequence/soundfonts/Standard_Drum_Kit.sf2",
                     filename="demos/HVO_Sequence/misc/multiSeg/Segment B.wav")

# ADD two sequences
hvo_seq_c = hvo_seq_a + hvo_seq_b
segs, starts = hvo_seq_c.consistent_segment_hvo_sequences
for i, (seg, start) in enumerate(zip(segs, starts)):
    seg.piano_roll(show_figure=False)

hvo_seq_c.piano_roll(filename="SEQ C", show_figure=True)
hvo_seq_c.save_hvo_to_midi("demos/HVO_Sequence/misc/multiSeg/Segment C.mid")
hvo_seq_c.save_audio(sf_path="hvo_sequence/soundfonts/Standard_Drum_Kit.sf2", filename="demos/HVO_Sequence/misc/multiSeg/Segment C.wav")
hvo_seq_c.add_note(pitch=42, velocity=.7, start_sec=1.3)
print("Sequence A - Metadata ---- ", hvo_seq_a.metadata)
print("Sequence B - Metadata ---- ", hvo_seq_b.metadata)
print("Sequence C - Metadata ---- ", hvo_seq_c.metadata)

# Check the time_steps registered for metadata
print("Sequence A - Metadata Times ---- ", hvo_seq_a.metadata.time_steps)
print("Sequence B - Metadata Times ---- ", hvo_seq_b.metadata.time_steps)
print("Sequence C - Metadata Times ---- ", hvo_seq_c.metadata.time_steps)

# Visualize
figs.append(hvo_seq_a.piano_roll("Sequence A"))
figs.append(hvo_seq_b.piano_roll("Sequence B"))
figs.append(hvo_seq_c.piano_roll("Sequence C"))

hvo_seq_2a = hvo_seq_a + hvo_seq_a
hvo_seq_a.piano_roll("Sequence A.html", show_figure=True)
hvo_seq_2a.piano_roll("Sequence A and A.html", show_figure=True)

hvo_seq_2c = hvo_seq_c + hvo_seq_c
hvo_seq_2c.piano_roll("Sequence C and C.html", show_figure=True)
hvo_seq_2c.save_hvo_to_midi("demos/HVO_Sequence/misc/multiSeg/Segment  C and C.mid")
hvo_seq_2c.save_audio(sf_path="hvo_sequence/soundfonts/Standard_Drum_Kit.sf2", filename="demos/HVO_Sequence/misc/multiSeg/Segment  C and C.wav")

hvo_seq_d = hvo_seq_2c.copy_zero()
hvo_seq_d.add_note(pitch=36, velocity=.7, start_sec=0)
hvo_seq_d.add_note(pitch=42, velocity=.9, start_sec=0)
hvo_seq_d.add_note(pitch=42, velocity=.2, start_sec=1)
hvo_seq_d.add_note(pitch=42, velocity=.7, start_sec=1.3)
hvo_seq_d.add_note(pitch=36, velocity=.7, start_sec=10)
hvo_seq_d.add_note(pitch=42, velocity=.9, start_sec=20)
hvo_seq_d.add_note(pitch=42, velocity=.2, start_sec=20)
hvo_seq_d.add_note(pitch=42, velocity=.7, start_sec=21.3)
hvo_seq_d.piano_roll("demos/HVO_Sequence/misc/multiSeg/Sequence D.html",
                     save_figure=True)
hvo_seq_d.save_hvo_to_midi("demos/HVO_Sequence/misc/multiSeg/Segment D.mid")
hvo_seq_d.save_audio(sf_path="hvo_sequence/soundfonts/Standard_Drum_Kit.sf2",
                     filename="demos/HVO_Sequence/misc/multiSeg/Segment D.wav")

from hvo_sequence import io_helpers as io
from hvo_sequence import ROLAND_REDUCED_MAPPING

hvo_seq_d2 = io.midi_to_hvo_sequence(filename="demos/HVO_Sequence/misc/multiSeg/Segment D.mid",
                        drum_mapping=ROLAND_REDUCED_MAPPING, beat_division_factors=[3, 4])
hvo_seq_d2.piano_roll("demos/HVO_Sequence/misc/multiSeg/Sequence D2.html",
                        save_figure=True)
hvo_seq_d2.save_hvo_to_midi("demos/HVO_Sequence/misc/multiSeg/Segment D2.mid")
hvo_seq_d2.save_audio(sf_path="hvo_sequence/soundfonts/Standard_Drum_Kit.sf2",
                        filename="demos/HVO_Sequence/misc/multiSeg/Segment D2.wav")

# from bokeh.io import save
# from bokeh.layouts import gridplot
# save(gridplot(figs, ncols=1), filename="demos/HVO_Sequence/misc/a.html")
# # hvo_seq_a.save_audio(sf_path="hvo_sequence/soundfonts/Standard_Drum_Kit.sf2", filename="demos/hvo_sequence/misc/added.wav", sf_path="hvo_sequence/soundfonts/Standard_Drum_Kit.sf2")



