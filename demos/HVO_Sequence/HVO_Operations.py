from hvo_sequence import HVO_Sequence
from hvo_sequence import ROLAND_REDUCED_MAPPING
from hvo_sequence import note_sequence_to_hvo_sequence, midi_to_hvo_sequence


figs = []

# ----------------------------------------------------------------
# -----------          CREATE TWO HVOs              --------------
# ----------------------------------------------------------------
# Sequence A
hvo_seq_a = HVO_Sequence(drum_mapping=ROLAND_REDUCED_MAPPING, beat_division_factors=[12])
hvo_seq_a.add_time_signature(0, 4, 4)
hvo_seq_a.add_tempo(0, 50)
hvo_seq_a.add_time_signature(10, 4, 4)
hvo_seq_a.add_tempo(15, 50)
hvo_seq_a.metadata["performer"] = "drummer_a"
hvo_seq_a.metadata["dataset"] = "datasetA"
hvo_seq_a.metadata["instrument"] = "drums"
hvo_seq_a.adjust_length(10)
# hvo_seq_a.add_note(42, .1, 1)
# hvo_seq_a.add_note(42, .7, 1.3)
# hvo_seq_a.add_note(42, .7, 2.35)
# hvo_seq_a.add_note(42, .7, 4.24325)
hvo_seq_a.random(100)

hvo_seq_a.piano_roll("Sequence A.html", save_figure=True)

hvo_seq_a.save_hvo_to_midi("hvo_seq_a.mid")

hvo_seq_a = midi_to_hvo_sequence("hvo_seq_a.mid", drum_mapping=ROLAND_REDUCED_MAPPING, beat_division_factors=[3, 4])
hvo_seq_a.piano_roll("hvo_seq_a.mid.html", save_figure=True)

# Sequence B
hvo_seq_b = HVO_Sequence(drum_mapping=ROLAND_REDUCED_MAPPING, beat_division_factors=[12])
hvo_seq_b.add_time_signature(0, 6, 8)
hvo_seq_b.add_tempo(0, 50)
hvo_seq_b.add_time_signature(5, 6, 8)
hvo_seq_b.add_tempo(20, 50)
hvo_seq_b.metadata["genre"] = "rock"
hvo_seq_b.metadata["dataset"] = "datasetB"
hvo_seq_b.random(100)

# ADD two sequences
hvo_seq_c = hvo_seq_a + hvo_seq_b

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

from bokeh.io import save
from bokeh.layouts import gridplot
save(gridplot(figs, ncols=1), filename="demos/HVO_Sequence/misc/a.html")
hvo_seq_a.save_audio(filename="demos/hvo_sequence/misc/added.wav", sf_path="hvo_sequence/soundfonts/Standard_Drum_Kit.sf2")