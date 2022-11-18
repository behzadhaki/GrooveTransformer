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
hvo_seq.random(32)

# OR CREATE A SCORE DIRECTLY
# hvo_seq.hvo = np.array of shape (num_grid_lines, 9*3)

# -------------------------------------------------------------------
# -----------           ADD META DATA                  --------------
# -------------------------------------------------------------------
from hvo_sequence.custom_dtypes import Metadata
metadata_first_bar = Metadata({
    'title': 'My first score',
    'style': 'Rock',
    'source': 'Dataset X'})
hvo_seq.metadata = metadata_first_bar

# Add additional metadata (Even with new information not in the first bar)
metadata_second_bar = Metadata({
    'title': 'My second score',
    'style': 'Pop',
    'source': 'Dataset Y',
    'human_performance': True})
hvo_seq.metadata.append(metadata_second_bar, start_at_time_step=16)

# print(hvo_seq.metadata.time_steps)
#       [0, 16]
# print(hvo_seq.metadata)
#           {'title': ['My first score', 'My second score'],
#           'style': ['Rock', 'Pop'],
#           'source': ['Dataset X', 'Dataset Y'],
#           'human_performance': [None, True]}

# -------------------------------------------------------------------
# -----------           saving                         --------------
# -------------------------------------------------------------------
hvo_seq.save("demos/HVO_Sequence/misc/empty.hvo")

# -------------------------------------------------------------------
# -----------           Loading                         --------------
# -------------------------------------------------------------------
from pickle import load
hvo_seq_loaded = load(open("demos/HVO_Sequence/misc/empty.hvo", "rb"))

if hvo_seq_loaded == hvo_seq:
    print ("Loaded sequence is equal to the saved one")

# ----------------------------------------------------------------
# -----------           Access Data                 --------------
# ----------------------------------------------------------------
hits = hvo_seq.get("h")    # get hits
vels = hvo_seq.get("v")    # get vel
offsets = hvo_seq.get("o")    # get offsets

hvo_seq.get("vo")    # get vel with offsets
hvo_seq.get("hv0")    # get hv with offsets replaced as 0
hvo_seq.get("ovhhv0")    # use h v o and 0 to create any tensor


# ----------------------------------------------------------------
# -----------           Plot PianoRoll              --------------
# ----------------------------------------------------------------
hvo_seq.to_html_plot(
    filename="DEMO_PATTERN.html",
    save_figure=False,
    show_figure=True)

# ----------------------------------------------------------------
# -----------           Synthesize/Export           --------------
# ----------------------------------------------------------------
# Export to midi
hvo_seq.save_hvo_to_midi(filename="demos/HVO_Sequence/misc/test.mid")

# Export to note_sequece
hvo_seq.to_note_sequence(midi_track_n=10)

# Synthesize to audio
audio = hvo_seq.synthesize(sr=44100, sf_path="hvo_sequence/soundfonts/Standard_Drum_Kit.sf2")

# Synthesize to audio and auto save
hvo_seq.save_audio(filename="demos/HVO_Sequence/misc/temp.wav", sr=44100,
                   sf_path="hvo_sequence/soundfonts/Standard_Drum_Kit.sf2")


# ----------------------------------------------------------------
# -----------           Load from Midi             --------------
# ----------------------------------------------------------------
from hvo_sequence import midi_to_hvo_sequence
hvo_seq = midi_to_hvo_sequence(
    filename='misc/test.mid',
    drum_mapping=ROLAND_REDUCED_MAPPING,
    beat_division_factors=[4])
