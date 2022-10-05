import os
from bokeh.plotting import show

from hvo_sequence.io_helpers import note_sequence_to_hvo_sequence

from hvo_sequence.io_helpers import get_pickled_note_sequences, get_pickled_hvos
# from hvo_sequence.io_helpers import save_note_sequence_to_audio
# from hvo_sequence.hvo_seq import HVO_Sequence
from hvo_sequence.drum_mappings import ROLAND_REDUCED_MAPPING

import note_seq
import pretty_midi

if __name__ == '__main__':

    # indices = [192 , 242]
    indices = 102    # 135

    # Note_sequence from midi
    #midi_data = pretty_midi.PrettyMIDI('test/misc/tempo_time_sig_changes.mid')
    #ns = note_seq.midi_io.midi_to_note_sequence(midi_data)
    #hvo_seq = note_sequence_to_hvo_sequence(ns, drum_mapping=ROLAND_REDUCED_MAPPING)

    # Test pickle loaders
    # 1. Load Note Sequence Pickle
    ns = get_pickled_note_sequences("test/misc/note_sequence_data.obj", item_list=indices)
    hvo_seq = note_sequence_to_hvo_sequence(ns, drum_mapping=ROLAND_REDUCED_MAPPING, beat_division_factors=[4])
    hvo_seq.to_html_plot("./temp.html", show_figure=True)
    #hvo_seq.save_audio("./temp.wav", sf_path="../hvo_sequence/soundfonts/Standard_Drum_Kit.sf2")
    # 2. Load HVO Pickle
    # hvo = get_pickled_hvos("test/misc/note_sequence_data.obj", item_list=indices)

    # Get the tempo from note_sequence (alternatively from the csv metadata)
    # qpm = ns.tempos[0].qpm

    # Test Data converters
    # 1. get hvo_from_ns ---->   To implement
    # ASAP: THE BUILTIN MAGENTA CONVERTER SEEMS TO BE WORKING WEIRDLY

    # 2. get note_sequence from hvo
    # ns_from_hvo = hvo_to_note_sequence(torch.tensor(hvo_po2), beat_division=4, qpm=qpm)

    # Test Midi Converters
    # 1. Note seq to MIDI
    # save_note_sequence_to_midi(ns_from_hvo, filename="misc/temp/ns_from_hvo.mid")

    # 2. HVO seq to MIDI
    # save_hvo_to_midi(torch.tensor(hvo), filename="misc/temp/hvo_preprocessed.mid", beat_division=4, qpm=qpm,
    #                 midi_track_n=9, max_len_in_beats=8)

    # Test HVO_Sequence Datatype
    """hvo_seq = HVO_Sequence(drum_mapping=ROLAND_REDUCED_MAPPING)
    hvo_seq = hvo_seq.from_note_sequence(ns, beat_division_factors=[4])
    print(len(hvo_seq.hvo))
    fig = hvo_seq.to_html_plot(filename="misc/temp/temp_mix.html")
    hvo_seq.save_audio(sr=44100)
    print(hvo_seq.time_signature)"""
    # hvo_seq.time_signature = {"numerator": 3, "denominator": 2}
    # print(hvo_seq.time_signature)

    # Test Audio Synthesizers
    # audio = save_note_sequence_to_audio(ns_from_hvo, filename="misc/temp/ns_from_hvo.wav", sr=44100,
    #                                     sf_path="utils/soundfonts/Standard_Drum_Kit.sf2")
    """_ = save_note_sequence_to_audio(ns, filename="misc/temp/ns_performance.wav", sr=44100,
                                    sf_path="utils/soundfonts/Standard_Drum_Kit.sf2")"""
    # _ = save_hvo_to_audio(torch.tensor(hvo_po2), filename="misc/temp/hvo_po2.wav", sr=44100,
    #                      sf_path="utils/soundfonts/Standard_Drum_Kit.sf2", beat_division=4,
    #                      qpm=qpm, midi_track_n=9, max_len_in_beats=None)

    # TEST HTML PLOTTERS
    # html_ns_from_hvo = note_sequence_to_html_plot(ns_from_hvo, filename="misc/temp/ns_from_hvo.html")
    # show(html_ns_from_hvo)
    # html_ns_performance = note_sequence_to_html_plot(ns, filename="misc/temp/ns_performance.html")
    # show(html_ns_performance)
    # html_fig = hvo_to_html_plot(torch.tensor(hvo), filename="misc/temp/hvo.html", qpm=qpm)
    # show(html_fig)
    # html_fig2 = hvo_to_html_plot(torch.tensor(hvo_mix), filename="misc/temp/hvo_mix.html", qpm=qpm)
    # show(html_fig2)
