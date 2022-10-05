from hvo_sequence.hvo_seq import HVO_Sequence, empty_like
from hvo_sequence.drum_mappings import ROLAND_REDUCED_MAPPING, Groove_Toolbox_5Part_keymap, GM1_FULL_MAP

import numpy as np


if __name__ == "__main__":

    # Create an instance of a HVO_Sequence
    hvo_seq = HVO_Sequence(drum_mapping=ROLAND_REDUCED_MAPPING)

    # Add two time_signatures
    hvo_seq.add_time_signature(0, 4, 4, [4])
    # hvo_seq.add_time_signature(13, 6, 8, [3,2])

    # Add two tempos
    hvo_seq.add_tempo(0, 50)

    #hvo_seq.add_tempo(12, 20)  # Tempo Change at the beginning of second bar

    # Create a random hvo
    hits = np.random.randint(0, 2, (16, 9))
    vels = hits * np.random.rand(16, 9)
    # vels = hit * np.zeros((16, 9))
    offs = hits * (np.random.rand(16, 9) -0.5)
    # offs = hits * np.zeros_like(hits)

    # Add hvo score to hvo_seq instance
    hvo_bar = np.concatenate((hits, vels, offs), axis=1)
    hvo_seq.hvo = np.concatenate((hvo_bar, hvo_bar), axis=0)

    # Micro-timing features
    """print(hvo_seq.laidbackness(
        kick_key_in_drum_mapping="KICK",
        snare_key_in_drum_mapping="SNARE",
        hihat_key_in_drum_mapping="HH_CLOSED"))"""

    print(hvo_seq.get_timing_accuracy())

    acorr = hvo_seq.get_total_autocorrelation_curve()
    print(hvo_seq.get_velocity_autocorrelation_features())
    print(hvo_seq.get_velocity_score_symmetry())
    #print(hvo_seq.get_velocity_intensity_mean_stdev())

    #hvo_seq.to_html_plot(show_figure=True, filename="temp.html")

    hvo_seq_5part = hvo_seq.convert_to_alternate_mapping(tgt_drum_mapping=Groove_Toolbox_5Part_keymap)
    #hvo_seq_5part.to_html_plot(show_figure=True, filename="temp.html")

    #print(hvo_seq.get_bar_beat_hvo(hvo_str="hvo"))
    # Returns flattened hvo (or ho) vector
    #flat_hvo = hvo_seq.flatten_voices()
    #flat_hvo_voice_2 = hvo_seq.flatten_voices(voice_idx=2)
    #flat_hvo_no_vel = hvo_seq.flatten_voices(get_velocities=False)
    #flat_hvo_one_voice = hvo_seq.flatten_voices(reduce_dim=True)
    #flat_hvo_one_voice_no_vel = hvo_seq.flatten_voices(get_velocities=False, reduce_dim=True)

    # Plot, synthesize and export to midi
    #hvo_seq.to_html_plot(show_figure=True)
    #hvo_seq.save_audio()
    #hvo_seq.save_hvo_to_midi()

    # print(hvo_seq.steps_per_beat_per_segments)
    # print(hvo_seq.grid_lines)
    # print(hvo_seq.grid_lines.shape)
    # print(hvo_seq.n_beats_per_segments)
    # hvo_seq.major_and_minor_grid_lines
    #

    oh = hvo_seq.get('oh')

    h = hvo_seq.hits
    v = hvo_seq.velocities
    o = hvo_seq.offsets

    # Reset voices
    hvo_reset, hvo_out_voices = hvo_seq.reset_voices(voice_idx=[0,1])
    print(hvo_seq.hvo[:10,0])
    print(hvo_reset.hvo[:10,0])
    print(hvo_out_voices.hvo[:10,0])

    # Remove random events
    hvo_reset, hvo_out_voices = hvo_seq.remove_random_events()


    #STFT
    #hvo_seq.stft(plot=True)
    #mel_spectrogram
    #hvo_seq.mel_spectrogram(plot=True)


