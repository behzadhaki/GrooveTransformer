from hvo_sequence.hvo_seq import HVO_Sequence
from hvo_sequence.drum_mappings import ROLAND_REDUCED_MAPPING, Groove_Toolbox_5Part_keymap, Groove_Toolbox_3Part_keymap

import numpy as np
from copy import deepcopy

if __name__ == "__main__":

    # Create an instance of a HVO_Sequence
    hvo_seq = HVO_Sequence(drum_mapping=ROLAND_REDUCED_MAPPING)

    # Add a time_signature
    hvo_seq.add_time_signature(0, 4, 4, [4])

    # Add two tempos
    hvo_seq.add_tempo(0, 120)

    n_steps = 16
    n_voices = 9

    # Create a empty hvo
    hits = np.zeros((n_steps, n_voices))
    offs = hits * np.zeros((n_steps, n_voices))

    # Add hvo score to hvo_seq instance

    #########################
    # Test Set 1: For these create a four on the floor kick/snare pattern with
    #             hihats steps 2, 6, 10, 14
    #             vels all one except second snare which is 0.5
    #             offsets zero --> fully quantized
    #

    hits[0::4, 0] = 1
    hits[4::8, 1] = 1
    hits[2::4, 2] = 1
    vels = hits * np.ones((n_steps, n_voices))
    vels[12, 1] = 0.5

    hvo_seq.hvo = np.concatenate((hits, vels, offs), axis=1)

    #hvo_seq.to_html_plot(show_figure=True)

    # ######################################################################
    #
    #           Rhythmic Features::Statistical Features Related
    #
    # ######################################################################

    assert hvo_seq.get_number_of_active_voices() == 3   # 3 as there are only kick/snare and hats

    assert hvo_seq.get_total_step_density() == 0.5      # half the steps have events hence 0.5

    assert hvo_seq.get_average_voice_density() == (1/9+1/9+2/9+1/9)/8
    # 1/9 for steps with only one kick or one hat, 2/9 for steps with kick and hat

    assert hvo_seq.get_hit_density_for_voice(0) == 4 / 16   # 4 kicks over 16 steps
    assert hvo_seq.get_hit_density_for_voice(1) == 2 / 16   # 2 snares over 16 steps
    assert hvo_seq.get_hit_density_for_voice(2) == 4 / 16   # 4 hats over 16 steps
    assert hvo_seq.get_hit_density_for_voice(3) == 0        # 0 onsets in all other voices

    assert hvo_seq.get_velocity_intensity_mean_stdev_for_voice(0) == (1, 0) # kick velocities are all 1
    assert hvo_seq.get_velocity_intensity_mean_stdev_for_voice(1) == (0.75, 0.25)  # Snare velocities are [1, 0.5]
    assert hvo_seq.get_velocity_intensity_mean_stdev_for_voice(0) == (1, 0) # hihat velocities are all 1

    # Set offsets to normal sampled
    hvo_seq.hvo = np.concatenate((hits, vels,  hits * (np.random.rand(n_steps, n_voices) - 0.5)), axis=1)
    assert hvo_seq.get_offset_mean_stdev_for_voice(0) != (0, 0)       # kick snare hats shouldnt have mean 0
    assert hvo_seq.get_offset_mean_stdev_for_voice(1) != (0, 0)
    assert hvo_seq.get_offset_mean_stdev_for_voice(2) != (0, 0)
    assert hvo_seq.get_offset_mean_stdev_for_voice(3) == (0, 0)       # Other instrument offsets are zero

    # Restore back to 0 offsets
    hvo_seq.hvo = np.concatenate((hits, vels, offs), axis=1)
    assert hvo_seq.get_offset_mean_stdev_for_voice(0) == (0, 0)       # Now kick ... are also zero offsets
    assert hvo_seq.get_offset_mean_stdev_for_voice(1) == (0, 0)
    assert hvo_seq.get_offset_mean_stdev_for_voice(2) == (0, 0)

    assert hvo_seq.get_lowness_midness_hiness() == (0.4, 0.2, 0.4)    # hiness = number of hi notes over total
    # if coinciding hits exist at steps within each of three streams, they are counted only once
    # hence, in such cases sum of lowness midness hiness will be less than one
    # as these values are calculate relative to total number of hits in the complete multivoice hvo
    # rather than the total hits in compressed versions in low mid hi streams
    hits[0::2, -1] = 1
    hvo_seq.hvo = np.concatenate((hits, vels,  hits * (np.random.rand(n_steps, n_voices) - 0.5)), axis=1)
    assert sum(hvo_seq.get_lowness_midness_hiness()) != 1

    # remove extra voice from previous test and set all velocities to 1
    hits[:, -1] = 0
    vels = hits * np.ones((n_steps, n_voices))
    hvo_seq.hvo = np.concatenate((hits, vels, offs), axis=1)
    assert hvo_seq.get_velocity_score_symmetry() == 1.0, "1 because first half and second half are the same"

    assert hvo_seq.get_total_weak_to_strong_ratio() == 4/6      # six onsets on beats and 4 on other positions

    # Set kick velocity to 1, snare to 0.5 and hat to .25
    # Set kick offsets to 0, snare to 0.5 and hat to -0.25
    vels[:, 0] = 1
    vels[:, 1] = 0.5
    vels[:, 2] = 0.25
    offs[:, 1] = 0.5
    offs[:, 2] = -0.25
    hvo_seq.hvo = np.concatenate((hits, vels, offs), axis=1)
    vel_values = np.array([1, 1, 1, 1, .5, .5, .25, .25, .25, .25])
    assert hvo_seq.get_polyphonic_velocity_mean_stdev() == (vel_values.mean(), vel_values.std())
    off_values = np.array([0, 0, 0, 0, .5, .5, -0.25, -0.25, -0.25, -0.25])
    assert hvo_seq.get_polyphonic_offset_mean_stdev() == (off_values.mean(), off_values.std())

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

    # Start with a quantized pattern same as before
    # make sure hits are all on beat positions (no syncopation)
    hits[0::4, 0] = 1
    hits[4::8, 1] = 1
    hits[0::4, 2] = 1
    vels = hits * np.ones((n_steps, n_voices))
    offs = np.zeros_like(hits)
    hvo_seq.hvo = np.concatenate((hits, vels, offs), axis=1)

    # because of quantization all syncopation values should be zero
    assert hvo_seq.get_monophonic_syncopation_for_voice(0) == 0.0
    assert hvo_seq.get_combined_syncopation() == 0.0
    assert hvo_seq.get_witek_polyphonic_syncopation() == 0.0
    assert sum(list(hvo_seq.get_low_mid_hi_syncopation_info().values())) == 0.0
    # sqrt(density^2+syncopation^2) = sqrt((4 kicks in 16 steps)^2+ (syncopation =0)^2)
    assert hvo_seq.get_complexity_for_voice(0) == np.sqrt((4/16)**2+(0)**2)
    hvo_seq.get_total_complexity()

    # now move pattern 1 step to right and check sync values
    rotated_hits = deepcopy(hits)
    rotated_hits[:, 0] = np.roll(rotated_hits[:, 0], 1, axis=0)
    rotated_hits[:, 2] = np.roll(rotated_hits[:, 0], 5, axis=0)
    rotated_vels = np.roll(vels, 1, axis=0)
    rotated_offs = np.roll(offs, 1, axis=0)
    hvo_seq.hvo = np.concatenate((rotated_hits, rotated_vels, rotated_offs), axis=1)
    print(hvo_seq.get_witek_polyphonic_syncopation())
    print(hvo_seq.get_combined_syncopation())
    print(hvo_seq.get_low_mid_hi_syncopation_info())
    print(hvo_seq.get_complexity_for_voice(0))
    print(hvo_seq.get_total_complexity())

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

    print(hvo_seq.get_velocity_autocorrelation_features())

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

    # Start with a quantized pattern same as before
    # make sure hits are all on beat positions (no syncopation)
    hits[1::4, 0] = 1
    hits[5::8, 1] = 1
    hits[1::4, 2] = 1
    vels = hits * np.ones((n_steps, n_voices))
    offs = hits * 0
    hvo_seq.hvo = np.concatenate((hits, vels, offs), axis=1)
    print(hvo_seq.swingness())
    print(hvo_seq.laidbackness())
    print(hvo_seq.get_timing_accuracy())

    # now give it max swing (by offsetting events at steps 1::4 or 3::4 by 0.5)
    offs = hits * 0.5
    hvo_seq.hvo = np.concatenate((hits, vels, offs), axis=1)
    print(hvo_seq.swingness())
    print(hvo_seq.laidbackness())
    print(hvo_seq.get_timing_accuracy())
