from hvo_sequence.hvo_seq import HVO_Sequence
from hvo_sequence.drum_mappings import ROLAND_REDUCED_MAPPING, Groove_Toolbox_5Part_keymap, Groove_Toolbox_3Part_keymap
import fluidsynth
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
    hits = np.random.rand(n_steps, n_voices)
    hits[hits > 0.5] = 1
    hits[hits <= 0.5] = 0

    offs = hits * np.zeros((n_steps, n_voices))

    # Add hvo score to hvo_seq instance
    vels = hits * np.ones((n_steps, n_voices))

    hvo_seq.hvo = np.concatenate((hits, vels, offs), axis=1)

    hvo_seq.save_audio()

    hvo_seq.to_html_plot(show_figure=True)
    # ######################################################################
    #
    #           Rhythmic Features::Statistical Features Related
    #
    # ######################################################################

    hvo_seq.get_number_of_active_voices()

    hvo_seq.get_total_step_density()

    hvo_seq.get_average_voice_density()

    hvo_seq.get_hit_density_for_voice(0)

    hvo_seq.get_velocity_intensity_mean_stdev_for_voice(0)
    hvo_seq.get_velocity_intensity_mean_stdev_for_voice(1)

    hvo_seq.get_offset_mean_stdev_for_voice(3)

    hvo_seq.get_lowness_midness_hiness()

    hvo_seq.get_velocity_score_symmetry()

    hvo_seq.get_total_weak_to_strong_ratio()

    hvo_seq.get_polyphonic_offset_mean_stdev()

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

    hvo_seq.get_monophonic_syncopation_for_voice(0)
    hvo_seq.get_combined_syncopation()
    hvo_seq.get_witek_polyphonic_syncopation()
    hvo_seq.get_low_mid_hi_syncopation_info()
    hvo_seq.get_complexity_for_voice(0)
    hvo_seq.get_total_complexity()

    hvo_seq.get_witek_polyphonic_syncopation()
    hvo_seq.get_combined_syncopation()
    hvo_seq.get_low_mid_hi_syncopation_info()
    hvo_seq.get_complexity_for_voice(0)
    hvo_seq.get_total_complexity()

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
