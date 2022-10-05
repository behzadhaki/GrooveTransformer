################################
#
#  DICT OF FEATURES TO EXTRACT
#   EMBEDDED IN HVO_Sequence
#
################################

FEATURES_TO_EXTRACT = {
    "Statistical":
        {
            "NoI": True,                            # number of instruments (HVO_Sequence.get_number_of_active_voices)
            "Total Step Density": True,             # (HVO_Sequence.get_total_step_density)
            "Avg Voice Density": True,              # (HVO_Sequence.get_average_voice_density)
            "Lowness": True,                        # (HVO_Sequence.get_lowness_midness_hiness)
            "Midness": True,                        # Same as above
            "Hiness": True,                         # Same as above
            "Vel Similarity Score": True,           # (HVO_Sequence.get_velocity_score_symmetry)
            "Weak to Strong Ratio": True,           # (HVO_Sequence.get_total_weak_to_strong_ratio)
            "Poly Velocity Mean": True,             # (HVO_Sequence.get_polyphonic_velocity_mean_stdev)
            "Poly Velocity std": True,              # Same as above
            "Poly Offset Mean": True,               # (HVO_Sequence.get_polyphonic_offset_mean_stdev)
            "Poly Offset std": True                 # Same as above
        },

    "Syncopation":
        {
            "Combined": True,                       # combines monophonic syncs (HVO_Sequence.get_combined_syncopation)
            "Polyphonic": True,                     # Witek Poly Sync (HVO_Sequence.get_witek_polyphonic_syncopation)
            "Lowsync": True,                        # (HVO_Sequence.get_low_mid_hi_syncopation_info)
            "Midsync": True,                        # Same as above
            "Hisync": True,                         # Same as above
            "Lowsyness": True,                      # Same as above
            "Midsyness": True,                      # Same as above
            "Hisyness": True,                       # Same as above
            "Complexity": True                      # (HVO_Sequence.get_total_complexity)
        },

    "Auto-Correlation":
        {
            "Skewness": True,                       # (HVO_Sequence.get_velocity_autocorrelation_features)
            "Max": True,                            # Same as above
            "Centroid": True,                       # Same as above
            "Harmonicity": True                     # Same as above
        },

    "Micro-Timing":
        {
            "Swingness": True,                      # (HVO_Sequence.swingness)
            "Laidbackness": True,                   # (HVO_Sequence.laidbackness)
            "Accuracy": True,                       # (HVO_Sequence.get_timing_accuracy)
        }
}
