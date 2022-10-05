##################################################################################################
#
#                           Template dictionaries for base_loaders
#           The filters are list of dictionaries with feature fields as keys
#                               and filter values as list of values
#
#       Example:
#           GROOVEMIDI_FILTER_TEMPLATE = [  {"style_primary": [["afrobeat", "afrocuban"], "bpm":[(60, 90)]},
#                                           {"style_primary": [["rock"], "bpm":[(50, 100), (120, 160)]},
#
#
#           Using these filters, two subsets will be returned:
#                   subset 1 --> afrobeat and afrocuban samples with BPM between 60 and 90
#                   subset 2 --> rock samples with BPM between 50-100 or 120-160
#
#
##################################################################################################



GROOVEMIDI_FILTER_TEMPLATE = [{

    "drummer": None,                                    # ["drummer1", ..., and/or "session9"]
    "session": None,                                    # ["session1", "session2", and/or "session3"]
    "loop_id": None,
    "master_id": None,
    "style_primary": None,                              #   [["afrobeat"], ["afrocuban"], ["blues"], ["country"],
                                                        #   ["dance"],
                                                        #   ["funk"], ["gospel"], ["highlife"], ["hiphop"], ["jazz"],
                                                        #   ["latin"], ["middleeastern"], ["neworleans"], ["pop"],
                                                        #   ["punk"], ["reggae"], ["rock"], ["soul"]]
    "style_secondary": None,
    "bpm": None,                                        # [(range_0_lower_bound, range_0_upper_bound), ...,
                                                        #   (range_n_lower_bound, range_n_upper_bound)]
    "beat_type": None,                                  # ["beat" or "fill"]
    "time_signature": None,                             # ["4-4", "3-4", "6-8"]
    "full_midi_filename": None,                         # list_of full_midi_filenames
    "full_audio_filename": None,                        # list_of full_audio_filename
    "number_of_instruments": None,                      # [(n_instruments_lower_bound, n_instruments_upper_bound), ...,
                                                        #  (n_instruments_lower_bound, n_instruments_upper_bound)]
}]

FILTER = { "gmd":
    {

    "drummer": None,                                    # ["drummer1", ..., and/or "session9"]
    "session": None,                                    # ["session1", "session2", and/or "session3"]
    "loop_id": None,
    "master_id": None,
    "style_primary": None,                              #   [["afrobeat"], ["afrocuban"], ["blues"], ["country"],
                                                        #   ["dance"],
                                                        #   ["funk"], ["gospel"], ["highlife"], ["hiphop"], ["jazz"],
                                                        #   ["latin"], ["middleeastern"], ["neworleans"], ["pop"],
                                                        #   ["punk"], ["reggae"], ["rock"], ["soul"]]
    "style_secondary": None,
    "bpm": None,                                        # [(range_0_lower_bound, range_0_upper_bound), ...,
                                                        #   (range_n_lower_bound, range_n_upper_bound)]
    "beat_type": ["beat"],                                  # ["beat" or "fill"]
    "time_signature": ["4-4"],                             # ["4-4", "3-4", "6-8"]
    "full_midi_filename": None,                         # list_of full_midi_filenames
    "full_audio_filename": None,                        # list_of full_audio_filename
    "number_of_instruments": None,                      # [(n_instruments_lower_bound, n_instruments_upper_bound), ...,
                                                        #  (n_instruments_lower_bound, n_instruments_upper_bound)]
    }
}