from data.dataLoaders import load_gmd_hvo_sequences

if __name__ == "__main__":
    # 2.1 - Load test set dataset
    dataset_setting_json_path = "dataset_setting.json"

    test_set = load_gmd_hvo_sequences(
        "data/gmd/resources/storedDicts/groove_2bar-midionly.bz2pickle",
        "gmd", dataset_setting_json_path, "test")

    # 2.1 - Create the Subsetter filters to divide up the dataset into subsets
    list_of_filter_dicts_for_subsets = []
    styles = [
        "afrobeat", "afrocuban", "blues", "country", "dance", "funk", "gospel", "highlife", "hiphop", "jazz",
        "latin", "middleeastern", "neworleans", "pop", "punk", "reggae", "rock", "soul"]
    for style in styles:
        list_of_filter_dicts_for_subsets.append(
            {"style_primary": [style]} #, "beat_type": ["beat"], "time_signature": ["4-4"]}
        )

    # 2.2 - Instantiate the evaluator
    from eval.GrooveEvaluator.src.evaluator import Evaluator
    evaluator_test_set = Evaluator(
        test_set,
        list_of_filter_dicts_for_subsets=list_of_filter_dicts_for_subsets,
        _identifier="test_set_full",
        n_samples_to_use=20, #-1,
        max_hvo_shape=(32, 27),
        need_heatmap=True,
        need_global_features=True,
        need_audio=True,
        need_piano_roll=True,
        n_samples_to_synthesize=5,   # "all",
        n_samples_to_draw_pianorolls=5,    # "all",
        disable_tqdm=False
    )

    # 2.3 -      Save Evaluator
    evaluator_test_set.dump(path="path", fname="fname")

    # 2.3 -      Load Evaluator using full path with extension
    from eval.GrooveEvaluator.src.evaluator import load_evaluator
    evaluator_test_set = load_evaluator("path/test_set_full_fname.Eval.bz2")

    # 2.4.1 - get ground truth hvo pianoroll scores
    evaluator_test_set.get_ground_truth_hvos_array()

    # 2.4.1 - get ground truth monotonic grooves
    import numpy as np
    input = np.array(
    [hvo_seq.flatten_voices(voice_idx=2) for hvo_seq in evaluator_test_set.get_ground_truth_hvo_sequences()])

    # 2.4.2 Pass the ground truth data to the model
    # predicted_hvos_array = model.predict(input)
    predicted_hvos_array = evaluator_test_set.get_ground_truth_hvos_array()   # This is here just to make sure the code doesnt rely on the model here

    # 2.4.3 - Add the predictions to the evaluator
    evaluator_test_set.add_predictions(predicted_hvos_array)


    # 2.5 - Get the results for general inspection
    # _gt_logging_data, _predicted_logging_data = evaluator_test_set.get_logging_dict()
    _predicted_logging_data = evaluator_test_set.get_logging_dict(need_groundTruth=False)

    # 2.5 Displaying Bokeh plot results
    from bokeh.io import show
    show(_predicted_logging_data['velocity_heatmaps'])
    show(_predicted_logging_data['global_feature_pdfs'])
    show(_predicted_logging_data['piano_rolls'])

    # save audio files
    import os
    import soundfile as sf
    def save_wav_file(filename, data, sample_rate):
        # make directories if filename has directories
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        # save file
        sf.write(filename, data, sample_rate)

    sample_audio_tuple = _predicted_logging_data['captions_audios'][0]
    fname = sample_audio_tuple[0]
    data = sample_audio_tuple[1]
    save_wav_file(os.path.join("misc", fname), data, 44100)

    # 2.5 - Exporting gt and predictions to midi files
    evaluator_test_set.export_to_midi(need_gt=True, need_pred=True, directory="misc")

    # 2.5 - Exporting boxplot statistics of gt OR predictions to csv files and a pandas dataframe
    df = evaluator_test_set.get_statistics_of_global_features(calc_gt=True, calc_pred=True, csv_file="misc/global_features_statistics.csv", trim_decimals=3)

    # 2.5 - Calculate rhythmic distances of gt and pred patterns
    rhythmic_distances = evaluator_test_set.get_statistics_of_rhythmic_distances_of_pred_to_gt(tag_by_identifier=False, csv_dir="misc/distances", trim_decimals=3)

    # 2.5 - Get Pos/Neg Hit statistics
    hit_scores = evaluator_test_set.get_statistics_of_pos_neg_hit_scores(hit_weight=1, trim_decimals=1, csv_file="misc/hit_scores.csv")

    # 2.5 - Get Statistics of velocity distributions
    velocitiy_distributions = evaluator_test_set.get_velocity_distributions()
    statistics_of_velocitiy_distributions = evaluator_test_set.get_statistics_of_velocity_distributions(trim_decimals=1, csv_file="misc/vel_stats.csv")
    velocity_MSE_entire_score = evaluator_test_set.get_velocity_MSE(ignore_correct_silences=False)
    velocity_MSE_non_silent_locs = evaluator_test_set.get_velocity_MSE(ignore_correct_silences=True)

    # 2.5 - Get Statistics of utiming distributions
    offset_distributions = evaluator_test_set.get_offset_distributions()
    statistics_of_offsetocitiy_distributions = evaluator_test_set.get_statistics_of_offset_distributions(
        trim_decimals=1, csv_file="misc/offset_stats.csv")
    offset_MSE_entire_score = evaluator_test_set.get_offset_MSE(ignore_correct_silences=False)
    offset_MSE_non_silent_locs = evaluator_test_set.get_offset_MSE(ignore_correct_silences=True)

    # 2.5 -
    from eval.GrooveEvaluator.src.evaluator import load_evaluator
    evaluator_test_set = load_evaluator("path/test_set_full_fname.Eval.bz2")
    predicted_hvos_array = evaluator_test_set.get_ground_truth_hvos_array()  # This is here just to make sure the code doesnt rely on the model here
    evaluator_test_set.add_predictions(predicted_hvos_array)
    offsetocitiy_distributions = evaluator_test_set.get_offset_distributions()

    # -----------------  Getting WandB data ----------------- #
    results = evaluator_test_set.get_wandb_logging_media(need_groundTruth=True)    # include ground truth data
    # results = evaluator_test_set.get_wandb_logging_media(need_groundTruth=False)   # exclude ground truth data



