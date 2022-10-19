if __name__ == "__main__":

    # Load Evaluator using full path with extension
    from eval.GrooveEvaluator.src.evaluator import load_evaluator
    evaluator_test_set = load_evaluator("path/test_set_full_fname.Eval.bz2")


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
    global_features = evaluator_test_set.get_global_features_values()
    get_statistics_of_global_features_df = evaluator_test_set.get_statistics_of_global_features(
        calc_gt=True, calc_pred=True, csv_file="misc/global_features_statistics.csv", trim_decimals=3)

    # 2.5 - Calculate rhythmic distances of gt and pred patterns
    from eval.GrooveEvaluator.src.plotting_utils import tabulated_violin_plot
    rhythmic_distances = evaluator_test_set.get_rhythmic_distances_of_pred_to_gt(return_as_pandas_df=True)
    rhythmic_distances_statistics_df = evaluator_test_set.get_statistics_of_rhythmic_distances_of_pred_to_gt(
        tag_by_identifier=False, csv_dir="misc/distances", trim_decimals=3)
    tabs = tabulated_violin_plot(rhythmic_distances, save_path="misc/violinplots/rhythmDistnces", kernel_bandwidth=0.05, width=1200, height=800)









    # 2.5 - Get Pos/Neg Hit statistics
    hit_scores = evaluator_test_set.get_pos_neg_hit_scores()
    statistics_of_hit_scores = evaluator_test_set.get_statistics_of_pos_neg_hit_scores(hit_weight=1, trim_decimals=1, csv_file="misc/hit_scores.csv")
    from eval.GrooveEvaluator.src.plotting_utils import tabulated_violin_plot
    tabs = tabulated_violin_plot(hit_scores, save_path="misc/violinplots/hits_stats", kernel_bandwidth=0.05, width=1200, height=800)

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

    # -----------------  Getting WandB data ----------------- #
    results = evaluator_test_set.get_wandb_logging_media(need_groundTruth=True)    # include ground truth data
    # results = evaluator_test_set.get_wandb_logging_media(need_groundTruth=False)   # exclude ground truth data



