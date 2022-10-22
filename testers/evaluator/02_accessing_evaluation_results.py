if __name__ == "__main__":

    # Load Evaluator using full path with extension
    from eval.GrooveEvaluator.src.evaluator import load_evaluator
    evaluator_test_set = load_evaluator("path/test_set_full_fname.Eval.bz2")


    # 3.A. Results as Dictionaries or Pandas.DataFrame
    # Quality of Hits
    hit_scores = evaluator_test_set.get_pos_neg_hit_scores(return_as_pandas_df=False)
    statistics_of_hit_scores = evaluator_test_set.get_statistics_of_pos_neg_hit_scores(hit_weight=1,
                                                                                       trim_decimals=1,
                                                                                       csv_file="testers/evaluator/misc/hit_scores.csv")

    # 3.B - Get Statistics of velocity distributions
    velocitiy_distributions = evaluator_test_set.get_velocity_distributions(return_as_pandas_df=True)
    statistics_of_velocitiy_distributions = evaluator_test_set.get_statistics_of_velocity_distributions(
        trim_decimals=1, csv_file="testers/evaluator/misc/vel_stats.csv")
    velocity_MSE_entire_score = evaluator_test_set.get_velocity_MSE(ignore_correct_silences=False)
    velocity_MSE_non_silent_locs = evaluator_test_set.get_velocity_MSE(ignore_correct_silences=True)


    # 3.C - Get Statistics of offset distributions
    offset_distributions = evaluator_test_set.get_offset_distributions(return_as_pandas_df=False)
    statistics_of_offsetocitiy_distributions = evaluator_test_set.get_statistics_of_offset_distributions(
        trim_decimals=1, csv_file="testers/evaluator/misc/offset_stats.csv")
    offset_MSE_entire_score = evaluator_test_set.get_offset_MSE(ignore_correct_silences=False)
    offset_MSE_non_silent_locs = evaluator_test_set.get_offset_MSE(ignore_correct_silences=True)


    # 3.D - Get Rhythmic DIstances
    rhythmic_distances = evaluator_test_set.get_rhythmic_distances_of_pred_to_gt(return_as_pandas_df=False)
    rhythmic_distances_statistics_df = evaluator_test_set.get_statistics_of_rhythmic_distances_of_pred_to_gt(
        tag_by_identifier=False, csv_dir="testers/evaluator/misc/distances", trim_decimals=3)

    # 3.E - Get Global Features
    global_features = evaluator_test_set.get_global_features_values(return_as_pandas_df=False)
    get_statistics_of_global_features_df = evaluator_test_set.get_statistics_of_global_features(
        calc_gt=True, calc_pred=True, csv_file="testers/evaluator/misc/global_features_statistics.csv", trim_decimals=3)

