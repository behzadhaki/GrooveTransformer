if __name__ == "__main__":

    # Load Evaluator using full path with extension
    from eval.GrooveEvaluator import load_evaluator
    # evaluator_test_set = load_evaluator("demos/GrooveEvaluator/examples/test_set_full_GT_without_Predictions.Eval.bz2")
    evaluator_test_set = load_evaluator("demos/GrooveEvaluator/examples/test_set_full_colorful_sweep_41.Eval.bz2")

    # ==================================================================================================
    # 3.1 Accessing Results
    # ==================================================================================================

    # 3.A. Results as Dictionaries or Pandas.DataFrame
    # Quality of Hits
    hit_scores = evaluator_test_set.get_pos_neg_hit_scores(return_as_pandas_df=False)
    statistics_of_hit_scores = evaluator_test_set.get_statistics_of_pos_neg_hit_scores(hit_weight=1,
                                                                                       trim_decimals=1,
                                                                                       csv_file="demos/evaluator/misc/GrooveEvaluator/hit_scores.csv")

    # 3.B - Get Statistics of velocity distributions
    velocitiy_distributions = evaluator_test_set.get_velocity_distributions(return_as_pandas_df=True)
    statistics_of_velocitiy_distributions = evaluator_test_set.get_statistics_of_velocity_distributions(
        trim_decimals=1, csv_file="demos/evaluator/misc/GrooveEvaluator/vel_stats.csv")
    velocity_MSE_entire_score = evaluator_test_set.get_velocity_MSE(ignore_correct_silences=False)
    velocity_MSE_non_silent_locs = evaluator_test_set.get_velocity_MSE(ignore_correct_silences=True)


    # 3.C - Get Statistics of offset distributions
    offset_distributions = evaluator_test_set.get_offset_distributions(return_as_pandas_df=False)
    statistics_of_offsetocitiy_distributions = evaluator_test_set.get_statistics_of_offset_distributions(
        trim_decimals=1, csv_file="demos/evaluator/misc/GrooveEvaluator/offset_stats.csv")
    offset_MSE_entire_score = evaluator_test_set.get_offset_MSE(ignore_correct_silences=False)
    offset_MSE_non_silent_locs = evaluator_test_set.get_offset_MSE(ignore_correct_silences=True)


    # 3.D - Get Rhythmic DIstances
    rhythmic_distances = evaluator_test_set.get_rhythmic_distances_of_pred_to_gt(return_as_pandas_df=False)
    rhythmic_distances_statistics_df = evaluator_test_set.get_statistics_of_rhythmic_distances_of_pred_to_gt(
        tag_by_identifier=False, csv_dir="demos/evaluator/misc/GrooveEvaluator/distances", trim_decimals=3)

    # 3.E - Get Global Features
    global_features = evaluator_test_set.get_global_features_values(return_as_pandas_df=False)
    get_statistics_of_global_features_df = evaluator_test_set.get_statistics_of_global_features(
        calc_gt=True, calc_pred=True, csv_file="demos/evaluator/misc/GrooveEvaluator/global_features_statistics.csv", trim_decimals=3)


    # ==================================================================================================
    # 3.2 Plotting the Results
    # ==================================================================================================
    # 3.A. Results as Dictionaries or Pandas.DataFrame
    # Quality of Hits
    pos_neg_hit_plots = evaluator_test_set.get_pos_neg_hit_plots(
        save_path="demos/evaluator/misc/GrooveEvaluator/pos_neg_hit_plots.html", prepare_for_wandb=True,
        plot_width=1200, plot_height=800,
        kernel_bandwidth=0.05)

    # 3.B - Get Statistics of velocity distributions
    velocity_plots = evaluator_test_set.get_velocity_distribution_plots(
        save_path="demos/evaluator/misc/GrooveEvaluator/velocity_plots.html", plot_width=1200, plot_height=800, prepare_for_wandb=True,
        kernel_bandwidth=0.05)

    # 3.C - Get Statistics of offset distributions
    offset_plots = evaluator_test_set.get_velocity_distribution_plots(
        save_path="demos/evaluator/misc/GrooveEvaluator/offset_plots.html", plot_width=1200, plot_height=800, prepare_for_wandb=True,
        kernel_bandwidth=0.05)

    # 3.D - Get Rhythmic DIstances
    rhythmic_distances_plot = evaluator_test_set.get_rhythmic_distances_of_pred_to_gt_plot(
        save_path="demos/evaluator/misc/GrooveEvaluator/rhythmic_distances_plots.html", plot_width=1200, plot_height=800, prepare_for_wandb=True,
        kernel_bandwidth=0.05)

    # 3.E - Get Global Features
    evaluator_test_set.get_global_features_plot(only_combined_data_needed=False, prepare_for_wandb=True,
                                                save_path="demos/evaluator/misc/GrooveEvaluator/global_features_all.html",
                                                plot_width=1200, plot_height=800,
                                                kernel_bandwidth=0.05)
    evaluator_test_set.get_global_features_plot(only_combined_data_needed=True, prepare_for_wandb=True,
                                                save_path="demos/evaluator/misc/GrooveEvaluator/global_features_combinedOnly.html",
                                                plot_width=1200, plot_height=800,
                                                kernel_bandwidth=0.05)

    # 3.F get heatmaps
    evaluator_test_set.get_velocity_heatmaps(
        s=(2, 4), bins=[32 * 4, 64], regroup_by_drum_voice=True, prepare_for_wandb=True,
        save_path="demos/evaluator/misc/GrooveEvaluator/velocity_heatmaps.html")

    # 3.G get KL/OA plots
    kl_plots = evaluator_test_set.get_kl_oa_inter_intra_plots()