from eval.post_training_evaluations.src.mgeval_rytm_utils import *
import pickle
from copy import deepcopy
from eval.GrooveEvaluator.src.back_compatible_loader import load_evaluator

if __name__ == '__main__':

    gmd_eval = pickle.load(open(
            f"post_training_evaluations/evaluators_monotonic_groove_transformer_v1/"
            f"validation_set_evaluator_run_misunderstood-bush-246_Epoch_26.Eval","rb"))

    down_size = 1024
    final_indices = sample_uniformly(gmd_eval, num_samples=down_size) if down_size < 1024 else list(range(1024))

    # Compile data (flatten styles)
    new_names = {
        "groovae": "GrooVAE",
        "rosy": "Model 1",
        "hopeful": "Model 2",
        "solar": "Model 3",
        "misunderstood": "Model 4",
        # "robust": "Model 5",
        # "colorful": "Model 6",

    }

    evaluators = {
        "groovae":
            pickle.load(open(f"post_training_evaluations/evaluators_monotonic_groove_transformer_v1/"
                             f"validation_set_evaluator_run_groovae.Eval", "rb")),

        "rosy":
            pickle.load(open(f"post_training_evaluations/evaluators_monotonic_groove_transformer_v1/"
                             f"validation_set_evaluator_run_rosy-durian-248_Epoch_26.Eval", "rb")),
        "hopeful":
            pickle.load(open(f"post_training_evaluations/evaluators_monotonic_groove_transformer_v1/"
                             f"validation_set_evaluator_run_hopeful-gorge-252_Epoch_90.Eval", "rb")),
        "solar":
            pickle.load(open(f"post_training_evaluations/evaluators_monotonic_groove_transformer_v1/"
                             f"validation_set_evaluator_run_solar-shadow-247_Epoch_41.Eval", "rb")),
        "misunderstood":
            pickle.load(open(f"post_training_evaluations/evaluators_monotonic_groove_transformer_v1/"
                             f"validation_set_evaluator_run_misunderstood-bush-246_Epoch_26.Eval", "rb")),
        #"robust":
        #    pickle.load(open(f"post_training_evaluations/evaluators_monotonic_groove_transformer_v1/"
        #                      f"validation_set_evaluator_run_robust_sweep_29.Eval", "rb")),
        #"colorful":
        #    pickle.load(open(f"post_training_evaluations/evaluators_monotonic_groove_transformer_v1/"
        #                     f"validation_set_evaluator_run_colorful_sweep_41.Eval", "rb"))
    }

    sets_evals = dict((new_names[key], value) for (key, value) in evaluators.items())

    # compile and flatten features
    feature_sets = {"gmd": flatten_subset_genres(get_gt_feats_from_evaluator(list(sets_evals.values())[0]))}
    feature_sets.update({
        set_name:flatten_subset_genres(get_pd_feats_from_evaluator(eval)) for (set_name, eval) in sets_evals.items()
    })




    # ----- grab selected indices (samples)
    for set_name, set_dict in feature_sets.items():
        for key, array in set_dict.items():
            feature_sets[set_name][key] = array[final_indices]

    # --- remove unnecessary features
    '''just_show = ["Statistical::NoI", "Statistical::Total Step Density", "Statistical::Avg Voice Density",
                        "Statistical::Lowness", "Statistical::Midness", "Statistical::Hiness",
                        "Statistical::Vel Similarity Score", "Statistical::Weak to Strong Ratio",
                        "Syncopation::Lowsync", "Syncopation::Midsync", "Syncopation::Hisync",
                        "Syncopation::Lowsyness", "Syncopation::Midsyness", "Syncopation::Hisyness", "Syncopation::Complexity"]'''
    just_show = None # Show all
    for set_name in feature_sets.keys():
        allowed_analysis = feature_sets[set_name].keys() if just_show is None else just_show
        for key in list(feature_sets[set_name].keys()):
            if key not in allowed_analysis:
                feature_sets[set_name].pop(key)


    # ================================================================
    # ---- Analysis 0:  Accuracy Vs. Precision
    # ----              Also, utiming and velocity analysis
    # from sklearn.metrics import precision_score, accuracy_score
    # ================================================================
    gmd_eval = deepcopy(list(sets_evals.values())[0])
    gmd_eval._prediction_hvos_array = gmd_eval._gt_hvos_array
    absolute_sets_evals = {"GMD": gmd_eval}
    absolute_sets_evals.update(sets_evals)

    n_hits_in_gt = sum(sets_evals[list(sets_evals.keys())[0]].get_ground_truth_hvos_array()[:,:,:9].flatten())
    n_silence_in_gt = sets_evals[list(sets_evals.keys())[0]].get_ground_truth_hvos_array()[:,:,:9].size - n_hits_in_gt
    hit_weight = n_silence_in_gt / n_hits_in_gt
    stats_sets = get_positive_negative_hit_stats(sets_evals, hit_weight=1)

    fig_path = "post_training_evaluations/evaluators_monotonic_groove_transformer_v1/mgeval_results/boxplots"

    group_hit_labels = ['Total Hits', 'True Hits (Matching GMD)', 'False Hits (Different from GMD)']
    gt_hits = stats_sets[list(stats_sets.keys())[0]]['Actual Hits']
    hit_analysis= {'GMD': {'Total Hits': gt_hits, 'True Hits (Matching GMD)': gt_hits, 'False Hits (Different from GMD)': gt_hits}}

    only_leave_these_in_stats = {''}
    for key in list(stats_sets.keys()):
        hit_analysis.update({key: {}})
        for key_ in list(stats_sets[key].keys()):
            if key_ in group_hit_labels:
                hit_analysis[key].update({key_: stats_sets[key][key_]})
                stats_sets[key].pop(key_)

    generate_these = False
    if generate_these is not False:
        boxplot_absolute_measures(hit_analysis, fs=30, legend_fs=20, legend_ncols=8, fig_path=fig_path, show=False, ncols=3,
                                  figsize=(30, 5), color_map="tab20c", filename="Stats_hits", share_legend=True, sharey=True,
                                  show_legend=False)

    generate_these = False
    if generate_these is not False:
        boxplot_absolute_measures(stats_sets, fs=30, legend_fs=10, legend_ncols=4, fig_path=fig_path, show=True, ncols=3,
                                  figsize=(30, 25), color_map="tab20c", filename="Hits_performance",
                                  sharey=False, share_legend=False, shift_colors_by=1,
                                  show_legend=False)

    # TPR -> How many of ground truth hits were correctly predicted
    # FPR -> How many of ground truth silences were predicted as hits
    # to

    generate_these = False
    if generate_these is not False:
        vel_stats_sets = get_positive_negative_vel_stats(absolute_sets_evals)
        boxplot_absolute_measures(vel_stats_sets, fs=30, legend_fs=30, legend_ncols=8, fig_path=fig_path, show=False, ncols=3,
                                  figsize=(30, 10), color_map="tab20c", filename="Stats_vels", share_legend=True, show_legend=False,
                                  shift_colors_by=1)
    generate_these = False
    if generate_these is not False:
        ut_stats_sets = get_positive_negative_utiming_stats(absolute_sets_evals)
        boxplot_absolute_measures(ut_stats_sets, fs=30, legend_fs=10, legend_ncols=4, fig_path=fig_path, show=True, ncols=3,
                                  figsize=(30, 10), color_map="tab20c", filename="Hits_performance",
                                  sharey=False, share_legend=False, shift_colors_by=1,
                                  show_legend=False)


    # ================================================================
    # ---- Analysis 1: Absolute Measures According to
    # Yang, Li-Chia, and Alexander Lerch. "On the evaluation of generative models in music."
    #           Neural Computing and Applications 32.9 (2020): 4773-4784.
    # ================================================================

    # Compile Absolute Measures
    generate_these = True
    if generate_these is not False:
        csv_path = "post_training_evaluations/evaluators_monotonic_groove_transformer_v1/" \
                   "mgeval_results/boxplots/absolute_measures.csv"
        pd_final = get_absolute_measures_for_multiple_sets(feature_sets, csv_file=csv_path)

        fig_path = "post_training_evaluations/evaluators_monotonic_groove_transformer_v1/mgeval_results/boxplots"
        boxplot_absolute_measures(feature_sets, fs=20, legend_fs=14, legend_ncols=4, fig_path=fig_path, show=False, ncols=4,
                                  figsize=(40, 40), color_map="tab20c", sharey=False, share_legend=False)

