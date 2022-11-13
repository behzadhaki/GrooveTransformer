from eval.GrooveEvaluator import load_evaluator
from eval.MultiSetEvaluator import MultiSetEvaluator

# prepare input data
eval_1 = load_evaluator("demos/GrooveEvaluator/examples/latest/test_set_full_misunderstood_bush_246.Eval.bz2")
eval_2 = load_evaluator("demos/GrooveEvaluator/examples/latest/test_set_full_testset_rosy_durian_248.Eval.bz2")

# ignore_feature_keys = ["Statistical::NoI", "Statistical::Total Step Density", "Statistical::NEWWWWW"]
ignore_feature_keys = None

# construct MultiSetEvaluator
msEvaluator = MultiSetEvaluator(
    groove_evaluator_sets={"Model 1": eval_1, "Model 2": eval_2, "Model 3": eval_2},
    # { "groovae": eval_1, "Model 1": eval_2, "Model 2": eval_3 },  # { "groovae": eval_1}
    ignore_feature_keys=None,  # ["Statistical::NoI", "Statistical::Total Step Density", "Statistical::NEWWWWW"]
    reference_set_label="GT",
    anchor_set_label=None,  # "groovae"
    need_pos_neg_hit_score_plots=True,
    need_velocity_distribution_plots=True,
    need_offset_distribution_plots=True,
    need_inter_intra_pdf_plots=True,
    need_kl_oa_plots=True
)

# dump MultiSetEvaluator
msEvaluator.dump("demos/MultiSetEvaluator/misc/inter_intra_evaluator.MSEval.bz2")

# load MultiSetEvaluator
from eval.MultiSetEvaluator import load_multi_set_evaluator
msEvaluator = load_multi_set_evaluator("demos/MultiSetEvaluator/misc/inter_intra_evaluator.MSEval.bz2")

# save statistics
msEvaluator.save_statistics_of_inter_intra_distances(dir_path="demos/MultiSetEvaluator/misc/multi_set_evaluator")

# save inter intra pdf plots
iid_pdfs_bokeh = msEvaluator.get_inter_intra_pdf_plots(
    filename="demos/MultiSetEvaluator/misc/multi_set_evaluator/iid_pdfs.html")

# save kl oa plots
KL_OA_plot = msEvaluator.get_kl_oa_plots(filename="demos/MultiSetEvaluator/misc/multi_set_evaluator")

# get pos neg hit score plots
pos_neg_hit_score_plots = msEvaluator.get_pos_neg_hit_score_plots(
    filename="demos/MultiSetEvaluator/misc/multi_set_evaluator/pos_neg_hit_scores.html")

# get velocity distribution plots
velocity_distribution_plots = msEvaluator.get_velocity_distribution_plots(
    filename="demos/MultiSetEvaluator/misc/multi_set_evaluator/velocity_distributions.html")

# get offset distribution plots
offset_distribution_plots = msEvaluator.get_offset_distribution_plots(
    filename="demos/MultiSetEvaluator/misc/multi_set_evaluator/offset_distributions.html")

# get logging media
logging_media = msEvaluator.get_logging_media(identifier="Analysis X")

# get Some of the logging media
logging_media_partial = msEvaluator.get_logging_media(identifier="Analysis X", need_pos_neg_hit_score_plots=False)

# get logging media and save to files
logging_media_and_saved = msEvaluator.get_logging_media(
    identifier="Analysis X",
    save_directory="demos/MultiSetEvaluator/misc/logging_media")

# get logging media for wandb
logging_media_wandb = msEvaluator.get_logging_media(
    identifier="Analysis X",
    save_directory="demos/MultiSetEvaluator/misc/logging_media",
    prepare_for_wandb=True, need_inter_intra_pdf_plots=False, need_kl_oa_plots=False,
    need_pos_neg_hit_score_plots=True, need_velocity_distribution_plots=True, need_offset_distribution_plots=True)