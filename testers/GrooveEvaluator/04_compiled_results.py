if __name__ == "__main__":

    # Load Evaluator using full path with extension
    from eval.GrooveEvaluator import load_evaluator
    evaluator_test_set = load_evaluator("testers/GrooveEvaluator/examples/test_set_full_colorful_sweep_41.Eval.bz2")

    # get logging media associated with default plot flags
    logging_media = evaluator_test_set.get_logging_media()

    # get and store logging media associated with custom plot flags
    logging_media_saved = evaluator_test_set.get_logging_media(save_directory="testers/GrooveEvaluator/misc/logged")

    # get piano rolls
    # get logging media formatted for wandb
    logging_media_wandb = evaluator_test_set.get_logging_media(prepare_for_wandb=True,
                                                               save_directory="testers/GrooveEvaluator/misc/logged")