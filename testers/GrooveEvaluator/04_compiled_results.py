if __name__ == "__main__":

    # Load Evaluator using full path with extension
    from eval.GrooveEvaluator import load_evaluator
    evaluator_test_set = load_evaluator("testers/GrooveEvaluator/examples/test_set_full_colorful_sweep_41.Eval.bz2")

    # get logging media associated with default plot flags (need_
    logging_media = evaluator_test_set.get_logging_media()

    # get piano rolls
    # get logging media formatted for wandb
    logging_media_wandb = evaluator_test_set.get_logging_media(prepare_for_wandb=True)