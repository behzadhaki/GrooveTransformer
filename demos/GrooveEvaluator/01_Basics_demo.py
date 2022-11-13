from data import load_gmd_hvo_sequences

if __name__ == "__main__":
    # 2.1 - Load test set dataset
    dataset_setting_json_path = "data/dataset_json_settings/4_4_Beats_gmd.json"

    test_set = load_gmd_hvo_sequences(
        dataset_setting_json_path, "test")

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
    from eval.GrooveEvaluator import Evaluator
    evaluator_test_set = Evaluator(
        test_set,
        list_of_filter_dicts_for_subsets=list_of_filter_dicts_for_subsets,
        _identifier="test_set_full",
        n_samples_to_use=20, #-1,
        max_hvo_shape=(32, 27),
        need_hit_scores=True,
        need_velocity_distributions=True,
        need_offset_distributions=True,
        need_rhythmic_distances=True,
        need_heatmap=True,
        need_global_features=True,
        need_audio=True,
        need_piano_roll=True,
        n_samples_to_synthesize_and_draw=5,   # "all",
        disable_tqdm=False
    )

    # 2.3.1 - get ground truth hvo pianoroll scores
    evaluator_test_set.get_ground_truth_hvos_array()

    # 2.3.1 - get ground truth monotonic grooves
    import numpy as np
    input = np.array(
    [hvo_seq.flatten_voices(voice_idx=2) for hvo_seq in evaluator_test_set.get_ground_truth_hvo_sequences()])

    # 2.3.2 Pass the ground truth data to the model
    from model import load_groove_transformer_encoder_model
    import torch
    import numpy as np

    model_name = "colorful_sweep_41" # robust_sweep_29
    model_path = f"model/saved/monotonic_groove_transformer_v1/latest/{model_name}.pth"
    GrooveTransformer = load_groove_transformer_encoder_model(model_path)
    predictions = GrooveTransformer.predict(torch.tensor(evaluator_test_set.get_ground_truth_hvos_array(), dtype=torch.float32))
    predictions = torch.cat(predictions, -1)

    # 2.3.3 - Add the predictions to the evaluator
    evaluator_test_set.add_predictions(predictions.detach().numpy())

    # 2.4 -      Save Evaluator
    evaluator_test_set.dump(path="demos/GrooveEvaluator/examples", fname=f"{model_name}.Eval.bz2")

    # 2.4 -      Load Evaluator using full path with extension
    from eval.GrooveEvaluator import load_evaluator
    evaluator_test_set = load_evaluator(f"demos/GrooveEvaluator/examples/test_set_full_{model_name}.Eval.bz2")




