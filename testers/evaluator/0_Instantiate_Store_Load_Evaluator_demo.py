from data.dataLoaders import load_gmd_hvo_sequences

if __name__ == "__main__":
    # 1 - Load test set dataset
    dataset_setting_json_path = "dataset_setting.json"

    test_set = load_gmd_hvo_sequences(
        "data/gmd/resources/storedDicts/groove_2bar-midionly.bz2pickle",
        "gmd", dataset_setting_json_path, "test")

    # 2 - Create the Subsetter filters to divide up the dataset into subsets
    list_of_filter_dicts_for_subsets = []
    styles = [
        "afrobeat", "afrocuban", "blues", "country", "dance", "funk", "gospel", "highlife", "hiphop", "jazz",
        "latin", "middleeastern", "neworleans", "pop", "punk", "reggae", "rock", "soul"]
    for style in styles:
        list_of_filter_dicts_for_subsets.append(
            {"style_primary": [style], "beat_type": ["beat"], "time_signature": ["4-4"]}
        )

    # 2 - Instantiate the evaluator
    from eval.GrooveEvaluator.src.evaluator import Evaluator
    evaluator_test_set = Evaluator(
        test_set,
        list_of_filter_dicts_for_subsets,
        _identifier="test_set_full",
        n_samples_to_use= len(test_set),
        max_hvo_shape=(32, 27),
        analyze_heatmap=False,
        analyze_global_features=False,
        analyze_audio=False,
        analyze_piano_roll=True,
        n_samples_to_synthesize="all",
        n_samples_to_draw_pianorolls=10,    # "all",
        disable_tqdm=False
    )

    # 3 - get ground truth data to pass to the model
    input = evaluator_test_set.get_ground_truth_hvos_array()

    # 4 - Pass the ground truth data to the model
    # predicted_hvos_array = model.predict(input)

    # 5 - Add the predictions to the evaluator
    evaluator_test_set.add_predictions(input)

    # 6 - Dump the evaluator
    evaluator_test_set.dump(path="misc", fname="evaluator_test_set_full")

    dict_ = evaluator_test_set.get_logging_dict()
    _gt_logging_data, _predicted_logging_data = dict_

    # 6 - Dump the evaluator
    from bokeh.plotting import show

    show(_gt_logging_data["global_feature_pdfs"])
    show(_gt_logging_data['piano_rolls'])

