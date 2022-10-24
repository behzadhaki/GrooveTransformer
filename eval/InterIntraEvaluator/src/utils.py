import numpy as np

def get_pd_feats_from_evaluator(evaluator_):
    # extracts the prediction features from a evaluator
    return evaluator_.prediction_SubSet_Evaluator.feature_extractor.get_global_features_dicts(True)


def get_gt_feats_from_evaluator(evaluator_):
    # extracts the ground truth features from a evaluator
    return evaluator_.gt_SubSet_Evaluator.feature_extractor.get_global_features_dicts(True)

def flatten_subset_genres(feature_dict):
    # combines the subset samples irregardless of their genre
    flattened_feature_dict = {x: np.array([]) for x in feature_dict.keys()}
    for feature_key in flattened_feature_dict.keys():
        for subset_key, subset_samples in feature_dict[feature_key].items():
            flattened_feature_dict[feature_key] = np.append(flattened_feature_dict[feature_key], subset_samples)
    return flattened_feature_dict