import sys
sys.path.append("../../")
sys.path.append("../")
import wandb


from data.gmd.src import subsetters
# , Set_Sampler, convert_hvos_array_to_subsets
from eval.GrooveEvaluator.src.evaluator import HVOSeq_SubSet_Evaluator, Evaluator

from bokeh.io import output_file, show, save


pickle_source_path = "../preprocessed_dataset/datasets_extracted_locally/GrooveMidi/hvo_0.5.2/" \
                     "Processed_On_21_07_2021_at_14_32_hrs"

# Create Subset Filters
styles = ["afrobeat", "afrocuban", "blues", "country", "dance", "funk", "gospel", "highlife", "hiphop", "jazz",
          "latin", "middleeastern", "neworleans", "pop", "punk", "reggae", "rock", "soul"]

list_of_filter_dicts_for_subsets = []
for style in styles:
    list_of_filter_dicts_for_subsets.append(
        {"style_primary": [style], "beat_type": ["beat"], "time_signature": ["4-4"]}
    )

gmd_pickle_path = "data/gmd/resources/cached/beat_division_factor_[4]/drum_mapping_label_['ROLAND_REDUCED_MAPPING']/beat_type_['beat']_time_signature_['4-4']/train.bz2pickle"

train_set_evaluator = Evaluator(
    gmd_pickle_path,
    list_of_filter_dicts_for_subsets=list_of_filter_dicts_for_subsets,
    max_hvo_shape=(32, 27),
    n_samples_to_use=64,
    disable_tqdm=False
)

gt_hvos_array = train_set_evaluator.get_ground_truth_hvos_array()

# Pass to model
# predicted_hvos_array = model.predict(gt_hvos_array)

train_set_evaluator.add_predictions(gt_hvos_array)

###################
###################
# EXTRACTION HERE
'''for key in distances_dict.keys():
    summary = {"mean": np.mean(distances_dict[key]),
               "min": np.min(distances_dict[key]), "max": np.max(distances_dict[key]), 
               "median": np.percentile(distances_dict[key], 50),
               "q1": np.percentile(distances_dict[key], 25),
               "q3": np.percentile(distances_dict[key], 75)}

    distances_dict[key] = {self._identifier: summary}'''

import numpy as np
import pandas as pd

def get_stats_from_samples(feature_value_dict, csv_file=None):
    stats = []  # list of lists stats[i] corresponds to [mean, std, min, max, median, q1, q3]
    labels = []

    for key in feature_value_dict.keys():
        # Compile all genre data together
        data = []
        for key2 in gt_feature_value_dict[key].keys():
            data.extend(gt_feature_value_dict[key][key2])

        # Calc stats
        stats.append([np.mean(data), np.std(data), np.min(data), np.max(data), np.percentile(data, 50), np.percentile(data, 25), np.percentile(data, 75)])
        labels.append(key)

    df2 = pd.DataFrame(np.array(stats).transpose(),
                       ["mean", "std", "min", "max", "median", "q1", "q3"],
                       labels)

    if csv_file is not None:
        df2.to_csv(csv_file)

    return df2

#del get_stats_from_evaluator

def get_stats_from_evaluator(evaluator_, calc_gt=True, calc_pred=True, csv_file=None):
    gt_df = get_stats_from_samples(
        evaluator_.gt_SubSet_Evaluator.feature_extractor.get_global_features_dicts(True), None
    ) if (calc_gt and evaluator_.gt_SubSet_Evaluator is not None) else None

    pd_df = get_stats_from_samples(
            evaluator_.prediction_SubSet_Evaluator.feature_extractor.get_global_features_dicts(True), None
    ) if (calc_gt and evaluator_.prediction_SubSet_Evaluator is not None) else None

    keys = []
    if gt_df is not None:
        keys.extend(gt_df.columns)
    if pd_df is not None:
        keys.extend(pd_df.columns)

    print(keys)

    datas = []
    labels = []

    for key in keys:
        if gt_df is not None:
            data = gt_df.iloc[:][key].values if key in gt_df.columns else [None] * 7
        else:
            data = [None] * 7

        labels.append(key+"__Ground_Truth")
        datas.append(data)

        if pd_df is not None:
            data = pd_df.iloc[:][key].values() if key in pd_df.columns else [None] * 7
        else:
            data = [None] * 7

        datas.append(data)
        labels.append(key + "__Prediction")

    df2 = pd.DataFrame(np.array(datas).transpose(),
                       ["mean", "std", "min", "max", "median", "q1", "q3"],
                       labels
                       )

    if csv_file is not None:
        df2.to_csv(csv_file)

    return df2

gt_feature_value_dict = train_set_evaluator.gt_SubSet_Evaluator.feature_extractor.get_global_features_dicts(True)

"""df = get_stats_from_samples(gt_feature_value_dict, csv_file="misc/test_data.csv")
a = df.values.tolist()"""

train_set_evaluator.dump("misc/")
df_compiled = get_stats_from_evaluator(train_set_evaluator, csv_file="misc/test_data.csv")


