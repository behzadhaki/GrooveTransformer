import numpy as np
import pandas as pd
from scipy import stats, integrate
import os
from tqdm import tqdm
from bokeh.plotting import figure
from bokeh.layouts import gridplot
from bokeh.models import Panel, Tabs, HoverTool, Legend
from bokeh.palettes import Magma


def flatten(t):
    return [item for sublist in t for item in sublist]


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


def get_absolute_measures_for_single_set(flat_feature_dict, csv_file=None):
    # Gets absolute measures of a set according to
    # Yang, Li-Chia, and Alexander Lerch. "On the evaluation of generative models in music."
    #           Neural Computing and Applications 32.9 (2020): 4773-4784.

    stats = []  # list of lists stats[i] corresponds to [mean, std, min, max, median, q1, q3]
    labels = []

    for key in flat_feature_dict.keys():
        data = flat_feature_dict[key]
        # Calc stats
        stats.append(
            [np.mean(data), np.std(data), np.min(data), np.max(data), np.percentile(data, 50), np.percentile(data, 25),
             np.percentile(data, 75)])
        labels.append(key)

    df2 = pd.DataFrame(np.round(np.array(stats),3).transpose(),
                       ["mean", "std", "min", "max", "median", "q1", "q3"],
                       labels).transpose()

    if csv_file is not None:
        df2.to_csv(csv_file)

    return df2


def get_intraset_distances_from_array(features_array):
    # Calculates l2 norm distance of each sample with every other sample
    intraset_distances = []
    features_array = features_array[np.logical_not(np.isnan(features_array))]
    ix = np.arange(features_array.size)
    for current_i, current_feature in enumerate(features_array):
        distance_to_all = np.abs(features_array[np.delete(ix, current_i)] - current_feature)
        intraset_distances.extend(distance_to_all)
    return np.array(intraset_distances)


def get_intraset_distances_from_set(flat_feature_dict):

    intraset_distances_feat_dict = {}

    for key, flat_feat_array in flat_feature_dict.items():
        intraset_distances_feat_dict[key] = get_intraset_distances_from_array(flat_feat_array)

    return intraset_distances_feat_dict


def get_interset_distances(flat_feature_dict_a, flat_feature_dict_b):

    interset_distances_feat_dict = {}

    for key, flat_feat_array in flat_feature_dict_a.items():
        flat_feat_array = flat_feat_array[np.logical_not(np.isnan(flat_feat_array))]
        flat_feat_array_b = flat_feature_dict_b[key][np.logical_not(np.isnan(flat_feature_dict_b[key]))]

        interset_distances = []
        for current_i, current_feature_in_a in enumerate(flat_feat_array):
            distance_to_all = np.abs(flat_feat_array_b - current_feature_in_a)
            interset_distances.extend(distance_to_all)

        interset_distances_feat_dict[key] = interset_distances

    return interset_distances_feat_dict


def kl_dist(A, B, pdf_A=None, pdf_B=None, num_sample=100):
    # Calculate KL distance between the two PDF

    # calc pdfs if necessary - helps to avoid redundant calculations for pdfs if already done
    pdf_A = stats.gaussian_kde(A) if pdf_A is None else pdf_A
    pdf_B = stats.gaussian_kde(B) if pdf_B is None else pdf_B

    sample_A = np.linspace(np.min(A), np.max(A), num_sample)
    sample_B = np.linspace(np.min(B), np.max(B), num_sample)

    return stats.entropy(pdf_A(sample_A), pdf_B(sample_B))


def overlap_area(A, B, pdf_A, pdf_B, max_sample_size=100):
    # Calculate overlap between the two PDF

    # calc pdfs if necessary - helps to avoid redundant calculations for pdfs if already done
    return integrate.quad(lambda x: min(pdf_A(x), pdf_B(x)), np.min((np.min(A), np.min(B))), np.max((np.max(A), np.max(B))))[0]


def convert_multi_feature_distances_to_pdf(distances_features_dict):
    pdf_dict = {}
    for feature_key, distances_for_feature in distances_features_dict.items():
        try:
            pdf_dict[feature_key] = stats.gaussian_kde(distances_for_feature)
        except:
            print(f"SINGULAR MATRIX Error calculating pdf for feature {feature_key}")
            print(f"distribution for key {feature_key} is {distances_for_feature}")
            print(f" pdf_dict so far are {pdf_dict}")
    return pdf_dict


def get_KL_OA_for_multi_feature_distances(distances_dict_A, distances_dict_B,
                                          pdf_distances_dict_A, pdf_distances_dict_B,
                                          num_sample=1000):
    KL_dict = {}
    OA_dict = {}

    for feature_key in distances_dict_A.keys():
        KL_dict[feature_key] = kl_dist(
            distances_dict_A[feature_key], distances_dict_B[feature_key],
            pdf_A=pdf_distances_dict_A[feature_key], pdf_B=pdf_distances_dict_B[feature_key],
            num_sample=num_sample)
        print(f"KL_{feature_key}")
        OA_dict[feature_key] = overlap_area(
            distances_dict_A[feature_key], distances_dict_B[feature_key],
            pdf_A=pdf_distances_dict_A[feature_key], pdf_B=pdf_distances_dict_B[feature_key])
        print(f"OA_{feature_key}")

    return KL_dict, OA_dict


def compare_two_sets_against_ground_truth(gt, set1, set2, set_labels=['gt', 'set1', 'set2'], csv_path=None,
                                          calc_OA_downsample_size = 100):
    # generates a table similar to that of No.4 in Yang et. al.
    if csv_path is not None:
        if not os.path.exists(os.path.dirname(csv_path)):
            os.makedirs(os.path.dirname(csv_path))

    gt_intra = get_intraset_distances_from_set(gt)
    set1_intra = get_intraset_distances_from_set(set1)
    set1_inter_gt = get_interset_distances(set1, gt)
    pdf_gt_intra = convert_multi_feature_distances_to_pdf(gt_intra)
    pdf_set1_inter_gt = convert_multi_feature_distances_to_pdf(set1_inter_gt)
    KL_set1inter_gt_intra, OA_set1inter_gt_intra = get_KL_OA_for_multi_feature_distances(
        set1_inter_gt, gt_intra,
        pdf_set1_inter_gt, pdf_gt_intra, num_sample=100)

    set2_inter_gt, pdf_set2_inter_gt, set2_intra = None, None, None
    if set2 is not None:
        set2_intra = get_intraset_distances_from_set(set2)
        set2_inter_gt = get_interset_distances(set2, gt)
        pdf_set2_inter_gt = convert_multi_feature_distances_to_pdf(set2_inter_gt)
        KL_set2inter_gt_intra, OA_set2inter_gt_intra = get_KL_OA_for_multi_feature_distances(
            set2_inter_gt, gt_intra,
            pdf_set2_inter_gt, pdf_gt_intra, num_sample=100)

    features = gt_intra.keys()

    data_for_feature = []
    for feature in features:
        try:
            data_row = []
            # calculate mean and std of gt_intra
            data_row.extend([np.round(np.mean(gt_intra[feature]), 3), np.round(np.std(gt_intra[feature]), 3)])
            data_row.extend([np.round(np.mean(set1_intra[feature]), 3), np.round(np.std(set1_intra[feature]), 3)])
            data_row.extend([np.round(KL_set1inter_gt_intra[feature], 3), np.round(OA_set1inter_gt_intra[feature], 3)])
            if set2 is not None:
                data_row.extend([np.round(np.mean(set2_intra[feature]), 3), np.round(np.std(set2_intra[feature]), 3)])
                data_row.extend([np.round(KL_set2inter_gt_intra[feature], 3), np.round(OA_set2inter_gt_intra[feature], 3)])

            data_for_feature.append(data_row)
        except:
            print(f"Can't calculate KL or OA for feature {feature}")

    header = pd.MultiIndex.from_arrays([
        np.array(
            [set_labels[0], set_labels[0], set_labels[1], set_labels[1], set_labels[1], set_labels[1],
             set_labels[2], set_labels[2], set_labels[2], set_labels[2]]
        ),
        np.array(
            ["Intra-set", "Intra-set", "Intra-set", "Intra-set", "Inter-set", "Inter-set", "Intra-set", "Intra-set",
             "Inter-set", "Inter-set"]
        ),
        np.array(
            ["mean", "STD", "mean", "STD", "KL", "OA", "mean", "STD", "KL", "OA"]
        ),
    ]) if set2 is not None else pd.MultiIndex.from_arrays([
        np.array(
            [set_labels[0], set_labels[0], set_labels[1], set_labels[1], set_labels[1], set_labels[1]]
        ),
        np.array(
            ["Intra-set", "Intra-set", "Intra-set", "Intra-set", "Inter-set", "Inter-set"]
        ),
        np.array(
            ["mean", "STD", "mean", "STD", "KL", "OA"]
        ),
    ])


    index = [x.split("::")[-1] for x in features]
    df = pd.DataFrame(data_for_feature,
                      index=index,
                      columns=header)

    if csv_path is not None:
        df.to_csv(csv_path)


    raw_data = (gt_intra, set1_intra, set2_intra, set1_inter_gt, set2_inter_gt, pdf_gt_intra, pdf_set1_inter_gt, pdf_set2_inter_gt)
    return df, raw_data


def plot_inter_intra_distance_distributions(raw_data, set_labels, ncols=3, figsize=(400,300), legend_fs="6pt"):
    gt_intra, set1_intra, set2_intra, set1_inter_gt, set2_inter_gt, pdf_gt_intra, pdf_set1_inter_gt, pdf_set2_inter_gt = raw_data
    pdf_set1_intra = convert_multi_feature_distances_to_pdf(set1_intra)
    pdf_set2_intra = convert_multi_feature_distances_to_pdf(set2_intra) if set2_intra is not None else None

    associated_tab_labels = [x.split("::")[0] for x in list(gt_intra.keys())]
    associated_tab_labels = list(set(associated_tab_labels))
    bokeh_tabs_dict = {x: [] for x in associated_tab_labels}

    num_sample = 1000


    for i, key in tqdm(enumerate(gt_intra.keys())):
        if set2_intra is not None:
            if key not in pdf_set2_intra.keys() or key not in pdf_set2_inter_gt.keys():
                continue
        if key not in pdf_set1_intra.keys():
            continue

        p = figure(width=figsize[0], height=figsize[1], title = key.split("::")[-1])
        associated_tab_labels.append(key.split("::")[0])

        x = np.linspace(np.min(gt_intra[key]), np.max(gt_intra[key]), num_sample)
        p.line(x, pdf_gt_intra[key](x), line_width=2, color="grey", legend_label=f"Intra ({set_labels[0]})",
                   line_dash="dashed")


        x = np.linspace(np.min(set1_intra[key]), np.max(set1_intra[key]), num_sample)
        y1 = pdf_set1_intra[key](x)
        p.line(x, y1, line_width=2, color="blue", legend_label=f"Intra ({set_labels[1]})", line_dash="dashed")

        x = np.linspace(np.min(set1_inter_gt[key]), np.max(set1_inter_gt[key]), num_sample)
        y2 = pdf_set1_inter_gt[key](x)
        p.line(x, y2, line_width=2, color="blue", legend_label=f"Inter ({set_labels[1]} vs {set_labels[0]})")

        if set2_intra is not None:
            x = np.linspace(np.min(set2_intra[key]), np.max(set2_intra[key]), num_sample)
            p.line(x, pdf_set2_intra[key](x), line_width=2, color="red", legend_label=f"Intra ({set_labels[2]})", line_dash="dashed")
            x = np.linspace(np.min(set2_inter_gt[key]), np.max(set2_inter_gt[key]), num_sample)
            p.line(x, pdf_set2_inter_gt[key](x), line_width=2, color="red", legend_label=f"Inter ({set_labels[2]} vs {set_labels[0]})")

        p.legend.click_policy = "hide"
        p.legend.label_text_font_size = legend_fs
        p.xaxis.axis_label = 'Euclidean Distance'
        p.yaxis.axis_label = 'Density'
        p.add_layout(p.legend[0], 'right')
        bokeh_tabs_dict[key.split("::")[0]].append(p)

    tabs = []
    for tab_label, plots in bokeh_tabs_dict.items():
        tab = Panel(child=gridplot(plots, ncols=ncols), title=tab_label)
        tabs.append(tab)

    return Tabs(tabs=tabs)


def get_KL_OA_plot(df, set_labels, figsize=(1200, 1000)):

    title = f"Intra ( {set_labels[1]} ⟁ )  to Inter {set_labels[0]}" if len(set_labels) <= 2 else f"Intra ( {set_labels[1]} ⟁ OR {set_labels[2]} ⃞ ) to Inter {set_labels[0]}"

    p = figure(width=figsize[0], height=figsize[1], title = title)

    print(f"len(df.index) = {len(df.index)}")
    palette = Magma[256]

    # divide up palette into df.index.size number of colors
    colors = [palette[int(x)] for x in np.linspace(0, 255, len(df.index))]

    legend_it = []

    for i, index in tqdm(enumerate(df.index)):

        handlers = []

        x1 = df[(set_labels[1], 'Inter-set', 'KL')][index]
        y1 = df[(set_labels[1], 'Inter-set', 'OA')][index]

        handlers.append(p.triangle(x=x1, y=y1, color=colors[i], name=f"{index} ({set_labels[1]})"))

        if len(set_labels) > 2:
            x2 = df[(set_labels[2], 'Inter-set', 'KL')][index]
            y2 = df[(set_labels[2], 'Inter-set', 'OA')][index]
            handlers.append(p.square(x=x2, y=y2, color=colors[i], name=f"{index} ({set_labels[2]})"))
            handlers.append(p.line(x=[x1, x2], y=[y1, y2], line_width=2, line_color=colors[i], name=f"{index} ({set_labels[1]} vs {set_labels[2]})"))

        title = f"{index} ({set_labels[1]})" if len(set_labels) <= 2 else f"{index} ({set_labels[1]} vs {set_labels[2]})"
        legend_it.append((title, handlers))

    legend = Legend(items=legend_it)
    legend.click_policy = "hide"
    p.add_layout(legend, 'right')

    p.add_tools(HoverTool(tooltips=[
        ('Feature', '$name'),
    ]))

    return p
