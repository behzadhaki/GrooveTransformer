import os, sys
import json
import pickle
import bz2
from tqdm import tqdm

import numpy as np
from bokeh.plotting import figure, gridplot
from bokeh.io import save
from bokeh.models import Tabs, #Panel
from hvo_sequence.io_helpers import note_sequence_to_hvo_sequence
from hvo_sequence.drum_mappings import get_drum_mapping_using_label

from math import pi

import pandas as pd

from bokeh.palettes import Category20c
from bokeh.plotting import figure, show
from bokeh.transform import cumsum

def does_pass_filter(hvo_sample, filter_dict):   # FIXME THERE IS AN ISSUE HERE
    passed_conditions = [True]  # initialized with true in case no filters are required
    for fkey_, fval_ in filter_dict.items():
        if fval_ is not None:
            passed_conditions.append(True if hvo_sample.metadata[fkey_] in fval_ else False)

    return all(passed_conditions)


def get_data_directory_using_filters(dataset_tag, dataset_setting_json_path):
    """
    returns the directory path from which the data corresponding to the
    specified data/dataset_json_settings/4_4_Beats_gmd.json file is located or should be stored

    :param dataset_tag: [str] (use "gmd" for groove midi dataset - must match the key in json file located in the data/dataset_json_settings)
    :param dataset_setting_json_path: [file.json path] (path to data/dataset_json_settings/4_4_Beats_gmd.json or similar filter jsons)
    :return: path to save/load from the train.pickle/test.pickle/validation.pickle hvo_sequence subsets
    """
    main_path = f"data/{dataset_tag}/resources/cached/"
    last_directory = ""
    filter_dict = json.load(open(dataset_setting_json_path, "r"))["settings"][dataset_tag]
    print("filter_dict: ", filter_dict)
    global_dict = json.load(open(dataset_setting_json_path, "r"))["global"]
    print("global_dict: ", global_dict)

    for key_, val_ in global_dict.items():
        main_path += f"{key_}_{val_}/"

    for key_, val_ in filter_dict.items():
        if val_ is not None:
            print(key_, " ", val_)
            last_directory += f"{key_}_{val_}_"

    return main_path + last_directory[:-1]  # remove last underline


def pickle_hvo_dict(hvo_dict, dataset_tag, dataset_setting_json_path):
    """ pickles a dictionary of train/test/validation hvo_sequences (see below for dict structure)

    :param hvo_dict: [dict] dict of format {"train": [hvo_seqs], "test": [hvo_seqs], "validation": [hvo_seqs]}
    :param dataset_tag: [str] (use "gmd" for groove midi dataset)
    :param dataset_setting_json_path: [file.json path] (path to data/dataset_json_settings/4_4_Beats_gmd.json or similar filter jsons)
    """


    # create directories if needed
    dir__ = get_data_directory_using_filters(dataset_tag, dataset_setting_json_path)
    if not os.path.exists(dir__):
        os.makedirs(dir__)

    # collect and dump samples that match filter
    filter_dict_ = json.load(open(dataset_setting_json_path, "r"))[dataset_tag]
    for set_key_, set_data_ in hvo_dict.items():
        filtered_samples = []
        num_samples = len(set_data_)
        for sample in tqdm(set_data_, total = num_samples, desc=f"filtering HVO_Sequences in subset {set_key_}"):
            if does_pass_filter(sample, filter_dict_):
                filtered_samples.append(sample)

        ofile = bz2.BZ2File(os.path.join(dir__, f"{set_key_}.bz2pickle"), 'wb')
        pickle.dump(filtered_samples, ofile)
        ofile.close()



def load_original_gmd_dataset_pickle(gmd_pickle_path):
    ifile = bz2.BZ2File(open(gmd_pickle_path, "rb"), 'rb')
    gmd_dict = pickle.load(ifile)
    ifile.close()
    return gmd_dict


def extract_hvo_sequences_dict(gmd_dict, beat_division_factor, drum_mapping):
    """ extracts hvo_sequences from the original gmd_dict

    :param gmd_dict: [dict] dict of format {"train": [note_sequences], "test": [note_sequences], "validation": [note_sequences]}
    :param gmd_dict: [dict] dict of format {"train": [note_sequences], "test": [note_sequences], "validation": [note_sequences]}
    :param beat_division_factor: list of ints (e.g. [4] for 16th note resolution)
    :param drum_mapping: [dict] (e.g. get_drum_mapping_using_label("gmd"))
    :return: [dict] dict of format {"train": [hvo_sequences], "test": [hvo_sequences], "validation": [hvo_sequences]}
    """
    gmd_hvo_seq_dict = dict()

    for set in gmd_dict.keys():             # train, test, validation
        hvo_seqs = []
        n_samples = len(gmd_dict[set]["note_sequence"])
        for ix in tqdm(range(n_samples), desc=f"converting to hvo_sequence --> {set} subset"):
            # convert to note_sequence
            note_sequence = gmd_dict[set]["note_sequence"][ix]
            _hvo_seq = note_sequence_to_hvo_sequence(
                ns=note_sequence,
                drum_mapping=drum_mapping,
                beat_division_factors=beat_division_factor
            )
            if len(_hvo_seq.time_signatures) == 1 and len(_hvo_seq.tempos) == 1:
                # get metadata
                for key_ in gmd_dict[set].keys():
                    if key_ not in ["midi", "note_sequence", "hvo_sequence"]:
                        _hvo_seq.metadata.update({key_: str(gmd_dict[set][key_][ix])})

                hvo_seqs.append(_hvo_seq)

        # add hvo_sequences to dictionary
        gmd_hvo_seq_dict.update({set: hvo_seqs})

    return gmd_hvo_seq_dict


def group_by_master_id(hvo_sequence_list, sort_each_group_by_loop_id=True, save_midis_at=None):
    """ groups hvo_sequences by master_id

    :param hvo_sequence_list: [list] list of hvo_sequences
    :param sort_each_group_by_loop_id: [bool] if True, sorts each group by loop_id such
            that samples are sequentially ordered
    :return: [dict] dict of lists of hvo_sequences grouped by master_id
    """
    # separate by master_id
    grouped_by_master_id = {}
    for hvo_sample in tqdm(hvo_sequence_list, desc="grouping by master_id"):
        master_id = hvo_sample.metadata["master_id"]
        if master_id not in grouped_by_master_id:
            grouped_by_master_id[master_id] = []
        loop_id = hvo_sample.metadata["loop_id"]
        grouped_by_master_id[master_id].append((loop_id, hvo_sample))

    # sort each group by loop_id
    if sort_each_group_by_loop_id:
        for key, item in grouped_by_master_id.items():
            grouped_by_master_id[key] = [y for _, y in sorted(item)]

    # save midis
    if save_midis_at is not None:
        for key, item in tqdm(grouped_by_master_id.items()):
            performance_path = os.path.join(save_midis_at, key)
            os.makedirs(performance_path, exist_ok=True)
            for ix, hvo_sample in tqdm(enumerate(item)):
                hvo_sample.save_hvo_to_midi(filename=os.path.join(performance_path, f"{ix}.mid"))

    return grouped_by_master_id

def get_bokeh_histogram(hist_dict, file_name=None, hist_width=0.3, x_axis_label=None, y_axis_label=None,
                        font_size=12):
    hist = np.array(list(hist_dict.values()))
    xtick_locs = np.arange(0.5, len(hist), 1)
    title = f"histogram of {y_axis_label} per {x_axis_label}".capitalize() if x_axis_label is not None and y_axis_label is not None else ""
    p = figure(title=title, toolbar_location="below")
    colors = Category20c[20]
    colors = [colors[int(i.split("drummer")[-1])] for i in hist_dict.keys()]
    p.quad(top=hist, bottom=0, left=xtick_locs-hist_width, right=xtick_locs+hist_width,
           fill_color=colors, line_color="white")
    p.xaxis.ticker = xtick_locs
    p.xaxis.major_label_overrides = {xtick_locs[i]: list(hist_dict.keys())[i] for i in range(len(xtick_locs))}
    p.xaxis.major_label_orientation = 45

    p.title.text_font_size = f"{font_size}pt"
    p.axis.axis_label_text_font_size = f"{int(font_size)}pt"

    if x_axis_label is not None:
        p.xaxis.axis_label = x_axis_label.capitalize()
    if y_axis_label is not None:
        p.yaxis.axis_label = y_axis_label.capitalize()

    p.xaxis.major_label_text_font_size = f"{int(font_size)}pt"
    p.yaxis.major_label_text_font_size = f"{int(font_size)}pt"

    if file_name is not None:
        if not file_name.endswith(".html"):
            file_name += ".html"
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        save(p, filename=file_name)
        print(f"saved at {file_name}")

    return p

def get_per_performer_bokeh_histogram(hvo_seq_list, dataset_label, ncols=3, filename=None):
    titles = []
    figures_to_plot = []

    # compile data into sequential list of all samples and attributes
    genres = []
    performers = []
    session_identifiers = []
    loop_ids = []

    for hvo_sample in hvo_seq_list:
        genres.append(hvo_sample.metadata["style_primary"])
        performer = hvo_sample.metadata["master_id"].split("/")[0]
        if int(performer.split("drummer")[-1]) < 10:
            performer = "drummer 0" + performer.split("drummer")[-1]
        else:
            performer = "drummer " + performer.split("drummer")[-1]
        performers.append(performer)
        session_identifiers.append("_".join(hvo_sample.metadata["master_id"].split("/")[1:]))
        loop_ids.append(hvo_sample.metadata["loop_id"])

    # place data in a dataframe
    import pandas as pd
    df = pd.DataFrame({
        "genre": genres,
        "performer": performers,
        "session_identifier": session_identifiers,
        "loop_id": loop_ids
    })

    print(f"there are {len(df.index)} number of samples in the dataset")
    # 16195 in train set

    # lets study the number of loops per performer =====================================================================
    df = pd.DataFrame({
        "performer": performers,
        "session_identifier": session_identifiers,
    })

    temp = df.groupby("performer").count()
    assert temp.values.flatten().sum() == len(df.index), "something went wrong! total samples do not match"
    loops_per_performer = {k: v for k, v in zip(temp.index.values, temp.values.flatten())}
    figures_to_plot.append(get_bokeh_histogram(loops_per_performer, y_axis_label="Loop Count", font_size=12))
    titles.append("Loop Count")

    # lets study the number of unique genres performed by each performer ===============================================
    df = pd.DataFrame({
        "performer": performers,
        "genre": genres,
    })
    temp = df.groupby("performer").nunique()
    genres_per_performer = {k: v for k, v in zip(temp.index.values, temp.values.flatten())}
    figures_to_plot.append(get_bokeh_histogram(genres_per_performer, y_axis_label="Genre Count", font_size=12))
    titles.append("Genre Count")

    # lets study the number of unique long performances by each performer ==============================================
    df = pd.DataFrame({
        "performer": performers,
        "session_identifier": session_identifiers
    })
    temp = df.groupby("performer").nunique()
    long_performances_per_performer = {k: v for k, v in zip(temp.index.values, temp.values.flatten())}
    figures_to_plot.append(
        get_bokeh_histogram(long_performances_per_performer, y_axis_label="Performance Count", font_size=12))
    titles.append("Performance Count")
    # =================================================================================================================

    tabify = False

    for fig in figures_to_plot:
        fig.x_range = figures_to_plot[0].x_range
        fig.title = dataset_label

    if tabify:
        p = Tabs(tabs=[Panel(child=t, title=title) for t, title in zip(figures_to_plot, titles)])
    else:
        p = gridplot(figures_to_plot, ncols=ncols, plot_width=800, plot_height=400)

    if filename is not None:
        if not filename.endswith(".html"):
            filename += ".html"

        filename = filename.replace(".html", f"_{dataset_label}.html")

        os.makedirs(os.path.dirname(filename), exist_ok=True)
        save(p, filename=filename)
        print(f"saved at {filename}")

    return p

def create_pi_chart(x_dict, use_percentage=True, title="", filename=None):
    if use_percentage:
        x_dict = {k: v / sum(x_dict.values()) * 100 for k, v in x_dict.items()}

    data = pd.Series(x_dict).reset_index(name='value').rename(columns={'index': 'index'})
    data['angle'] = data['value'] / data['value'].sum() * 2 * pi
    colors =  Category20c[20]
    data['color'] = [colors[int(i.split("drummer")[-1])] for i in x_dict.keys()]

    p = figure(title=title,
               tools="hover", tooltips="@index: @value", x_range=(-0.5, 1.0))

    p.wedge(x=0, y=1, radius=0.3,
            start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'),
            line_color="white", fill_color='color', legend_field='index', source=data)

    p.axis.axis_label = None
    p.axis.visible = False
    p.grid.grid_line_color = None

    if filename is not None:
        if not filename.endswith(".html"):
            filename += ".html"

        os.makedirs(os.path.dirname(filename), exist_ok=True)
        save(p, filename=filename)
        print(f"saved at {filename}")

    return p

def get_per_performer_bokeh_pi_chart(hvo_seq_list, dataset_label, ncols=3, filename=None):
    titles = []
    figures_to_plot = []

    # compile data into sequential list of all samples and attributes
    genres = []
    performers = []
    session_identifiers = []
    loop_ids = []

    for hvo_sample in hvo_seq_list:
        genres.append(hvo_sample.metadata["style_primary"])
        performer = hvo_sample.metadata["master_id"].split("/")[0]
        if int(performer.split("drummer")[-1]) < 10:
            performer = "drummer0" + performer.split("drummer")[-1]
        else:
            performer = "drummer" + performer.split("drummer")[-1]

        performers.append(performer)
        session_identifiers.append("_".join(hvo_sample.metadata["master_id"].split("/")[1:]))
        loop_ids.append(hvo_sample.metadata["loop_id"])

    # place data in a dataframe
    import pandas as pd
    df = pd.DataFrame({
        "genre": genres,
        "performer": performers,
        "session_identifier": session_identifiers,
        "loop_id": loop_ids
    })

    print(f"there are {len(df.index)} number of samples in the dataset")
    # 16195 in train set
    total_loops = len(df.index)

    # lets study the number of loops per performer =====================================================================
    df = pd.DataFrame({
        "performer": performers,
        "session_identifier": session_identifiers,
    })

    temp = df.groupby("performer").count()
    print(temp)
    loops_per_performer = {k: v for k, v in zip(temp.index.values, temp.values.flatten())}
    figures_to_plot.append(create_pi_chart(loops_per_performer, title=f"Loop Count - (Total Number of Loops = {total_loops})"))
    titles.append("Loop Count")

    # lets study the number of unique genres performed by each performer ===============================================
    df = pd.DataFrame({
        "performer": performers,
        "genre": genres,
    })
    temp = df.groupby("performer").nunique()
    genres_per_performer = {k: v for k, v in zip(temp.index.values, temp.values.flatten())}
    figures_to_plot.append(create_pi_chart(genres_per_performer, title=f"Genre Count - ({dataset_label}) "))
    titles.append("Genre Count")

    # lets study the number of unique long performances by each performer ==============================================
    df = pd.DataFrame({
        "performer": performers,
        "session_identifier": session_identifiers
    })
    temp = df.groupby("performer").nunique()
    long_performances_per_performer = {k: v for k, v in zip(temp.index.values, temp.values.flatten())}
    figures_to_plot.append(create_pi_chart(long_performances_per_performer, title=f"Performance Count - ({dataset_label})" ))
    titles.append("Performance Count")

    p = gridplot(figures_to_plot, ncols=ncols, plot_width=800, plot_height=400)

    if filename is not None:
        if not filename.endswith(".html"):
            filename += ".html"

        filename = filename.replace(".html", f"_{dataset_label}.html")

        os.makedirs(os.path.dirname(filename), exist_ok=True)
        save(p, filename=filename)
        print(f"saved at {filename}")

    return p


def get_genre_performer_heatmaps(data_set, subset_tag, data_identifier, filename=None):
    genres = []
    performers = []
    session_identifiers = []
    loop_ids = []

    for hvo_sample in data_set:
        genres.append(hvo_sample.metadata["style_primary"])
        performer = hvo_sample.metadata["master_id"].split("/")[0]
        if int(performer.split("drummer")[-1]) < 10:
            performer = "drummer 0" + performer.split("drummer")[-1]
        else:
            performer = "drummer " + performer.split("drummer")[-1]
        performers.append(performer)
        session_identifiers.append("_".join(hvo_sample.metadata["master_id"].split("/")[1:]))
        loop_ids.append(hvo_sample.metadata["loop_id"])

    # place data in a dataframe
    import pandas as pd

    df = pd.DataFrame({
        "genre": genres,
        "performer": performers,
        "session_identifier": session_identifiers,
        "loop_id": loop_ids
    })

    summarized_heatmaps = df.groupby(["genre", "performer"]).nunique()
    # get first row

    # histogrmas of performance, genre in terms of total performances
    session_identifier_tuples = list()
    loop_id_tuples = list()
    for index in summarized_heatmaps.index:
        session_identifier_tuples.append(
            (index[0], index[1], summarized_heatmaps.loc[index[0], index[1]]['session_identifier']))
        loop_id_tuples.append((index[0], index[1], summarized_heatmaps.loc[index[0], index[1]]['loop_id']))

    session_identifier_tuples = sorted(session_identifier_tuples, key=lambda x: x[0], reverse=True)
    loop_id_tuples = sorted(loop_id_tuples, key=lambda x: x[0], reverse=True)

    import holoviews as hv
    from holoviews import opts
    hv.extension('bokeh')
    from bokeh.io import save

    hm_sessions = hv.HeatMap(session_identifier_tuples)
    hm_sessions = hm_sessions * hv.Labels(hm_sessions).opts(padding=0, text_color='white')
    fig_sessions = hv.render(hm_sessions, backend='bokeh')
    fig_sessions.plot_height = 400
    fig_sessions.plot_width = 800
    fig_sessions.title = f"Number of unique sessions per genre and performer ({subset_tag.capitalize()} - {data_identifier})"
    fig_sessions.xaxis.axis_label = ""
    fig_sessions.yaxis.axis_label = ""
    fig_sessions.xaxis.major_label_orientation = 45
    fig_sessions.xaxis.major_label_text_font_size = "12pt"
    fig_sessions.yaxis.major_label_text_font_size = "12pt"

    hm_loops = hv.HeatMap(loop_id_tuples)
    hm_loops = hm_loops * hv.Labels(hm_loops).opts(padding=0, text_color='white')
    fig_loops = hv.render(hm_loops, backend='bokeh')
    fig_loops.plot_height = 400
    fig_loops.plot_width = 800
    fig_loops.title = f"Number of unique loops per genre and performer ({subset_tag.capitalize()} - {data_identifier})"
    fig_loops.xaxis.axis_label = ""
    fig_loops.yaxis.axis_label = ""
    fig_loops.xaxis.major_label_orientation = 45
    fig_loops.xaxis.major_label_text_font_size = "12pt"
    fig_loops.yaxis.major_label_text_font_size = "12pt"

    p = gridplot([fig_sessions, fig_loops], ncols=2, plot_width=800, plot_height=400)

    if filename is not None:
        if not filename.endswith(".html"):
            filename += ".html"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        filename = filename.replace(".html", f"_{subset_tag.capitalize()} - {data_identifier}.html")
        save(p, filename)

    return p


if __name__ == "__main__":

    gmd_pickle_path = "data/gmd/resources/storedDicts/groove_2bar-midionly.bz2pickle"
    dataset_tag = "gmd"
    dataset_setting_json_path = "data/dataset_json_settings/4_4_Beats_gmd.json"
    beat_division_factor = [4]
    drum_mapping_label = "ROLAND_REDUCED_MAPPING"
    subset_tag = "train"

    gmd_dict = load_original_gmd_dataset_pickle(gmd_pickle_path)
    hvo_dict = extract_hvo_sequences_dict (gmd_dict, [4], get_drum_mapping_using_label("ROLAND_REDUCED_MAPPING"))
    pickle_hvo_dict(hvo_dict, dataset_tag, dataset_setting_json_path)


