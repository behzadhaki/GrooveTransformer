## PLOTTING
import os
from scipy.ndimage.filters import gaussian_filter

import colorcet as cc
from numpy import linspace
from scipy.stats.kde import gaussian_kde

import numpy as np
from bokeh.io import output_file, save
from bokeh.models import ColumnDataSource, Span, Legend, SingleIntervalTicker
from bokeh.plotting import figure

from bokeh.models.annotations import Title

try:
    import holoviews as hv
    from holoviews import opts
    hv.extension('bokeh')
    _HAS_HOLOVIEWS = True
except ImportError:
    print("Holoviews not installed, will not be able to generate violin plots")
    _HAS_HOLOVIEWS = False

from bokeh.models import Tabs, Panel

##############################################
###
#      Plotting methods for plotting velocity
#               profiles/Histograms
###
##############################################

def separate_figues_by_tabs(bokeh_fig_list, tab_titles=None, top_panel_identifier="::"):

    tab_titles = [str(tab_ix) for tab_ix in range(bokeh_fig_list)] if tab_titles is None else tab_titles

    print(" ------ TAB TITLES", tab_titles)

    top_tab_bottom_tabs_dicts = {"other": []}

    for ix, tab_title in enumerate(tab_titles):
        _p = bokeh_fig_list[ix]
        if len(tab_title.split("::")) > 1:
            top_key = tab_title.split("::")[0]
            title_ = tab_title.split("::")[1]
            if top_key not in top_tab_bottom_tabs_dicts.keys():
                top_tab_bottom_tabs_dicts.update({top_key: [Panel(child=_p, title=title_)]})
            else:
                top_tab_bottom_tabs_dicts[top_key].append(Panel(child=_p, title=title_))
        else:
            title_ = tab_title
            top_tab_bottom_tabs_dicts["other"].append(Panel(child=_p, title=title_))

    top_panels = []
    for major_key in top_tab_bottom_tabs_dicts.keys():
        top_panels.append(Panel(child=Tabs(tabs=top_tab_bottom_tabs_dicts[major_key]), title=major_key))

    return Tabs(tabs=top_panels)

def simplify_tags(tags, separator="_AND_"):
    tags_ = list()
    for tag in tags:
        tags_.append(tag.split(separator)[0])
    return tags_

def heat_map_plot(x, y, s, bins=[32*10, 127]):
    """
    Converts a set of x,y scatter locations to heat maps
    :param x:               x values for dots in scatter plot
    :param y:               corresponding y values
    :param s:               Smoothing factor
    :param bins:            [x axis number of bins, y axis number of bins]
    :return:
            heatmap         (a matrix of same size as specified in bins)
            extent          [starting x position, ending x position, starting y, ending y] for the created heat map

    """
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=bins)
    heatmap = gaussian_filter(heatmap, sigma=s)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    return heatmap.T, extent


def velocity_timing_heatmaps_scatter_plotter(
        heatmaps_dict,
        scatters_dict,
        number_of_loops_per_subset_dict,
        number_of_unique_performances_per_subset_dict=None,
        organized_by_drum_voice=True,               # denotes that the first key in heatmap and dict corresponds to drum voices
        title_prefix="",
        plot_width=1200, plot_height_per_set=100,legend_fnt_size="12px",
        synchronize_plots=True,
        downsample_heat_maps_by=1
):

    # Create a separate figure for first keys
    major_keys = sorted(list(heatmaps_dict.keys()))                    # (either drum voice or subset tag)
    minor_keys = sorted(list(heatmaps_dict[major_keys[0]].keys()))      # (either drum voice or subset tag)

    if organized_by_drum_voice is True:
        # Majors are drum voice and minors are subset tags
        major_titles = ["{} {}".format(title_prefix, str(major_key)) for major_key in major_keys]

        minor_tags = ["{}".format(
            str(minor_key)).replace("_AND_", "") for minor_key in minor_keys]

        y_labels = ["{} n_loops={}".format(
            str(minor_key), number_of_loops_per_subset_dict[minor_key]
        ).replace("_AND_", "") for minor_key in minor_keys]

        if number_of_unique_performances_per_subset_dict is not None:
            y_labels = ["{}, unique_perfs={}".format(
                y_labels[ix], number_of_unique_performances_per_subset_dict[minor_key]
            ) for ix, minor_key in enumerate(minor_keys)]

    else:
        # Majors are subset tags and minors are drum voice
        major_titles = ["{} {} n_loops={}".format(
            title_prefix, str(major_key), number_of_loops_per_subset_dict[major_key]
        ).replace("_AND_", "") for major_key in major_keys]

        if number_of_unique_performances_per_subset_dict is not None:
            major_titles = ["{}, unique_perfs={}".format(
                major_titles[ix], number_of_unique_performances_per_subset_dict[major_key]
            ) for ix, major_key in enumerate(major_keys)]

        minor_tags = ["{}".format(str(minor_key)) for minor_key in minor_keys]
        y_labels = minor_tags


    # Create a color palette for plots
    n_groups_per_plot = len(minor_tags)
    palette_resolution = int(254 // n_groups_per_plot)
    palette = [cc.rainbow[i * palette_resolution] for i in range(n_groups_per_plot)]

    # Figure holder for returning at the very end
    final_figure_layout = list()

    # Height of each subplot can be specified via the input parameters
    plot_height = int(plot_height_per_set * n_groups_per_plot)

    for major_ix, major_key in enumerate(major_keys):

        # Bokeh object holders
        legend_it = list()
        histogram_figures = list()
        scatter_figures = list()

        # Attach the X/Y axis of all plots
        if major_ix == 0 or synchronize_plots is False:
            p = figure(plot_width=plot_width, plot_height=plot_height)
        else:
            p = figure(plot_width=plot_width, plot_height=plot_height, x_range=p.x_range, y_range=p.y_range, title=None)

        t = Title()
        t.text = major_titles[major_ix]
        p.title = t

        vel_lines = []  # horizontal lines to show 0, 127 velocity limits
        for minor_ix, minor_key in enumerate(minor_keys):
            scatter_times, scatter_vels = scatters_dict[major_key][minor_key]
            heatmap_data, heatmap_extents = heatmaps_dict[major_key][minor_key]


            # Scatter Plot
            c = p.circle(x=scatter_times, y=(scatter_vels + 127 * 1.02 * minor_ix), color=palette[minor_ix],
                         alpha=0.1)
            legend_it.append(("{}".format(minor_tags[minor_ix]), [c]))
            scatter_figures.append(c)

            # Heat map plot Get Velocity Profile and Heat map
            im = p.image(
                image=[heatmap_data[:,::downsample_heat_maps_by]],
                x=heatmap_extents[0], y=heatmap_extents[2] + 127 * 1.02 * minor_ix,
                dw=heatmap_extents[1] - heatmap_extents[0],
                dh=heatmap_extents[3] - heatmap_extents[2],
                palette="Spectral11", level="image"
            )
            histogram_figures.append(im)

            # ygrid and yaxis settings
            vline_0vel = Span(location=minor_ix * (1.02 * 127), dimension='width', line_color='gray', line_width=0.55)
            #vline_127vel = Span(location=minor_ix * (1.02 * 127) + 127, dimension='width', line_color='black', line_width=0.55)
            vel_lines.extend([vline_0vel])

        # last vel line
        vline_127vel = Span(location=minor_ix * (1.02 * 127) + 127, dimension='width', line_color='black',
                            line_width=0.55)
        vel_lines.extend([vline_127vel])
        # add vel lines
        p.renderers.extend(vel_lines)

        # Legend stuff here
        legend_it.append(("Hide Heat Maps", histogram_figures))
        legend_it.append(("Hide Scatters", scatter_figures))
        legend = Legend(items=legend_it)
        legend.label_text_font_size = legend_fnt_size
        legend.click_policy = "hide"
        p.add_layout(legend, 'right')

        p.ygrid.grid_line_color = None
        p.yaxis.minor_tick_line_color = None
        p.yaxis.major_tick_line_color = None
        p.yaxis.ticker = 127 * 1.02 * np.arange(len(legend_it)) + 127 / 2
        p.yaxis.major_label_overrides = dict(
            zip(127 * 1.02 * np.arange(len(legend_it)) + 127 / 2, y_labels))
        p.yaxis.minor_tick_line_color = "#efefef"
        p.y_range.range_padding = 0.12

        # xgrid and xaxis settings
        p.xgrid.minor_grid_line_color = 'navy'
        p.xgrid.minor_grid_line_alpha = 0.1
        p.xgrid.grid_line_width = 5
        p.xaxis.ticker.num_minor_ticks = 4
        ticker = SingleIntervalTicker(interval=4, num_minor_ticks=4)
        p.xaxis.ticker = ticker
        p.xgrid.ticker = p.xaxis.ticker


        final_figure_layout.append(p)
    return final_figure_layout


def global_features_plotter(global_features_dict,
                            title_prefix="ridgeplot",
                            normalize_data=False,
                            analyze_combined_sets=True,
                            force_extract=False, plot_width=800, plot_height=1200, legend_fnt_size="8px",
                            scale_y=False, resolution=1000, plot_with_complement=False):

    """
    Creates multiple bokeh figures each of which corresponds to a feature extractable from a HVO_Sequence object via
    Feature_Extractor_From_HVO_Set object

    :param feature_extractors_list:         list of Feature_Extractor_From_HVO_Set instances
                                            each Feature_Extractor_From_HVO_Set instance corresponds to one
                                            HVO_Sequence set
    :param title_prefix:                    Prefix to title
    :param normalize_data:                  Normalizes the extracted feature values using mean and std of values
    :param analyze_combined_sets:           Adds an extra row which shows the feature information for
                                            all the sets combined
    :param force_extract:                   re-extracts features from the feature_extractor
    :param plot_width:                      width of each feature plot
    :param plot_height:                     height of each feature plot
    :param legend_fnt_size:                 font size of legends
    :param scale_y:                         if True, normalizes y-value of feature histograms to their max
    :param resolution:                      number of points used for calculating the probability distribution (pdf)
                                            using scipy.stats.kde.gaussian_kde
    :param plot_with_complement:            For each subplot, creates an additional plot on the right,
                                            this subplot shows the pdf of all the rows combined except that of the
                                            feature itself (adding rows of left plot to right plot  gives the
                                            same result)
    :return:
            a list of bokeh figures which can be externally plotted using show(grid(p, ncols=3))
    """

    # Extract features if not done already or force_extract required
    # also get the major/minor features available in the feature set
    # exp:  feat_major_fields   ["statistical", "micro-timing"]
    #       feat_minor_fields   ["noi", "lowness", "swing"]

    figure_layout = list()

    # compile a list of major keys and minor keys
    major_keys = []
    minor_keys = []
    for major_key in global_features_dict.keys():
        major_keys.append(major_key)
        for minor_key in global_features_dict[major_key].keys():
            if minor_key not in minor_keys:
                minor_keys.append(minor_key)

    # Each bokeh figure will show a single features value across all sets (accessible via feature_extractors_list)
    for major_key in major_keys:
        data_list = list()
        tags = list()
        for minor_key in minor_keys:
            if minor_key in global_features_dict[major_key].keys():
                data_for_minor_key = global_features_dict[major_key][minor_key]
                data_for_minor_key = np.where(np.isnan(data_for_minor_key), 0, data_for_minor_key)
                data_for_minor_key = np.where(np.isinf(data_for_minor_key), 0, data_for_minor_key)
                if data_for_minor_key.std()>0 and normalize_data is True:
                    data_for_minor_key = (data_for_minor_key - data_for_minor_key.mean())/data_for_minor_key.std()
                data_list.append(data_for_minor_key)
                tags.append("{} ".format(minor_key))
            # todo implement else section for filling zeros if key missing

        title = "{} - {}".format(title_prefix, major_key)

        # if requested to add another analysis containing all sets mixed together
        if analyze_combined_sets is True and len(data_list)>=1:
            combined_set = np.array([])
            for x in data_list:
                if x.size >= 1:
                    combined_set = np.append(combined_set, x).flatten()
            data_list.append(np.array(combined_set).flatten())
            tags.append("{} ".format("Combined"))

        if len(data_list) >= 1:
            if plot_with_complement is True:
                figure_layout.append(
                    ridge_kde_multi_feature_with_complement_set(
                        tags, data_list,
                        title=title,
                        resolution=resolution, plot_width=plot_width, plot_height=plot_height,
                        legend_fnt_size=legend_fnt_size, scale_y=scale_y)
                )
            else:
                figure_layout.append(
                    ridge_kde_multi_feature(
                        tags, data_list, resolution=resolution,
                        plot_width=plot_width, plot_height=plot_height,
                        title=title,
                        legend_fnt_size=legend_fnt_size,
                        plot_set_complement=False, scale_y=scale_y)
                )

    return figure_layout






def multi_voice_velocity_profile_plotter(
        feature_extractors_list,
        title_prefix="",
        drum_voice_keys=None,
        plot_width=1200, plot_height_per_set=400,legend_fnt_size="12px",
):
    """
    Plots all the velocities (with actual timing) for all (or a specified subset) of drum voices across all
    sets

    :param feature_extractors_list:                 A list of all feature_extractor instances created for each set
    :param title_prefix:                            Prefix to add to plot titles
    :param drum_voice_keys:                         A list of drum voices to plot info for, if None, works with all
    :param plot_width:                              Plot width
    :param plot_height_per_set:                     Controls height of plots
    :param legend_fnt_size:                         Size used for legend
    :return:
           a list of bokeh figures which can be externally plotted using show(grid(p, ncols=3))
    """

    # Compile a dictionary containing all velocity profiles for all sets
    # formatted as {"set_name": {"KICK": (time[...], velocity[...]), "SNARE": ... }}
    # Also compile the number of loops in each set
    velocity_profile_dicts = {}
    number_of_loops_in_sets = {}
    number_of_unique_performances_in_sets = {}
    for set_ix,  feature_extractor in enumerate(feature_extractors_list):
        # Get the velocity profiles for each set
        set_name = feature_extractor.name
        velocity_profile_dicts.update({
            set_name: feature_extractor.extract_velocity_profile()
        })
        # Get the number of loops in the set
        number_of_loops_in_sets.update({
            set_name: len(feature_extractor.hvo_dataset),
        })
        # Get the number of unique performances in the dataset
        number_of_unique_performances_in_sets.update({
            set_name: len(list(set(([hvo_seq.master_id for hvo_seq in feature_extractor.hvo_dataset]))))
        })


    # Create a color palette for plots
    n_sets = len(velocity_profile_dicts.keys())
    palette_resolution = int(254//n_sets)
    palette = [cc.rainbow[i*palette_resolution] for i in range(n_sets)]

    # If a specific subset of voices are NOT required, plot all
    if drum_voice_keys is None:
        drum_voice_keys = list(feature_extractors_list[0].hvo_dataset[0].drum_mapping.keys())

    # Figure holder for returning at the very end
    figure_layout_for_drum_voice = list()

    # Height of each subplot can be specified via the input parameters
    plot_height = int(plot_height_per_set * n_sets)

    # Start plotting
    # For each voice, stack the plots corresponding to the sets
    for voice_ix, drum_voice in enumerate(drum_voice_keys):

        # Bokeh object holders
        legend_it = list()
        histogram_figures = list()
        scatter_figures = list()

        # Attach the X/Y axis of all plots
        if voice_ix == 0:
            p = figure(plot_width=plot_width, plot_height=plot_height)
        else:
            p = figure(plot_width=plot_width, plot_height=plot_height, x_range=p.x_range, y_range=p.y_range, title=None)
        p.title = "{} Velocity Profile::{}".format(title_prefix, drum_voice)

        # In the figure for the voice, iterate over different sets
        for ix, set_key in enumerate(velocity_profile_dicts.keys()):
            (times, vels) = velocity_profile_dicts[set_key][drum_voice]

            # Scatter Plot
            c = p.circle(x=times, y=(vels+127*1.02*ix), color=palette[ix])
            legend_it.append(("{}".format(set_key), [c]))
            scatter_figures.append(c)

            # Heat map plot Get Velocity Profile and Heat map
            heat_map_plot_, extent = heat_map_plot(times, vels, s=8)
            im = p.image(image=[heat_map_plot_], x=extent[0], y=extent[2] + 127 * 1.02 * ix, dw=extent[1]-extent[0], dh=extent[3]-extent[2],
                         palette="Spectral11", level="image")
            histogram_figures.append(im)

        # Legend stuff here
        legend_it.append(("Hide Heat Maps", histogram_figures))
        legend_it.append(("Hide Scatters", scatter_figures))
        legend = Legend(items=legend_it)
        legend.label_text_font_size = legend_fnt_size
        legend.click_policy = "hide"
        p.add_layout(legend, 'right')


        # p.outline_line_color = None

        # ygrid and yaxis
        p.ygrid.grid_line_color = None
        p.yaxis.minor_tick_line_color = None
        p.yaxis.major_tick_line_color = None
        p.yaxis.ticker = 127*1.02*np.arange(len(legend_it))+127/2
        y_labels = ["{}, n loops = {},  unique performances = {}".format(
            set_name, number_of_loops_in_sets[set_name], number_of_unique_performances_in_sets[set_name]) for
            set_name in velocity_profile_dicts.keys()]
        p.yaxis.major_label_overrides = dict(
            zip(127*1.02*np.arange(len(legend_it))+127/2, y_labels))
        p.yaxis.minor_tick_line_color = "#efefef"
        p.y_range.range_padding = 0.12

        # xgrid and xaxis settings
        p.xgrid.minor_grid_line_color = 'navy'
        p.xgrid.minor_grid_line_alpha = 0.1
        p.xgrid.grid_line_width =5
        p.xaxis.ticker.num_minor_ticks = 4
        ticker = SingleIntervalTicker(interval=4, num_minor_ticks=4)
        p.xaxis.ticker = ticker
        p.xgrid.ticker = p.xaxis.ticker

        figure_layout_for_drum_voice.append(p)

    return figure_layout_for_drum_voice


##############################################
###
#      Plotting methods for rhythmic features
#       Extracted from one or multiple
#           Hvo_sequences set(s)
###
##############################################


def multi_feature_plotter(feature_extractors_list, title_prefix="ridgeplot", normalize_data=False, analyze_combined_sets=True,
                          force_extract=False, plot_width=800, plot_height=1200, legend_fnt_size="8px",
                          scale_y=False, resolution=1000, plot_with_complement=False):

    """
    Creates multiple bokeh figures each of which corresponds to a feature extractable from a HVO_Sequence object via
    Feature_Extractor_From_HVO_Set object

    :param feature_extractors_list:         list of Feature_Extractor_From_HVO_Set instances
                                            each Feature_Extractor_From_HVO_Set instance corresponds to one
                                            HVO_Sequence set
    :param title_prefix:                    Prefix to title
    :param normalize_data:                  Normalizes the extracted feature values using mean and std of values
    :param analyze_combined_sets:           Adds an extra row which shows the feature information for
                                            all the sets combined
    :param force_extract:                   re-extracts features from the feature_extractor
    :param plot_width:                      width of each feature plot
    :param plot_height:                     height of each feature plot
    :param legend_fnt_size:                 font size of legends
    :param scale_y:                         if True, normalizes y-value of feature histograms to their max
    :param resolution:                      number of points used for calculating the probability distribution (pdf)
                                            using scipy.stats.kde.gaussian_kde
    :param plot_with_complement:            For each subplot, creates an additional plot on the right,
                                            this subplot shows the pdf of all the rows combined except that of the
                                            feature itself (adding rows of left plot to right plot  gives the
                                            same result)
    :return:
            a list of bokeh figures which can be externally plotted using show(grid(p, ncols=3))
    """

    # Extract features if not done already or force_extract required
    # also get the major/minor features available in the feature set
    # exp:  feat_major_fields   ["statistical", "micro-timing"]
    #       feat_minor_fields   ["noi", "lowness", "swing"]

    figure_layout = list()

    # Run the extractor in case it hasn't been done already
    [feature_extractor.extract() for feature_extractor in feature_extractors_list]

    # Each bokeh figure will show a single features value across all sets (accessible via feature_extractors_list)
    for feat_key in feature_extractors_list[0].extracted_features_dict.keys():
        data_list = list()
        tags = list()
        for i in range(len(feature_extractors_list)):
            feature_extractors_list[i].extract()
            if feat_key in feature_extractors_list[i].extracted_features_dict.keys():
                data_for_feature = feature_extractors_list[i].extracted_features_dict[feat_key]
                data_for_feature = np.where(np.isnan(data_for_feature), 0, data_for_feature)
                data_for_feature = np.where(np.isinf(data_for_feature), 0, data_for_feature)
                if data_for_feature.std()>0 and normalize_data is True:
                    data_for_feature = (data_for_feature - data_for_feature.mean())/data_for_feature.std()
                data_list.append(data_for_feature)
                tags.append("{} ".format(feature_extractors_list[i].name))
            # todo implement else section for filling zeros if key missing

        title = "{} - {}".format(title_prefix, feat_key)

        # if requested to add another analysis containing all sets mixed together
        if analyze_combined_sets is True and len(data_list)>=1:
            combined_set = np.array([])
            for x in data_list:
                if x.size >= 1:
                    combined_set = np.append(combined_set, x).flatten()
            data_list.append(np.array(combined_set).flatten())
            tags.append("{} ".format("Combined"))

        if len(data_list) >= 1:
            if plot_with_complement is True:
                figure_layout.append(
                    ridge_kde_multi_feature_with_complement_set(
                        tags, data_list,
                        title=title,
                        resolution=resolution, plot_width=plot_width, plot_height=plot_height,
                        legend_fnt_size=legend_fnt_size, scale_y=scale_y)
                )
            else:
                figure_layout.append(
                    ridge_kde_multi_feature(
                        tags, data_list, resolution=resolution,
                        plot_width=plot_width, plot_height=plot_height,
                        title=title,
                        legend_fnt_size=legend_fnt_size,
                        plot_set_complement=False, scale_y=scale_y)
                )

    return figure_layout

def is_invertible(a):
    return a.shape[0] == a.shape[1] and np.linalg.matrix_rank(a) == a.shape[0]

def ridge_kde_multi_feature(tags, data_list, title="",
                        resolution=1000,
                        plot_width=1200, plot_height=2000,
                        legend_fnt_size="8px",
                        plot_set_complement=False, scale_y=True):

    """
    For a nd list of numpy arrays with corresponding tags, this method plots the probability ditstribution estimation
    of the numpy arrays in the same figure

    :param tags:                        ["Funk", "Rock"]
    :param data_list:                   [[  np.array(example 1), ..., np.array(example M)],
                                           [np.array(example 1), ..., np.array(example N)]  ]

                        Note: For calculating the pdfs the individual arrays within each group are appended together
                                inside the method
                                        [[  np.array(example 1, ..., example M)],
                                           [np.array(example 1, ..., example N)]  ]
    :param title:                       Figure Title
    :param resolution:                      number of points used for calculating the probability distribution (pdf)
                                            using scipy.stats.kde.gaussian_kde
    :param plot_width:                  Figure width
    :param plot_height:                 Figure height
    :param legend_fnt_size:             Size of legend
    :param plot_set_complement:         If true, plots the complement of each set, rather than the set itself
    :param scale_y:                     Normalizes max of pdfs to one
    :return:
        A bokeh figure with multiple rows, where each row shows the pdf data_list[i]
    """
    #todo add median, and quartiles to plot

    # create color palette (max index of cc.rainbow is 256)
    n_sets = len(data_list)
    palette_resolution = int(254//n_sets)
    palette = [cc.rainbow[i*palette_resolution] for i in range(n_sets)]

    for ix, data in enumerate(data_list):
        if data.size < 2:
            data = np.zeros(100)
            data_list[ix] = data

    range_min = min([d_.min() for d_ in data_list])
    range_max = max([d_.max() for d_ in data_list])
    delta_range = range_max - range_min
    range_min = range_min + delta_range * 2
    range_max = range_max - delta_range * 2

    # Create x axis data with required resolution
    x = linspace(range_min, range_max, resolution)

    # Create bokeh ColumnDataSource containing data structure
    source = ColumnDataSource(data=dict(x=x))

    def ridge(category, data, scale=1):
        # scale = (1/data.max()) if scale_y is True else 1
        return list(zip([category]*len(data), scale*data))

    p = figure(y_range=list(set(tags)), plot_width=plot_width, plot_height=plot_height)

    pdfs = []
    pdfs_max = []
    # calculate pdfs and keep track of their max vals
    for ix, data in enumerate(data_list):
        if plot_set_complement is True:
            complementary_data = np.array([])
            for ix_ in range(n_sets):
                if ix_ != ix:
                    complementary_data = np.append(complementary_data, data_list[ix_].flatten())
            data = complementary_data

        # Find kernel bandwidth using Scott's Rule of Thumb
        # https://en.wikipedia.org/wiki/Histogram#Scott's_normal_reference_rule
        if data.mean() == 0.0 and data.std() == 0.0:
            pdf = np.zeros(100)
            pdfs.append(pdf)
            pdfs_max.append(0)
        else:
            if data.min() != data.max():
                try:
                    pdf = gaussian_kde(data)
                    pdfs.append(pdf(x))
                    pdfs_max.append(pdf(x).max())
                except:
                    pdf = np.zeros(100)
                    pdfs.append(pdf)
                    pdfs_max.append(0)
            else:
                pdf = np.zeros(100)
                pdfs.append(pdf)
                pdfs_max.append(0)

    max_value = np.nanmax(np.array(pdfs_max))
    scale = 1 if max_value == 0 else 1.0 / max_value

    # start plotting
    legend_it = []
    for ix, data in enumerate(data_list):
        if plot_set_complement is True:
            complementary_data = np.array([])
            for ix_ in range(n_sets):
                if ix_ != ix:
                    complementary_data = np.append(complementary_data, data_list[ix_].flatten())
            data = complementary_data

        # Find kernel bandwidth using Scott's Rule of Thumb
        # https://en.wikipedia.org/wiki/Histogram#Scott's_normal_reference_rule

        tag = tags[ix]

        if data.mean() == 0.0 and data.std() == 0.0:
            y = ridge(tags[ix], data)
        else:
            y = ridge(tags[ix], pdfs[ix], scale)

        source.add(y, tag)

        legend_ = "{} ~ N({}, {})".format(tags[ix], round(data.mean(), 2), round(data.std(), 2))
        legend_ = ("~"+legend_ ) if plot_set_complement else legend_
        c = p.patch('x', tags[ix], alpha=0.6, color=palette[ix], line_color="black", source=source)
        legend_it.append((legend_, [c]))
        t = Title()
        t.text = title
        p.title = t

    legend = Legend(items=legend_it)
    legend.label_text_font_size = legend_fnt_size
    legend.click_policy = "hide"
    p.add_layout(legend, 'right')

    p.outline_line_color = None
    p.background_fill_color = "#efefef"

    p.ygrid.grid_line_color = None
    p.xgrid.grid_line_color = "#dddddd"
    p.xgrid.ticker = p.xaxis.ticker

    p.axis.minor_tick_line_color = None
    p.axis.major_tick_line_color = None
    p.axis.axis_line_color = None

    p.y_range.range_padding = 0.12

    return p


def ridge_kde_multi_feature_with_complement_set(tags, data_list,
                                                title="",
                                                resolution=1000,
                                                plot_width=1200, plot_height=1200,
                                                legend_fnt_size="8px",
                                                scale_y=True):
    """
        Same as ridge_kde_multi_feature, except that it returns two figures one exactly same as
        ridge_kde_multi_feature and the other one the complement for ridge_kde_multi_feature

        :param tags:                        ["Funk", "Rock"]
        :param data_list:                   [[  np.array(example 1), ..., np.array(example M)],
                                               [np.array(example 1), ..., np.array(example N)]  ]

                            Note: For calculating the pdfs the individual arrays within each group are appended together
                                    inside the method
                                            [[  np.array(example 1, ..., example M)],
                                               [np.array(example 1, ..., example N)]  ]
        :param title:                       Figure Title
        :param resolution:                      number of points used for calculating the probability distribution (pdf)
                                                using scipy.stats.kde.gaussian_kde
        :param plot_width:                  Figure width
        :param plot_height:                 Figure height
        :param legend_fnt_size:             Size of legend
        :param scale_y:                     Normalizes max of pdfs to one
        :return:
            A bokeh figure with multiple rows, where each row shows the pdf data_list[i]
        """

    p1 = ridge_kde_multi_feature(tags, data_list, title=title,
                                 resolution=resolution,
                                 plot_width=plot_width, plot_height=plot_height,
                                 legend_fnt_size=legend_fnt_size,
                                 plot_set_complement=False, scale_y=scale_y)

    p2 = ridge_kde_multi_feature(tags, data_list, title=title,
                                 resolution=resolution,
                                 plot_width=plot_width, plot_height=plot_height,
                                 legend_fnt_size=legend_fnt_size,
                                 plot_set_complement=True, scale_y=scale_y)

    p2.x_range = p1.x_range
    p2.y_range = p1.y_range

    return [p1, p2]


##############################################
###
#      Plotting violin plots
###
##################################################
def tabulated_violin_plot(data_dictionary, save_path=None, kernel_bandwidth=0.01,
                          width=1200, height=800, scatter_color='red', scatter_size=10, xrotation=45, font_size=16):
    '''
    **WORKS only if holoviews is installed**

    Plots the data in a dictionary as violin plots
    {
    "tab1-Data_xx": [data1, data2, ...],            # plots in tab1 -        Title tab1 -       labels as Data_xx
    "tab1-Data_yy": [data1, data2, ...],            # plots in tab1 -        Title tab1 -       labels as Data_yy
    "tab2-Data_xx": [data1, data2, ...],            # plots in tab2 -        Title tab2 -       labels as Data_xx
    "TTL::tab2-Data_yy": [data1, data2, ...],     # plots in TTL -   Title tab2::TTL  -        labels as Data_yy
    }

    :param data_dictionary:  KEYS can include tab title (left of a dash "-") and data category (right of a dash "-")
    :param save_path: (optional) if given, the figure is saved to the given path
    :param width:  (optional) width of the figure
    :param height: (optional) height of the figure
    :param scatter_color: (optional) color of the scatter points
    :param scatter_size: (optional) size of the scatter points
    :param xrotation: (optional) rotation of the x-axis labels
    :param font_size: (optional) font size of all the labels

    :return:    A bokeh figure
    '''
    # source https://holoviews.org/reference/elements/bokeh/Violin.html

    if _HAS_HOLOVIEWS:
        print("HAS HOLOVIEWS")
        groups = []
        categories = []
        values = []
        for key, val in data_dictionary.items():
            if (isinstance(val, dict)):
                for k, v_arr in val.items():
                    for v in v_arr:
                        groups.append(key)
                        categories.append(k)
                        values.append(v)
            else:
                for v in val:
                    groups.append(key.split("-")[0] if len(key.split("-")) > 1 else " ")
                    categories.append(key.split("-")[1] if len(key.split("-")) > 1 else key)
                    values.append(v)

        tab_labels = set(groups)
        panels = list()
        for tab_label in tab_labels:
            g_ = []
            c_ = []
            v_ = []
            for (group, category, val) in zip(groups, categories, values):
                if group == tab_label:
                    g_.append(group.replace("_", " "))
                    c_.append(category.replace("_", " "))
                    v_.append(val)

            violin = hv.Violin((c_, v_), ['Category'], 'Value', label='Violin Plots')

            scatter = hv.Scatter((c_, v_), label='Scatter Plots').opts(color=scatter_color, size=scatter_size).opts(opts.Scatter(jitter=0.2, alpha=0.5, size=6, height=400, width=600))

            violin = violin.opts(opts.Violin(height=height, show_legend=False, width=width,
                                             violin_color=hv.dim('Category').str(),
                                             xrotation=xrotation,
                                             fontsize={'xticks': font_size, 'yticks': font_size, 'xlabel': font_size, 'ylabel': font_size, 'title': font_size},
                                             bandwidth=kernel_bandwidth), clone=True)

            overlay = (violin * scatter).opts(title=tab_label.replace("_", " "), ylabel=" ", xlabel=" ")
            overlay.options(opts.NdOverlay(show_legend=True))

            fig = hv.render(overlay, backend='bokeh')
            fig.legend.click_policy="hide"
            panels.append(Panel(child=fig, title=tab_label.replace("_", " ").split("::")[-1]))

        tabs = Tabs(tabs=panels)

        if save_path is not None:
            if not save_path.endswith(".html"):
                save_path += ".html"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            save(tabs, save_path)

        return tabs
    else:
        return None


if __name__ == '__main__':

    output_file("ridgeplot.html")
    set_a = np.random.rand(200)
    set_b = 3*np.random.rand(100)-1
    set_c = 3*np.random.rand(100)+4
    """
    p = ridge_kde_multi_feature(tags=["rand1", "rand2", "rand3"], data_list=[set_a, set_b, set_b],
                            plot_set_complement=True)
    show(p)
    output_file("ridgeplot2.html")
    p = ridge_kde_multi_feature(tags=["rand1", "rand2", "rand3"], data_list=[set_a, set_b, set_b],
                            plot_set_complement=False)
    show(p)"""

    output_file("combined.html")

    p = ridge_kde_multi_feature(tags=["rand1", "rand2"], data_list=[set_a, set_b],
                                                resolution=1000,
                                                plot_width=600, plot_height=1200, legend_fnt_size="8px",
                                                scale_y=True)

    # show(layout(p))
