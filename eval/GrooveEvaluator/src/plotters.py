## PLOTTING
import os
from scipy.ndimage.filters import gaussian_filter

import colorcet as cc
from numpy import linspace
from scipy.stats.kde import gaussian_kde

from bokeh.io import output_file, show
from bokeh.models import ColumnDataSource, FixedTicker, PrintfTickFormatter, Legend, SingleIntervalTicker, LinearAxis
from bokeh.plotting import figure
from bokeh.sampledata.perceptions import probly

from bokeh.layouts import layout, column, row

from bokeh.models import (BasicTicker, ColorBar, ColumnDataSource,
                          LinearColorMapper, PrintfTickFormatter,)
from bokeh.plotting import figure
from bokeh.transform import transform

import numpy as np
import pandas as pd


def velocity_timing_heatmaps_scatter_plotter(
        heatmaps_dict,
        scatters_dict,
        number_of_loops_per_subset_dict,
        number_of_unique_performances_per_subset_dict=None,
        organized_by_drum_voice=True,               # denotes that the first key in heatmap and dict corresponds to drum voices
        title_prefix="",
        plot_width=1200, plot_height_per_set=400,legend_fnt_size="12px",
        synchronize_plots=True,
        downsample_heat_maps_by=1
):

    # Create a separate figure for first keys
    major_keys = list(heatmaps_dict.keys())                     # (either drum voice or subset tag)
    minor_keys = list(heatmaps_dict[major_keys[0]].keys())      # (either drum voice or subset tag)

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

        p.title = major_titles[major_ix]

        for minor_ix, minor_key in enumerate(minor_keys):
            scatter_times, scatter_vels = scatters_dict[major_key][minor_key]
            heatmap_data, heatmap_extents = heatmaps_dict[major_key][minor_key]
            df = convert_heatmap_2d_arr_to_pandas (heatmap_data, heatmap_extents,
                                                   vel_offset = + 127 * 1.02 * minor_ix)

            # Scatter Plot
            c = p.circle(x=scatter_times, y=(scatter_vels + 127 * 1.02 * minor_ix), color=palette[minor_ix])
            legend_it.append(("{}".format(minor_tags[minor_ix]), [c]))
            scatter_figures.append(c)

            # Heat map plot Get Velocity Profile and Heat map
            mapper = LinearColorMapper(palette="Spectral11", low=df.Density.min(), high=df.Density.max())
            hist = p.rect(x="Time", y="Velocity", width=1, height=1, source=df,
                   line_color=None, fill_color=transform('Density', mapper), level="image")
            histogram_figures.append(hist)

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


def convert_heatmap_2d_arr_to_pandas(heatmap_data, heatmap_extents, vel_offset=0):
    [x0, x1, y0, y1] = heatmap_extents
    vel_time_data = heatmap_data
    data = heatmap_data.flatten()
    n_steps, vel_range = vel_time_data.shape[1], vel_time_data.shape[0]
    time_stamps = np.linspace(x0, x1, n_steps)
    vel_stamps = np.linspace(y0, y1, vel_range)
    times_ = []
    vels_ = []
    dens_ = []
    for time_ix, time_stamp in enumerate(time_stamps):
        for vel_ix, vel_val in enumerate(vel_stamps):
            times_.append(time_stamp),
            vels_.append(0)#vel_val + vel_offset),
            dens_.append(vel_time_data[vel_ix, time_ix])
    heatmap = {"Time": times_, "Velocity": vels_, "Density": dens_}
    df = pd.DataFrame(heatmap)
    return df