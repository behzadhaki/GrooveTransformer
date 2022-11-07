import sys
sys.path.insert(1, "../../")
sys.path.insert(1, "../")


#import wandb
"""wandb.login(key="b3cd55ed9905b389b249e4184179fb347e30e32a")"""
#run = wandb.init(project='GMD Analysis', entity='behzadhaki')

from preprocessed_dataset.Subset_Creators import subsetters

from GrooveEvaluator.feature_extractor import Feature_Extractor_From_HVO_SubSets
from GrooveEvaluator.plotting_utils import velocity_timing_heatmaps_scatter_plotter, global_features_plotter, separate_figues_by_tabs

from bokeh.io import output_file, show, save
from bokeh.layouts import layout, grid

# Create Subset Filters
styles = ["afrobeat", "afrocuban", "blues", "country", "dance",
          "funk", "gospel", "highlife", "hiphop", "jazz",
          "latin", "middleeastern", "neworleans", "pop",
          "punk", "reggae", "rock", "soul"]

# styles = ["rock"]

list_of_filter_dicts_for_subsets = []
for style in styles:
    list_of_filter_dicts_for_subsets.append({"style_primary": [style], "beat_type": ["beat"], "time_signature": ["4-4"]})

list_of_filter_dicts_for_subsets
tags_by_style_and_beat, subsets_by_style_and_beat = subsetters.HVOSetSubsetter(
    pickle_source_path="../preprocessed_dataset/datasets_extracted_locally/GrooveMidi/hvo_0.4.2/"
                       "Processed_On_17_05_2021_at_22_32_hrs",
    subset="GrooveMIDI_processed_train",
    hvo_pickle_filename="hvo_sequence_data.obj",
    list_of_filter_dicts_for_subsets=list_of_filter_dicts_for_subsets,
    max_len=32,
    ).create_subsets()

feature_extractors_for_subsets = Feature_Extractor_From_HVO_SubSets(
    hvo_subsets=subsets_by_style_and_beat,
    tags=tags_by_style_and_beat,
    auto_extract=True,
    max_samples_in_subset=10000
)
feature_extractors_for_subsets.get_few_hvo_samples(10)

output_file("misc/{}.html".format("temp_global_features"))
global_features_dict = feature_extractors_for_subsets.get_global_features_dicts(regroup_by_feature=True)
p = global_features_plotter(global_features_dict,
                            title_prefix="train_ground_truth",
                            normalize_data=False,
                            analyze_combined_sets=True,
                            force_extract=False, plot_width=800, plot_height=1200, legend_fnt_size="8px",
                            scale_y=False, resolution=100, plot_with_complement=False)
tabs = separate_figues_by_tabs(p, tab_titles=list(global_features_dict.keys()))
show(tabs)










#####
# HEATMAP PLOT TEST
#####

regroup_by_drum_voice = True
heatmaps_dict, scatters_dict = feature_extractors_for_subsets.get_velocity_timing_heatmap_dicts(
    s=(4, 10),
    bins=[32*8, 127],
    regroup_by_drum_voice=regroup_by_drum_voice)
#feature_dicts_grouped = feature_extractors_for_subsets.get_global_features_dicts()

output_file("misc/{}.html".format("temp_heat"))

p = velocity_timing_heatmaps_scatter_plotter(
    heatmaps_dict,
    scatters_dict,
    number_of_loops_per_subset_dict=feature_extractors_for_subsets.number_of_loops_in_sets,
    number_of_unique_performances_per_subset_dict=feature_extractors_for_subsets.number_of_unique_performances_in_sets,
    organized_by_drum_voice=regroup_by_drum_voice,  # denotes that the first key in heatmap and dict corresponds to drum voices
    title_prefix="",
    plot_width=1200, plot_height_per_set=100, legend_fnt_size="8px",
    synchronize_plots=True,
    downsample_heat_maps_by=1
)


# Assign the panels to Tabs
tabs = separate_figues_by_tabs(p, tab_titles=list(heatmaps_dict.keys()))

show(tabs)









"""


wandb.log({"Velocity Profiles Train Set4": [wandb.Html(open("misc/{}.html".format("temp")), inject=False)]})


from bokeh.models.widgets import Tabs, Panel
panels = []
titles = list(heatmaps_dict.keys())

for ix, _p in enumerate(p):
    panels.append(Panel(child=_p, title = titles[ix]))

# Assign the panels to Tabs
tabs = Tabs(tabs=panels)

show(tabs)

wandb.log(
{
                        'epoch': 0,
                        "Velocity Profiles Train Set": [wandb.Html(open("misc/{}.html".format("temp")))]
                    }
)

"""


"""

import numpy as np
import pandas as pd


from bokeh.io import output_file, show
from bokeh.models import (BasicTicker, ColorBar, ColumnDataSource,
                          LinearColorMapper, PrintfTickFormatter,)
from bokeh.plotting import figure
from bokeh.transform import transform

tag_ix = 0
[x0, x1, y0, y1] = list(heatmaps_dict["KICK"].values())[tag_ix][-1]
vel_time_data = list(heatmaps_dict["KICK"].values())[tag_ix][0]
n_steps, vel_range = len(vel_time_data[0]), len(vel_time_data)
time_stamps = np.linspace(x0, x1, n_steps)
vel_stamps = np.linspace(y0, y1, vel_range)
times_= []
vels_ = []
dens_ = []
for time_ix, time_stamp in enumerate(time_stamps):
    for vel_ix, vel_val in enumerate(vel_stamps):
        times_.append(time_stamp),
        vels_.append(vel_val),
        dens_.append(vel_time_data[vel_ix, time_ix])
heatmap = {"Time": times_, "Velocity": vels_, "Density": dens_}
df = pd.DataFrame(heatmap)
df.head()

# this is the colormap from the original NYTimes plot
colors = ["#75968f", "#a5bab7", "#c9d9d3", "#e2e2e2", "#dfccce", "#ddb7b1", "#cc7878", "#933b41", "#550b1d"]
mapper = LinearColorMapper(palette=colors, low=df.Density.min(), high=df.Density.max())

p = figure(plot_width=800, plot_height=300, title="HEATMAP",
           x_range=(0, 32), y_range=(0, 127))

p.rect(x="Time", y="Velocity", width=1, height=1, source=df,
       line_color=None, fill_color=transform('Density', mapper))

color_bar = ColorBar(color_mapper=mapper,
                     ticker=BasicTicker(desired_num_ticks=len(colors)),
                     formatter=PrintfTickFormatter(format="%d%%"))

p.add_layout(color_bar, 'right')

show(p)

# palette="Spectral11"
"""