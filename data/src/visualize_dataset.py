import os, sys

from itertools import combinations
from tqdm import tqdm
from bokeh.layouts import gridplot
from bokeh.models import Tabs, Panel
from data.src.utils import get_per_performer_bokeh_histogram, get_genre_performer_heatmaps, get_per_performer_bokeh_pi_chart
from data import load_gmd_hvo_sequences

if __name__ == '__main__':
    data_identifier = "2Bar 4-4 Beats GMD"

    histograms_and_pi_charts = []
    pi_charts = []
    heatmaps = []


    for subset_tag in ["train", "test", "validation"]:
        # ==================================================================================================================
        # Load Data
        # ==================================================================================================================
        data_set = load_gmd_hvo_sequences(
            dataset_setting_json_path=f"data/dataset_json_settings/4_4_Beats_gmd.json",
            subset_tag=subset_tag,
            force_regenerate=False)

        # ==================================================================================================================
        # Visualize dataset
        # ==================================================================================================================
        p1 = get_per_performer_bokeh_histogram(data_set, f"GMD ({subset_tag.capitalize()} - {data_identifier})", ncols=3)
        p2 = get_per_performer_bokeh_pi_chart(data_set, f"{data_identifier}", ncols=3)
        histograms_and_pi_charts.append(gridplot([p1, p2], ncols=1))
        heatmaps.append(get_genre_performer_heatmaps(data_set, subset_tag, data_identifier))


    from bokeh.io import save
    path = f"documentation/chapter1_Data/figures/{data_identifier}_per_performer_histograms_and_pis.html"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tabs = Tabs(tabs=[Panel(child=histograms_and_pi_charts[0], title="Train"),
                      Panel(child=histograms_and_pi_charts[1], title="Test"),
                      Panel(child=histograms_and_pi_charts[2], title="Validation")])
    # save(gridplot(histograms_and_pi_charts, ncols=1), filename=path)
    save(tabs, filename=path)

    path = f"documentation/chapter1_Data/figures/{data_identifier}_performer_genre_counts.html"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tabs = Tabs(tabs=[Panel(child=heatmaps[0], title="Train"), Panel(child=heatmaps[1], title="Test"), Panel(child=heatmaps[2], title="Validation")])
    #save(gridplot(heatmaps, ncols=1), filename=path)
    save(tabs, filename=path)

