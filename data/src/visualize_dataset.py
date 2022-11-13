import os
from bokeh.layouts import gridplot
from bokeh.models import Tabs, Panel
from data.src.utils import get_per_performer_bokeh_histogram, get_genre_performer_heatmaps, get_per_performer_bokeh_pi_chart
from data import load_gmd_hvo_sequences
from eval.GrooveEvaluator import Evaluator

# Filters for genre division
list_of_filter_dicts_for_subsets = []
styles = [
    "afrobeat", "afrocuban", "blues", "country", "dance", "funk", "gospel", "highlife", "hiphop", "jazz",
    "latin", "middleeastern", "neworleans", "pop", "punk", "reggae", "rock", "soul"]
for style in styles:
    list_of_filter_dicts_for_subsets.append(
        {"style_primary": [style]} #, "beat_type": ["beat"], "time_signature": ["4-4"]}
    )

if __name__ == '__main__':
    data_identifier = "2Bar 4-4 Beats GMD"

    histograms_and_pi_charts = []
    pi_charts = []
    heatmaps = []
    evaluator_all_genres_combined = []
    evaluator_per_genre = []

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

        # ==================================================================================================================
        # velocity heatmaps
        # ==================================================================================================================
        evaluator_all_genres_combined.append(
            Evaluator(
                data_set,
                list_of_filter_dicts_for_subsets=None,
                _identifier=subset_tag,
                n_samples_to_use=-1,  # -1,
                max_hvo_shape=(32, 27),
                need_hit_scores=False,
                need_velocity_distributions=True,
                need_offset_distributions=True,
                need_rhythmic_distances=False,
                need_heatmap=True,
                need_global_features=True,
                need_audio=False,
                need_piano_roll=False,
                n_samples_to_synthesize_and_draw="all",  # "all",
                disable_tqdm=False
            )
        )

        evaluator_per_genre.append(
            Evaluator(
                data_set,
                list_of_filter_dicts_for_subsets=list_of_filter_dicts_for_subsets,
                _identifier=subset_tag,
                n_samples_to_use=-1,  # -1,
                max_hvo_shape=(32, 27),
                need_hit_scores=False,
                need_velocity_distributions=True,
                need_offset_distributions=True,
                need_rhythmic_distances=False,
                need_heatmap=True,
                need_global_features=True,
                need_audio=False,
                need_piano_roll=False,
                n_samples_to_synthesize_and_draw="all",  # "all",
                disable_tqdm=False
            )
        )





    from bokeh.io import save
    # ------------------------------------------------------------------------------------------------------------------
    # Save per performer plots (histograms and pie charts)
    # ------------------------------------------------------------------------------------------------------------------
    path = f"documentation/chapter1_Data/figures/{data_identifier}_per_performer_histograms_and_pis.html"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tabs = Tabs(tabs=[Panel(child=histograms_and_pi_charts[0], title="Train"),
                      Panel(child=histograms_and_pi_charts[1], title="Test"),
                      Panel(child=histograms_and_pi_charts[2], title="Validation")])
    # save(gridplot(histograms_and_pi_charts, ncols=1), filename=path)
    save(tabs, filename=path)

    # ------------------------------------------------------------------------------------------------------------------
    # Save genre performer heatmaps
    # ------------------------------------------------------------------------------------------------------------------
    path = f"documentation/chapter1_Data/figures/{data_identifier}_performer_genre_counts.html"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tabs = Tabs(tabs=[Panel(child=heatmaps[0], title="Train"),
                      Panel(child=heatmaps[1], title="Test"),
                      Panel(child=heatmaps[2], title="Validation")])
    #save(gridplot(heatmaps, ncols=1), filename=path)
    save(tabs, filename=path)

    # ------------------------------------------------------------------------------------------------------------------
    # Save velocity heatmaps (all genres combined / all voices combined)
    # ------------------------------------------------------------------------------------------------------------------
    path = f"documentation/chapter1_Data/figures/{data_identifier}_all_genres_velocity_heatmaps_all_voices_combined.html"
    os.makedirs(os.path.dirname(path), exist_ok=True)

    tabs = Tabs(tabs=[Panel(child=evaluator_all_genres_combined[0].get_velocity_heatmaps(regroup_by_drum_voice=False),
                            title="Train"),
                        Panel(child=evaluator_all_genres_combined[1].get_velocity_heatmaps(regroup_by_drum_voice=False),
                            title="Test"),
                        Panel(child=evaluator_all_genres_combined[2].get_velocity_heatmaps(regroup_by_drum_voice=False),
                            title="Validation")])
    save(tabs, filename=path)

    # ------------------------------------------------------------------------------------------------------------------
    # Save velocity heatmaps (per genre / grouped by drum voice)
    # ------------------------------------------------------------------------------------------------------------------
    path = f"documentation/chapter1_Data/figures/{data_identifier}_per_genre_velocity_heatmaps_grouped_by_drum_voice.html"
    os.makedirs(os.path.dirname(path), exist_ok=True)

    tabs = Tabs(tabs=[Panel(child=evaluator_per_genre[0].get_velocity_heatmaps(regroup_by_drum_voice=True),
                            title="Train"),
                        Panel(child=evaluator_per_genre[1].get_velocity_heatmaps(regroup_by_drum_voice=True),
                            title="Test"),
                        Panel(child=evaluator_per_genre[2].get_velocity_heatmaps(regroup_by_drum_voice=True),
                            title="Validation")])
    save(tabs, filename=path)

    # ------------------------------------------------------------------------------------------------------------------
    # Save velocity heatmaps (per genre / all voices combined)
    # ------------------------------------------------------------------------------------------------------------------
    path = f"documentation/chapter1_Data/figures/{data_identifier}_per_genre_velocity_heatmaps_all_voices_combined.html"
    os.makedirs(os.path.dirname(path), exist_ok=True)

    tabs = Tabs(tabs=[Panel(child=evaluator_per_genre[0].get_velocity_heatmaps(regroup_by_drum_voice=False),
                            title="Train"),
                      Panel(child=evaluator_per_genre[1].get_velocity_heatmaps(regroup_by_drum_voice=False),
                            title="Test"),
                      Panel(child=evaluator_per_genre[2].get_velocity_heatmaps(regroup_by_drum_voice=False),
                            title="Validation")])
    save(tabs, filename=path)

    # ------------------------------------------------------------------------------------------------------------------
    # Global Feature (all genres combined / all voices combined)
    # ------------------------------------------------------------------------------------------------------------------
    path = f"documentation/chapter1_Data/figures/{data_identifier}_all_genres_global_features_all_voices_combined.html"
    os.makedirs(os.path.dirname(path), exist_ok=True)

    tabs = Tabs(tabs=[Panel(child=evaluator_all_genres_combined[0].get_global_features_plot(only_combined_data_needed=True),
                            title="Train"),
                        Panel(child=evaluator_all_genres_combined[1].get_global_features_plot(only_combined_data_needed=True),
                            title="Test"),
                        Panel(child=evaluator_all_genres_combined[2].get_global_features_plot(only_combined_data_needed=True),
                            title="Validation")])
    save(tabs, filename=path)

    # ------------------------------------------------------------------------------------------------------------------
    # Global Feature (per genre / grouped by drum voice)
    # ------------------------------------------------------------------------------------------------------------------
    path = f"documentation/chapter1_Data/figures/{data_identifier}_per_genre_global_features_grouped_by_drum_voice.html"
    os.makedirs(os.path.dirname(path), exist_ok=True)

    tabs = Tabs(tabs=[Panel(child=evaluator_per_genre[0].get_global_features_plot(only_combined_data_needed=False),
                            title="Train"),
                        Panel(child=evaluator_per_genre[1].get_global_features_plot(only_combined_data_needed=False),
                            title="Test"),
                        Panel(child=evaluator_per_genre[2].get_global_features_plot(only_combined_data_needed=False),
                            title="Validation")])
    save(tabs, filename=path)

