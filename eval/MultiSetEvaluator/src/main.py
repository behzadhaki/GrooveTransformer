from eval.MultiSetEvaluator.src.utils import *
from bokeh.plotting import figure, output_file, show, save
from bokeh.models import Panel, Range
import pickle, bz2
import os

import holoviews as hv
from holoviews import opts
from bokeh.models import Tabs, Panel

hv.extension('bokeh')


def get_violin_bokeh_plot(feature_label, value_dict, kernel_bandwidth=0.01,
                          scatter_color='red', scatter_size=10, xrotation=45, font_size=16):
    c_, v_ = [], []
    for key, val in value_dict.items():
        c_.extend([key] * len(val))
        v_.extend(val)

    violin = hv.Violin((c_, v_), ['Category'], 'Value', label=feature_label)

    scatter = hv.Scatter((c_, v_), label='Scatter Plots').opts(color=scatter_color, size=scatter_size).opts(
        opts.Scatter(jitter=0.2, alpha=0.5, size=6, height=400, width=600))

    violin = violin.opts(opts.Violin(violin_color=hv.dim('Category').str(),
                                     xrotation=xrotation,
                                     fontsize={'xticks': font_size, 'yticks': font_size, 'xlabel': font_size,
                                               'ylabel': font_size, 'title': font_size},
                                     bandwidth=kernel_bandwidth), clone=True)

    overlay = (violin * scatter).opts(ylabel=" ", xlabel=" ")
    overlay.options(opts.NdOverlay(show_legend=True))

    fig = hv.render(overlay, backend='bokeh')
    fig.title = feature_label
    fig.legend.click_policy = "hide"
    # panels.append(Panel(child=fig, title=tab_label.replace("_", " ").split("::")[-1]))
    return fig



class MultiSetEvaluator:
    def __init__(self, groove_evaluator_sets, ignore_feature_keys=None, reference_set_label = "GT", anchor_set_label = None):
        """
        :param groove_evaluator_sets: dictionary of GrooveEvaluator objects to compare, example:
               groove_evaluator_sets = {
                                "groovae":
                                    GrooveEvaluator(...),
                                "rosy":
                                    GrooveEvaluator(...),
                                "hopeful":
                                    GrooveEvaluator(...),
                            }

        
        :param ignore_feature_keys: list of keys to ignore when comparing the GrooveEvaluators
                        (["Statistical::NoI", "Statistical::Total Step Density", "Statistical::Avg Voice Density",
                        "Statistical::Lowness", "Statistical::Midness", "Statistical::Hiness",
                        "Statistical::Vel Similarity Score", "Statistical::Weak to Strong Ratio",
                        "Syncopation::Lowsync", "Syncopation::Midsync", "Syncopation::Hisync",
                        "Syncopation::Lowsyness", "Syncopation::Midsyness", "Syncopation::Hisyness",
                        "Syncopation::Complexity"])
        """

        # ======================== INITIALIZATION ========================
        # =========     Static/Picklaable Attributes
        self.ignore_feature_keys = ignore_feature_keys
        self.groove_evaluator_sets = groove_evaluator_sets

        assert reference_set_label == "GT" or reference_set_label in self.groove_evaluator_sets.keys(), \
            f"Reference set label >>> {anchor_set_label} <<< not found in the set labels. " \
            f"Use either 'GT or one of the following: " \
            f"{self.groove_evaluator_sets.keys()}"
        self.reference_set_label = reference_set_label

        self.anchor_set_label = list(self.groove_evaluator_sets.keys())[0] if anchor_set_label is None else anchor_set_label
        assert self.anchor_set_label in self.groove_evaluator_sets.keys(), \
            f"Anchor set label >>> {self.anchor_set_label} <<< not found in the set labels. " \
            f"Use one of the following: " \
            f"{self.groove_evaluator_sets.keys()}"

        # =========     Dynamically Constructed Attributes
        self.feature_sets = dict()
        self.iid = dict()
        self.compile_necessary_attributes()

    # ================================================================
    # =========     Pickling Methods
    # ================================================================
    def __getstate__(self):
        state = {
            'ignore_feature_keys': self.ignore_feature_keys,
            'groove_evaluator_sets': self.groove_evaluator_sets,
            'reference_set_label': self.reference_set_label,
            'anchor_set_label': self.anchor_set_label
        }
        return state

    def __setstate__(self, state):
        # stored attributes
        self.ignore_feature_keys = state['ignore_feature_keys']
        self.groove_evaluator_sets = state['groove_evaluator_sets']
        self.reference_set_label = state['reference_set_label']
        self.anchor_set_label = state['anchor_set_label']

        # compute the rest of the attributes
        self.eval_labels = list()   # list of lists of set labels to compare
        self.feature_sets = dict()  # dict of dicts of features
        self.iid = dict()           # dict of dicts of inter intra distances and their stats
        self.compile_necessary_attributes()

    def compile_necessary_attributes(self):
        # ======================== Compile Feature Dicts for Each Set ========================
        # attributes to dynamically construct using the static data that are picklable
        # (to use in constructor and __setstate__)
        self.feature_sets = {
            "GT": flatten_subset_genres(get_gt_feats_from_evaluator(list(self.groove_evaluator_sets.values())[0]))}
        self.feature_sets.update({
            set_name: flatten_subset_genres(get_pd_feats_from_evaluator(eval)) for (set_name, eval) in
            self.groove_evaluator_sets.items()
        })

        allowed_analysis = list(self.feature_sets['GT'].keys())
        # remove ignored keys from allowed analysis
        if self.ignore_feature_keys is not None:
            for key in self.ignore_feature_keys:
                allowed_analysis.remove(key) if key in allowed_analysis else print(
                    f"Can't ignore requested Key [`{key}`] as is not found in the feature set")

        for set_name in self.feature_sets.keys():
            for key in list(self.feature_sets[set_name].keys()):
                if key not in allowed_analysis:
                    self.feature_sets[set_name].pop(key)

        # ======================== Compile Eval Labels ========================
        if len(self.groove_evaluator_sets.keys()) > 1:
            self.eval_labels = [[self.reference_set_label, self.anchor_set_label, k]
                                for k in self.groove_evaluator_sets.keys()
                           if k != self.anchor_set_label]
        else:
            self.eval_labels = [[self.reference_set_label, self.anchor_set_label]]

        # ======================== Compile Inter Intra Distances ========================
        self.calculate_inter_intra_distances()

    def dump(self, path):
        # check path ends with .MSEval.bz2
        if not path.endswith(".MSEval.bz2"):
            path += ".MSEval.bz2"

        # make sure path is a valid path
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with bz2.BZ2File(path, 'w') as f:
            pickle.dump(self, f)

    # ================================================================
    # =========     Inter Intra Statistics
    # ================================================================
    def calculate_inter_intra_distances(self):
        for set_labels in self.eval_labels:
            gt = self.feature_sets[set_labels[0]]
            set1 = self.feature_sets[set_labels[1]]
            set2 = self.feature_sets[set_labels[2]] if len(set_labels) > 2 else None

            distance_set_key = f"{set_labels[0]}_{set_labels[1]}_{set_labels[2]}" if len(set_labels) > 2 else \
                f"{set_labels[0]}_{set_labels[1]}"

            df, raw_data = compare_two_sets_against_ground_truth(gt, set1, set2, set_labels=set_labels)

            self.iid.update({distance_set_key: {"df": df, "raw_data": raw_data}})

    def save_statistics_of_inter_intra_distances(self, dir_path):

        # make sure path is a valid path
        os.makedirs(dir_path, exist_ok=True)
        dir_path = dir_path if os.path.isdir(dir_path) else os.path.dirname(dir_path)

        # save the statistics of the inter intra distances
        for set_tag, data in self.iid.items():
            data['df'].to_csv(os.path.join(dir_path, f"{set_tag}_inter_intra_statistics.csv"))
            print("Saved statistics of inter intra distances to: ", os.path.join(dir_path, f"{set_tag}_inter_intra_statistics.csv"))

    def get_inter_intra_pdf_plots(self, filename=None):
        """
        """
        inter_intra_pdf_tabs = []
        inter_intra_pdf_tab_labels = []

        for set_labels in self.eval_labels:
            # get precalculated inter intra distances
            distance_set_key = f"{set_labels[0]}_{set_labels[1]}_{set_labels[2]}" if len(set_labels) > 2 else \
                f"{set_labels[0]}_{set_labels[1]}"
            raw_data = self.iid[distance_set_key]['raw_data']

            # plot inter/intra pdfs

            bokeh_figs = plot_inter_intra_distance_distributions(
                raw_data, set_labels, ncols=3, figsize=(400, 300), legend_fs="6pt")

            # save figure
            inter_intra_pdf_tabs.append(bokeh_figs)
            inter_intra_pdf_tab_labels.append(" Vs. ".join(set_labels))

        tabs = Tabs(tabs=[Panel(child=inter_intra_pdf_tabs[i], title=inter_intra_pdf_tab_labels[i]) for i in
                          range(len(inter_intra_pdf_tabs))])

        if filename is not None:
            # make sure filename is html
            if not filename.endswith(".html"):
                filename = os.path.join(filename, "inter_intra_pdf_plots.html")

            # make sure path is a valid path
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            save(tabs, filename=filename)

        return tabs

    def get_kl_oa_plots(self, filename=None, figsize=(1200, 1000)):
        """
        """

        kl_oa_tabs = []
        kl_oa_tab_labels = []

        for set_labels in self.eval_labels:
            title = f"intra({set_labels[1]}) to inter {set_labels[0]}" if len(
                set_labels) <= 2 else f"intra({set_labels[1]}/{set_labels[2]}) to inter {set_labels[0]}"
            kl_oa_tab_labels.append(title)

            # get precalculated inter intra distances
            distance_set_key = f"{set_labels[0]}_{set_labels[1]}_{set_labels[2]}" if len(set_labels) > 2 else \
                f"{set_labels[0]}_{set_labels[1]}"

            # get inter intra pdf plot for current set
            bokeh_figs = get_KL_OA_plot(self.iid[distance_set_key]['df'], set_labels, figsize=figsize)

            kl_oa_tabs.append(Panel(child=bokeh_figs, title=title))

        tabs = Tabs(tabs=kl_oa_tabs)

        if filename is not None:
            # make sure filename is html
            if not filename.endswith(".html"):
                filename = os.path.join(filename, "kl_oa_plots.html")

            # make sure path is a valid path
            os.makedirs(os.path.dirname(filename), exist_ok=True)

            save(tabs, filename=filename)

        return tabs

    def get_pos_neg_hit_score_plots(self, filename=None, ncols=4, plot_width=400, plot_height=400,
                                    kernel_bandwidth=0.1,
                                    scatter_color='red', scatter_size=10, xrotation=45, font_size=10):
        pos_neg_hit_scores = dict()
        for set_label, groove_eval in self.groove_evaluator_sets.items():
            temp = groove_eval.get_pos_neg_hit_scores()
            for feat, value in temp.items():
                # reorganize data using feat as highest level key
                tab_label = feat.split(" - ")[0]
                feat_label = feat.split(" - ")[1]
                if tab_label not in pos_neg_hit_scores.keys():
                    pos_neg_hit_scores[tab_label] = dict()
                if feat_label not in pos_neg_hit_scores[tab_label].keys():
                    pos_neg_hit_scores[tab_label][feat_label] = dict()

                pos_neg_hit_scores[tab_label][feat_label][set_label] = value

        tab_grid = []
        for tab_label, tab_dict in pos_neg_hit_scores.items():
            figs = []
            for feat_label, feat_dict in tab_dict.items():
                figs.append(
                    get_violin_bokeh_plot(
                        feat_label, feat_dict, kernel_bandwidth=kernel_bandwidth,
                        scatter_color=scatter_color, scatter_size=scatter_size, xrotation=xrotation,
                        font_size=font_size))

            # sync axes
            y_max = max([fig.y_range.end for fig in figs])
            for fig in figs:
                fig.y_range.start = 0
                fig.y_range.end = y_max

            # sync all axes
            for fig in figs:
                fig.x_range = figs[0].x_range
                fig.y_range = figs[0].y_range

            tab_grid.append(Panel(child=gridplot(figs, ncols=ncols, plot_width=plot_width, plot_height=plot_height), title=tab_label))

        tabs = Tabs(tabs=tab_grid)

        if filename is not None:
            # make sure filename is html
            if not filename.endswith(".html"):
                filename = os.path.join(filename, "pos_neg_hit_scores.html")

            # make sure path is a valid path
            os.makedirs(os.path.dirname(filename), exist_ok=True)

            save(tabs, filename=filename)

        return tabs

    def get_velocity_distribution_plots(self, filename=None, ncols=4, plot_width=400, plot_height=400,
                                        kernel_bandwidth=0.1,
                                        scatter_color='red', scatter_size=10, xrotation=45, font_size=10):
        velocity_distributions = dict()
        for set_label, groove_eval in msEvaluator.groove_evaluator_sets.items():
            temp = groove_eval.get_velocity_distributions()
            for feat, value in temp.items():
                # reorganize data using feat as highest level key
                tab_label = feat.split(" - ")[0]
                feat_label = feat.split(" - ")[1]
                if tab_label not in velocity_distributions.keys():
                    velocity_distributions[tab_label] = dict()
                if feat_label not in velocity_distributions[tab_label].keys():
                    velocity_distributions[tab_label][feat_label] = dict()

                velocity_distributions[tab_label][feat_label][set_label] = value

        tab_grid = []
        for tab_label, tab_dict in velocity_distributions.items():
            figs = []
            for feat_label, feat_dict in tab_dict.items():
                figs.append(
                    get_violin_bokeh_plot(
                        feat_label, feat_dict, kernel_bandwidth=kernel_bandwidth,
                        scatter_color=scatter_color, scatter_size=scatter_size, xrotation=xrotation,
                        font_size=font_size))

            # sync axes
            for fig in figs:
                fig.y_range.start = 0
                fig.y_range.end = 1

            # sync all axes
            for fig in figs:
                fig.x_range = figs[0].x_range
                fig.y_range = figs[0].y_range

            tab_grid.append(Panel(child=gridplot(figs, ncols=ncols, plot_width=plot_width, plot_height=plot_height), title=tab_label))

        tabs = Tabs(tabs=tab_grid)



        if filename is not None:
            # make sure filename is html
            if not filename.endswith(".html"):
                filename = os.path.join(filename, "velocity_distributions.html")

            # make sure path is a valid path
            os.makedirs(os.path.dirname(filename), exist_ok=True)

            save(tabs, filename=filename)

        return tabs

    def get_offset_distribution_plots(self, filename=None, ncols=4, plot_width=400, plot_height=400,
                                      kernel_bandwidth=0.1,
                                      scatter_color='red', scatter_size=10, xrotation=45, font_size=10):
        offset_distributions = dict()
        for set_label, groove_eval in msEvaluator.groove_evaluator_sets.items():
            temp = groove_eval.get_offset_distributions()
            for feat, value in temp.items():
                # reorganize data using feat as highest level key
                tab_label = feat.split(" - ")[0]
                feat_label = feat.split(" - ")[1]
                if tab_label not in offset_distributions.keys():
                    offset_distributions[tab_label] = dict()
                if feat_label not in offset_distributions[tab_label].keys():
                    offset_distributions[tab_label][feat_label] = dict()

                offset_distributions[tab_label][feat_label][set_label] = value

        tab_grid = []
        for tab_label, tab_dict in offset_distributions.items():
            figs = []
            for feat_label, feat_dict in tab_dict.items():
                figs.append(
                    get_violin_bokeh_plot(
                        feat_label, feat_dict, kernel_bandwidth=kernel_bandwidth,
                        scatter_color=scatter_color, scatter_size=scatter_size, xrotation=xrotation, font_size=font_size))

            # sync axes
            for fig in figs:
                fig.y_range.start = -0.5
                fig.y_range.end = 0.5

            # sync all axes
            for fig in figs:
                fig.x_range = figs[0].x_range
                fig.y_range = figs[0].y_range

            tab_grid.append(Panel(child=gridplot(figs, ncols=ncols, plot_width=plot_width, plot_height=plot_height), title=tab_label))

        tabs = Tabs(tabs=tab_grid)

        if filename is not None:
            # make sure filename is html
            if not filename.endswith(".html"):
                filename = os.path.join(filename, "offset_distributions.html")

            # make sure path is a valid path
            os.makedirs(os.path.dirname(filename), exist_ok=True)

            save(tabs, filename=filename)

        return tabs


def load_multi_set_evaluator(path):
    with bz2.BZ2File(path, 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    from eval.GrooveEvaluator.src.evaluator import load_evaluator

    # prepare input data
    eval_1 = load_evaluator("testers/GrooveEvaluator/examples/test_set_full_robust_sweep_29.Eval.bz2")
    eval_2 = load_evaluator("testers/GrooveEvaluator/examples/test_set_full_colorful_sweep_41.Eval.bz2")

    # ignore_feature_keys = ["Statistical::NoI", "Statistical::Total Step Density", "Statistical::NEWWWWW"]
    ignore_feature_keys = None

    # construct MultiSetEvaluator
    msEvaluator = MultiSetEvaluator(
        groove_evaluator_sets={ "Model 1": eval_1, "Model 2": eval_2, "Model 3": eval_2}, #{ "groovae": eval_1, "Model 1": eval_2, "Model 2": eval_3 },  # { "groovae": eval_1}
        ignore_feature_keys=None, # ["Statistical::NoI", "Statistical::Total Step Density", "Statistical::NEWWWWW"]
        reference_set_label="GT",
        anchor_set_label=None # "groovae"
    )

    # dump MultiSetEvaluator
    msEvaluator.dump("testers/MultiSetEvaluator/misc/inter_intra_evaluator.MSEval.bz2")

    # load MultiSetEvaluator
    msEvaluator = load_multi_set_evaluator("testers/MultiSetEvaluator/misc/inter_intra_evaluator.MSEval.bz2")

    # save statistics
    msEvaluator.save_statistics_of_inter_intra_distances(dir_path="testers/MultiSetEvaluator/misc/multi_set_evaluator")

    # save inter intra pdf plots
    iid_pdfs_bokeh = msEvaluator.get_inter_intra_pdf_plots(filename="testers/MultiSetEvaluator/misc/multi_set_evaluator/iid_pdfs.html")

    # save kl oa plots
    KL_OA_plot = msEvaluator.get_kl_oa_plots(filename="testers/MultiSetEvaluator/misc/multi_set_evaluator")

    # get pos neg hit score plots
    pos_neg_hit_score_plots = msEvaluator.get_pos_neg_hit_score_plots(filename="testers/MultiSetEvaluator/misc/multi_set_evaluator/pos_neg_hit_scores.html")

    # get velocity distribution plots
    velocity_distribution_plots = msEvaluator.get_velocity_distribution_plots(filename="testers/MultiSetEvaluator/misc/multi_set_evaluator/velocity_distributions.html")

    # get offset distribution plots
    offset_distribution_plots = msEvaluator.get_offset_distribution_plots(filename="testers/MultiSetEvaluator/misc/multi_set_evaluator/offset_distributions.html")

   