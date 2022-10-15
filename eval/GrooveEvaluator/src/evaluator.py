import sys

sys.path.insert(1, "../../../")
sys.path.insert(1, "../../")
import numpy as np
import bz2
import wandb
from eval.GrooveEvaluator.src.feature_extractor import Feature_Extractor_From_HVO_SubSets
from eval.GrooveEvaluator.src.plotting_utils import global_features_plotter, velocity_timing_heatmaps_scatter_plotter
from eval.GrooveEvaluator.src.plotting_utils import separate_figues_by_tabs
from bokeh.embed import file_html
from bokeh.resources import CDN
from data.gmd.src import subsetters  # FIXME add preprocess_data directory
import pickle
import os
from tqdm import tqdm
import copy
from eval.post_training_evaluations.src.mgeval_rytm_utils import flatten_subset_genres


class Evaluator:
    # Eval 1. Test set loss
    #
    # todo 1. From Training Subsets, grab n_samples and pass to model

    def __init__(
            self,
            hvo_sequences_list,
            list_of_filter_dicts_for_subsets,
            _identifier="Train",
            n_samples_to_use=1024,
            max_hvo_shape=(32, 27),
            analyze_heatmap=True,
            analyze_global_features=True,
            analyze_piano_roll=True,
            analyze_audio=True,
            n_samples_to_synthesize="all",
            n_samples_to_draw_pianorolls="all",
            disable_tqdm=False,
            # todo distance measures KL, Overlap, intra and interset
    ):
        """
        This class will perform a thorough Intra- and Inter- evaluation between ground truth data and predictions

        :param hvo_sequences_list:        A list of hvo_sequences samples
        :param list_of_filter_dicts_for_subsets
        :param _identifier:               Text identifier for set comparison --> Train if dealing with evaluating
                                            predictions of the training set. Test if evaluating performance on test set
        :param max_hvo_shape:               tuple of (steps, 3*n_drum_voices) --> fits all sequences to this shape
                                                    by trimming or padding them
        :param n_samples_to_use:            number of samples to use for evaluation (uniformly samples n_samples_to_use
                                            from all classes in the ground truth set)
        :param analyze_heatmap:

        """
        self.analyze_heatmap, self.analyze_global_features = analyze_heatmap, analyze_global_features
        self.analyze_piano_roll, self.analyze_audio = analyze_piano_roll, analyze_audio
        self.disable_tqdm = disable_tqdm

        # Create subsets of data
        gt_subsetter_sampler = subsetters.GrooveMidiSubsetterAndSampler(
            hvo_sequences_list,
            list_of_filter_dicts_for_subsets=list_of_filter_dicts_for_subsets,
            number_of_samples=n_samples_to_use,
            max_hvo_shape=max_hvo_shape
        )

        # _gt_tags --> ground truth tags for each subset in _gt_subsets
        self._gt_tags, self._gt_subsets = gt_subsetter_sampler.get_subsets()

        # _gt_hvos_array_tags --> ground truth tags for each
        # _gt_hvos_arrayS --> a numpy array containing all samples in hvo format
        self._gt_hvo_sequences = []
        self._gt_hvos_array_tags, self._gt_hvos_array, self._prediction_hvo_seq_templates = [], [], []
        for subset_ix, tag in enumerate(self._gt_tags):
            for sample_ix, sample_hvo in enumerate(self._gt_subsets[subset_ix]):
                self._gt_hvo_sequences.append(sample_hvo)
                self._gt_hvos_array_tags.append(tag)
                self._gt_hvos_array.append(sample_hvo.get("hvo"))
                self._prediction_hvo_seq_templates.append(sample_hvo.copy_empty())

        self._gt_hvos_array = np.stack(self._gt_hvos_array)

        # a text to identify the evaluator (exp "Train_Epoch1", "Test_Epoch1")
        self._identifier = _identifier

        # Subset evaluator for ground_truth data
        self.gt_SubSet_Evaluator = HVOSeq_SubSet_Evaluator(
            self._gt_subsets,  # Ground Truth typically
            self._gt_tags,
            "{}_Ground_Truth".format(self._identifier),  # a name for the subset
            disable_tqdm=self.disable_tqdm,
            group_by_minor_keys=True,
            analyze_heatmap=analyze_heatmap,
            analyze_global_features=analyze_global_features
        )

        # Empty place holder for predictions, also Placeholder Subset evaluator for predicted data
        self._prediction_tags, self._prediction_subsets = None, None
        self.prediction_SubSet_Evaluator = None
        self._prediction_hvos_array = None

        # Get the index for the samples that will be synthesized and also for which the piano-roll can be generated
        self.audio_sample_locations = self.get_sample_indices(n_samples_to_synthesize)
        self.piano_roll_sample_locations = self.get_sample_indices(n_samples_to_draw_pianorolls)

        self._gt_logged_once = False  # Flag will be set to True when ground truth data is once evaluated
        self._gt_logged_once_wandb = False  # Flag will be set to True when ground truth data is once evaluated
        # for WANDB

    def get_logging_dict(self, velocity_heatmap_html="default", global_features_html="default",
                         piano_roll_html="default", audio_files="default",
                         sf_paths="default",
                         recalculate_ground_truth=False):

        assert velocity_heatmap_html == "default" or type(velocity_heatmap_html) == bool
        assert global_features_html == "default" or type(global_features_html) == bool
        assert piano_roll_html == "default" or type(piano_roll_html) == bool
        assert audio_files == "default" or type(audio_files) == bool

        sf_paths = ["hvo_sequence/soundfonts/Standard_Drum_Kit.sf2"] if sf_paths == "default" else sf_paths
        velocity_heatmap_html = self.analyze_heatmap if "default" in velocity_heatmap_html else velocity_heatmap_html
        global_features_html = self.analyze_global_features if "default" in global_features_html else global_features_html
        piano_roll_html = self.analyze_piano_roll if "default" in piano_roll_html else piano_roll_html
        audio_files = self.analyze_audio if "default" in audio_files else audio_files

        _gt_logging_data = None
        # Get logging data for ground truth data
        if recalculate_ground_truth is True or self._gt_logged_once is False:
            _gt_logging_data = self.gt_SubSet_Evaluator.get_logging_dict(
                velocity_heatmap_html=velocity_heatmap_html,
                global_features_html=global_features_html,
                piano_roll_html=piano_roll_html,
                audio_files=audio_files,
                sf_paths=sf_paths,
                for_audios_use_specific_samples_at=self.audio_sample_locations,
                for_piano_rolls_use_specific_samples_at=self.piano_roll_sample_locations
            )
            self._gt_logged_once = True


        _predicted_logging_data = self.prediction_SubSet_Evaluator.get_logging_dict(
            velocity_heatmap_html=velocity_heatmap_html,
            global_features_html=global_features_html,
            piano_roll_html=piano_roll_html,
            audio_files=audio_files,
            sf_paths=sf_paths,
            for_audios_use_specific_samples_at=self.audio_sample_locations,
            for_piano_rolls_use_specific_samples_at=self.piano_roll_sample_locations
        ) if self.prediction_SubSet_Evaluator is not None else None

        return _gt_logging_data, _predicted_logging_data

    def get_wandb_logging_media(self, velocity_heatmap_html="default", global_features_html="default",
                                piano_roll_html="default", audio_files="default",
                                sf_paths="default",
                                recalculate_ground_truth=False):

        assert velocity_heatmap_html == "default" or type(velocity_heatmap_html) == bool
        assert global_features_html == "default" or type(global_features_html) == bool
        assert piano_roll_html == "default" or type(piano_roll_html) == bool
        assert audio_files == "default" or type(audio_files) == bool

        velocity_heatmap_html = self.analyze_heatmap if "default" in velocity_heatmap_html else velocity_heatmap_html
        global_features_html = self.analyze_global_features if "default" in global_features_html else global_features_html
        piano_roll_html = self.analyze_piano_roll if "default" in piano_roll_html else piano_roll_html
        audio_files = self.analyze_audio if "default" in audio_files else audio_files

        # Get logging data for ground truth data
        if recalculate_ground_truth is True or self._gt_logged_once_wandb is False:

            gt_logging_media = self.gt_SubSet_Evaluator.get_wandb_logging_media(
                velocity_heatmap_html=velocity_heatmap_html,
                global_features_html=global_features_html,
                piano_roll_html=piano_roll_html,
                audio_files=audio_files,
                sf_paths=sf_paths,
                use_specific_samples_at=self.audio_sample_locations
            )
            self._gt_logged_once_wandb = True
        else:
            gt_logging_media = {}

        predicted_logging_media = self.prediction_SubSet_Evaluator.get_wandb_logging_media(
            velocity_heatmap_html=velocity_heatmap_html,
            global_features_html=global_features_html,
            piano_roll_html=piano_roll_html,
            audio_files=audio_files,
            sf_paths=sf_paths,
            use_specific_samples_at=self.audio_sample_locations
        ) if self.prediction_SubSet_Evaluator is not None else {}

        results = {x: {} for x in gt_logging_media.keys()}
        results.update({x: {} for x in predicted_logging_media.keys()})

        for key in results.keys():
            if key in gt_logging_media.keys():
                results[key].update(gt_logging_media[key])
            if key in predicted_logging_media.keys():
                results[key].update(predicted_logging_media[key])

        return results

    def get_hits_accuracies(self, drum_mapping):
        n_drum_voices = len(drum_mapping.keys())
        gt = self._gt_hvos_array[:, :, :n_drum_voices]
        pred = self._prediction_hvos_array[:, :, :n_drum_voices]
        n_examples = gt.shape[0]
        # Flatten
        accuracies = {"Hits_Accuracy": {self._identifier: {}}}
        for i, drum_voice in enumerate(drum_mapping.keys()):
            _gt = gt[:, :, i]
            _pred = pred[:, :, i]
            n_hits = _gt.shape[-1]
            accuracies["Hits_Accuracy"][self._identifier].update({"{}".format(drum_voice, self._identifier):
                                                                      ((_gt == _pred).sum(axis=-1) / n_hits).mean()})

        gt = gt.reshape((n_examples, -1))
        pred = pred.reshape((n_examples, -1))
        n_hits = gt.shape[-1]
        accuracies["Hits_Accuracy"][self._identifier].update(
            {"Overall".format(self._identifier): ((gt == pred).sum(axis=-1) / n_hits).mean()})

        return accuracies

    def get_velocity_errors(self, drum_mapping):
        n_drum_voices = len(drum_mapping.keys())
        gt = self._gt_hvos_array[:, :, n_drum_voices:2 * n_drum_voices]
        pred = self._prediction_hvos_array[:, :, n_drum_voices:2 * n_drum_voices]

        n_examples = gt.shape[0]
        # Flatten
        errors = {"Velocity_MSE": {self._identifier: {}}}
        for i, drum_voice in enumerate(drum_mapping.keys()):
            _gt = gt[:, :, i]
            _pred = pred[:, :, i]
            errors["Velocity_MSE"][self._identifier].update({"{}".format(drum_voice, self._identifier):
                                                                 (((_gt - _pred) ** 2).mean(axis=-1)).mean()})

        gt = gt.reshape((n_examples, -1))
        pred = pred.reshape((n_examples, -1))
        errors["Velocity_MSE"][self._identifier].update(
            {"Overall".format(self._identifier): (((gt - pred) ** 2).mean(axis=-1)).mean()})

        return errors

    def get_micro_timing_errors(self, drum_mapping):
        n_drum_voices = len(drum_mapping.keys())
        gt = self._gt_hvos_array[:, :, 2 * n_drum_voices:]
        pred = self._prediction_hvos_array[:, :, 2 * n_drum_voices:]
        n_examples = gt.shape[0]
        # Flatten
        errors = {"Micro_Timing_MSE": {self._identifier: {}}}
        for i, drum_voice in enumerate(drum_mapping.keys()):
            _gt = gt[:, :, i]
            _pred = pred[:, :, i]
            errors["Micro_Timing_MSE"][self._identifier].update({"{}".format(drum_voice, self._identifier):
                                                                     (((_gt - _pred) ** 2).mean(axis=-1)).mean()})

        gt = gt.reshape((n_examples, -1))
        pred = pred.reshape((n_examples, -1))
        errors["Micro_Timing_MSE"][self._identifier].update(
            {"Overall".format(self._identifier): (((gt - pred) ** 2).mean(axis=-1)).mean()})

        return errors

    def get_rhythmic_distances(self):
        gt_set = {self._gt_tags[ix]: subset for ix, subset in enumerate(self._gt_subsets)}
        predicted_set = {self._prediction_tags[ix]: subset for ix, subset in enumerate(self._prediction_subsets)}

        distances_dict = None

        for tag in tqdm(predicted_set.keys(),
                        desc='Calculating Rhythmic Distances - {}'.format(self._identifier),
                        disable=self.disable_tqdm
                        ):
            for sample_ix, predicted_sample_hvo in enumerate(predicted_set[tag]):
                distances_dictionary = predicted_sample_hvo.calculate_all_distances_with(gt_set[tag][sample_ix])

                if distances_dict is None:
                    distances_dict = {x: [] for x in distances_dictionary.keys()}

                for key in distances_dictionary.keys():
                    distances_dict[key].append(distances_dictionary[key])

        for key in distances_dict.keys():
            summary = {"mean": np.mean(distances_dict[key]),
                       "min": np.min(distances_dict[key]), "max": np.max(distances_dict[key]),
                       "median": np.percentile(distances_dict[key], 50),
                       "q1": np.percentile(distances_dict[key], 25),
                       "q3": np.percentile(distances_dict[key], 75)}

            distances_dict[key] = {self._identifier: summary}

        return distances_dict

    def get_ground_truth_hvo_sequences(self):
        return copy.deepcopy(self._gt_hvo_sequences)

    def get_ground_truth_hvos_array(self):
        return copy.deepcopy(self._gt_hvos_array)

    def add_predictions(self, prediction_hvos_array):
        self._prediction_hvos_array = prediction_hvos_array
        self._prediction_tags, self._prediction_subsets = \
            subsetters.convert_hvos_array_to_subsets(
                self._gt_hvos_array_tags,
                prediction_hvos_array,
                self._prediction_hvo_seq_templates
            )

        self.prediction_SubSet_Evaluator = HVOSeq_SubSet_Evaluator(
            self._prediction_subsets,
            self._prediction_tags,
            "{}_Predictions".format(self._identifier),  # a name for the subset
            disable_tqdm=self.disable_tqdm,
            group_by_minor_keys=True,
            analyze_heatmap=self.analyze_heatmap,
            analyze_global_features=self.analyze_global_features
        )

    def dump(self, path=None, fname="evaluator"):  # todo implement in comparator
        if path is None:
            path = os.path.join("misc")

        if not os.path.exists(path):
            os.makedirs(path)

        ofile = bz2.BZ2File(os.path.join(path, f"{self._identifier}_{fname}.Eval.bz2"), 'wb')
        pickle.dump(self, ofile)
        ofile.close()

        print(f"Dumped Evaluator to {os.path.join(path, f'{self._identifier}_{fname}.Eval.bz2')}")

    def get_sample_indices(self, n_samples_per_subset="all"):
        assert n_samples_per_subset == "all" or isinstance(n_samples_per_subset, int)

        subsets = self._gt_subsets
        tags = self._gt_tags
        sample_locations = {tag: [] for tag in tags}

        for subset_ix, subset in enumerate(subsets):
            n_samples = min(len(subset), n_samples_per_subset) if n_samples_per_subset != "all" else len(subset)
            for i in range(n_samples):
                sample_locations[tags[subset_ix]].append(i)

        return sample_locations


PATH_DICT_TEMPLATE = {
    "root_dir": "",  # ROOT_DIR to save data
    "project_name": '',  # GROOVE_TRANSFORMER_INFILL or GROOVE_TRANSFORMER_TAP2DRUM
    "run_name": '',  # WANDB RUN NAME run = wandb.init(...
    "set_identifier": '',  # TRAIN OR TEST
    "epoch": '',
}


class HVOSeq_SubSet_Evaluator(object):
    # todo 1. From Training Subsets, grab n_samples and pass to model

    def __init__(
            self,
            set_subsets,
            set_tags,
            set_identifier,
            max_samples_in_subset=None,
            n_samples_to_synthesize_visualize=10,
            disable_tqdm=True,
            group_by_minor_keys=True,
            analyze_heatmap=True,
            analyze_global_features=True
    ):

        """
        Used for evaluating a set containing multiple subsets of hvo_sequence samples

        :param set_subsets:             list of list of HVO_Sequences (i.e. list of subsets)
                                                Example --> [[hvo_seq0, ..., N], ..., [hvo_seq0, ..., N]]
        :param set_tags:                list of tags for each subset
                                                Example --> [     "ROCK"       , ...,      "SOUL"       ]
        :param set_identifier:          A tag for the set to be evaluated
                                                (example: "TRAIN_Ground_Truth" or "TRAIN_Predictions")
        :param max_samples_in_subset:   Max number of samples in each subset (will be randomly sampled from master set)
        :param n_samples_to_synthesize_visualize:
                                        Number of samples (per subset) that will be used for generating audios and/or
                                            pianorolls

        :param disable_tqdm:                True if you don't want to use tqdm
        :param group_by_minor_keys:         if True, plots are grouped per feature/drum voice or,
                                            otherwise grouped by style
        :param analyze_heatmap:             True/False if velocity heatmap analysis is needed
        :param analyze_global_features:     True/False if global feature  analysis is needed
        """
        self.__version__ = "0.0.0"

        self.set_identifier = set_identifier

        self.disable_tqdm = disable_tqdm
        self.group_by_minor_keys = group_by_minor_keys

        self.feature_extractor = None  # instantiates in

        self.max_samples_in_subset = max_samples_in_subset
        self.analyze_heatmap = analyze_heatmap
        self.analyze_global_features = analyze_global_features
        self.vel_heatmaps_dict = None
        self.vel_scatters_dict = None
        self.vel_heatmaps_bokeh_fig = None
        self.global_features_dict = None
        self.global_features_bokeh_fig = None

        # Store subsets and tags locally, and auto-extract
        self.__set_subsets = None
        self.__set_tags = None
        self.tags_subsets = (set_tags, set_subsets)         # auto extracts here!!!

        # Flag to re-extract if data changed
        self.__analyze_Flag = True

        # Audio Params
        self.n_samples_to_synthesize_visualize = n_samples_to_synthesize_visualize
        self._sampled_hvos = None

    @property
    def tags_subsets(self):
        return self.__set_tags, self.__set_subsets

    @tags_subsets.setter
    def tags_subsets(self, tags_subsets_tuple):
        tags = tags_subsets_tuple[0]
        subsets = tags_subsets_tuple[1]
        assert len(tags) == len(subsets), "Length mismatch between Tags and HVO Subsets : {} Tags vs " \
                                          "{} HVO_Seq Subsets".format(len(tags), len(subsets))
        self.__set_tags = tags
        self.__set_subsets = subsets

        # Reset calculator Flag
        self.__analyze_Flag = True

        # Create a new feature extractor for subsets
        self.feature_extractor = Feature_Extractor_From_HVO_SubSets(
            hvo_subsets=self.__set_subsets,
            tags=self.__set_tags,
            auto_extract=False,
            max_samples_in_subset=self.max_samples_in_subset,
        )

        if self.analyze_global_features:
            self.feature_extractor.extract(use_tqdm=not self.disable_tqdm)
            self.global_features_dict = self.feature_extractor.get_global_features_dicts(
                regroup_by_feature=self.group_by_minor_keys)

        print(self.analyze_heatmap)
        if self.analyze_heatmap:
            self.vel_heatmaps_dict, self.vel_scatters_dict = self.feature_extractor.get_velocity_timing_heatmap_dicts(
                s=(4, 10),
                bins=[32 * 8, 127],
                regroup_by_drum_voice=self.group_by_minor_keys
            )

            # todo
        # self.global_features_dict = self.feature_extractor.

    def get_vel_heatmap_bokeh_figures(
            self, plot_width=1200, plot_height_per_set=100, legend_fnt_size="8px",
            synchronize_plots=True,
            downsample_heat_maps_by=1
    ):

        # order
        tags = copy.deepcopy(self.tags_subsets[0])
        tags.sort()

        _vel_heatmaps_dict = {x: {} for x in self.vel_heatmaps_dict.keys()}
        _vel_scatters_dict = {x: {} for x in self.vel_scatters_dict.keys()}

        for inst in self.vel_heatmaps_dict.keys():
            for tag in tags:
                _vel_heatmaps_dict[inst][tag] = self.vel_heatmaps_dict[inst][tag]
                _vel_scatters_dict[inst][tag] = self.vel_scatters_dict[inst][tag]

        self.vel_heatmaps_dict = _vel_heatmaps_dict
        self.vel_scatters_dict = _vel_scatters_dict

        p = velocity_timing_heatmaps_scatter_plotter(
            self.vel_heatmaps_dict,
            self.vel_scatters_dict,
            number_of_loops_per_subset_dict=self.feature_extractor.number_of_loops_in_sets,
            number_of_unique_performances_per_subset_dict=self.feature_extractor.number_of_unique_performances_in_sets,
            organized_by_drum_voice=self.group_by_minor_keys,
            title_prefix=self.set_identifier,
            plot_width=plot_width, plot_height_per_set=plot_height_per_set, legend_fnt_size=legend_fnt_size,
            synchronize_plots=synchronize_plots,
            downsample_heat_maps_by=downsample_heat_maps_by
        )
        tabs = separate_figues_by_tabs(p, tab_titles=list(self.vel_heatmaps_dict.keys()))
        return tabs

    def get_global_features_bokeh_figure(self, plot_width=800, plot_height=1200,
                                         legend_fnt_size="8px", resolution=100):
        p = global_features_plotter(
            self.global_features_dict,
            title_prefix=self.set_identifier,
            normalize_data=False,
            analyze_combined_sets=True,
            force_extract=False, plot_width=plot_width, plot_height=plot_height,
            legend_fnt_size=legend_fnt_size,
            scale_y=False, resolution=resolution)
        tabs = separate_figues_by_tabs(p, tab_titles=list(self.global_features_dict.keys()))
        return tabs

    def get_hvo_samples_located_at(self, use_specific_samples_at, force_get=False):
        tags, subsets = self.tags_subsets

        if use_specific_samples_at is None:
            use_specific_samples_at = {tag: list(range(len(subsets[ix]))) for ix, tag in
                                       enumerate(tags)}  # TODO test this

        if self._sampled_hvos is None or force_get:
            self._sampled_hvos = {x: [] for x in tags}
            for subset_ix, tag in enumerate(tags):
                self._sampled_hvos[tag] = [subsets[subset_ix][ix] for ix in use_specific_samples_at[tag]]

            return self._sampled_hvos

        else:
            return self._sampled_hvos

    def get_audios(self, sf_paths, use_specific_samples_at=None):
        """ use_specific_samples_at: must be a list of tuples of (subset_ix, sample_ix) denoting to get
        audio from the sample_ix in subset_ix """

        self._sampled_hvos = self.get_hvo_samples_located_at(use_specific_samples_at)

        if not isinstance(sf_paths, list):
            sf_paths = [sf_paths]

        audios = []
        captions = []

        for key in tqdm(self._sampled_hvos.keys(),
                        desc='Synthesizing samples - {} '.format(self.set_identifier),
                        disable=self.disable_tqdm):
            for sample_hvo in self._sampled_hvos[key]:
                # randomly select a sound font
                sf_path = sf_paths[np.random.randint(0, len(sf_paths))]
                audios.append(sample_hvo.synthesize(sf_path=sf_path))
                captions.append("{}_{}_{}.wav".format(
                    self.set_identifier, sample_hvo.metadata["style_primary"],
                    sample_hvo.metadata["master_id"].replace("/", "_")
                ))

        # sort so that they are alphabetically ordered in wandb
        sort_index = np.argsort(captions)
        captions = np.array(captions)[sort_index].tolist()
        audios = np.array(audios)[sort_index].tolist()

        return list(zip(captions, audios))

    def get_piano_rolls(self, use_specific_samples_at=None):
        """ use_specific_samples_at: must be a dict of lists of (sample_ix) """

        self._sampled_hvos = self.get_hvo_samples_located_at(use_specific_samples_at)
        tab_titles = []
        piano_roll_tabs = []
        for subset_ix, tag in tqdm(enumerate(self._sampled_hvos.keys()),
                                   desc='Creating Piano rolls for ' + self.set_identifier,
                                   disable=self.disable_tqdm):
            piano_rolls = []
            for sample_hvo in self._sampled_hvos[tag]:
                title = "{}_{}_{}".format(
                    self.set_identifier, sample_hvo.metadata["style_primary"],
                    sample_hvo.metadata["master_id"].replace("/", "_"))
                piano_rolls.append(sample_hvo.to_html_plot(filename=title))
            piano_roll_tabs.append(separate_figues_by_tabs(piano_rolls, [str(x) for x in range(len(piano_rolls))]))
            tab_titles.append(tag)

        # sort so that they are alphabetically ordered in wandb
        sort_index = np.argsort(tab_titles)
        tab_titles = np.array(tab_titles)[sort_index].tolist()
        piano_roll_tabs = np.array(piano_roll_tabs)[sort_index].tolist()

        return separate_figues_by_tabs(piano_roll_tabs, [tag for tag in tab_titles])

    def get_features_statistics_dict(self):
        wandb_features_data = {}
        for major_key in self.global_features_dict.keys():
            for minor_key in self.global_features_dict[major_key].keys():
                feature_data = self.global_features_dict[major_key][minor_key]
                main_key = "{}.{}.".format(major_key.split('_AND_')[0].replace("['", ''),
                                           minor_key.split('_AND_')[0].replace("']", ''))
                wandb_features_data.update(
                    {
                        main_key + "mean": feature_data.mean(), main_key + "std": feature_data.std(),
                        main_key + "median": np.percentile(feature_data, 50),
                        main_key + "q1": np.percentile(feature_data, 25),
                        main_key + "q3": np.percentile(feature_data, 75)}
                )
        return wandb_features_data

    def get_logging_dict(self, velocity_heatmap_html=True, global_features_html=True,
                         piano_roll_html=True, audio_files=True, sf_paths=None,
                         for_audios_use_specific_samples_at=None, for_piano_rolls_use_specific_samples_at=None):

        if audio_files is True:
            assert sf_paths is not None, "Provide sound_file path(s) for synthesizing samples"

        logging_dict = {}
        if velocity_heatmap_html is True:
            logging_dict.update({"velocity_heatmaps": self.get_vel_heatmap_bokeh_figures()})
        if global_features_html is True:
            logging_dict.update({"global_feature_pdfs": self.get_global_features_bokeh_figure()})
        if audio_files is True:
            captions_audios_tuples = self.get_audios(sf_paths, for_audios_use_specific_samples_at)
            captions_audios = [(c_a[0], c_a[1]) for c_a in captions_audios_tuples]
            logging_dict.update({"captions_audios": captions_audios})
        if piano_roll_html is True:
            logging_dict.update({"piano_rolls": self.get_piano_rolls(for_piano_rolls_use_specific_samples_at)})

        return logging_dict

    def get_wandb_logging_media(self, velocity_heatmap_html=True, global_features_html=True,
                                piano_roll_html=True, audio_files=True, sf_paths=None,
                                for_audios_use_specific_samples_at=None, for_piano_rolls_use_specific_samples_at=None):

        logging_dict = self.get_logging_dict(velocity_heatmap_html, global_features_html,
                                             piano_roll_html, audio_files, sf_paths,
                                             for_audios_use_specific_samples_at,
                                             for_piano_rolls_use_specific_samples_at)

        wandb_media_dict = {}
        for key in logging_dict.keys():
            if velocity_heatmap_html is True and key in "velocity_heatmaps":
                wandb_media_dict.update(
                    {
                        "velocity_heatmaps":
                            {
                                self.set_identifier:
                                    wandb.Html(file_html(
                                        logging_dict["velocity_heatmaps"], CDN, "vel_heatmap_" + self.set_identifier))
                            }
                    }
                )

            if global_features_html is True and key in "global_feature_pdfs":
                wandb_media_dict.update(
                    {
                        "global_feature_pdfs":
                            {
                                self.set_identifier:
                                    wandb.Html(file_html(
                                        logging_dict["global_feature_pdfs"], CDN,
                                        "feature_pdfs_" + self.set_identifier))
                            }
                    }
                )

            if audio_files is True and key in "captions_audios":
                captions_audios_tuples = logging_dict["captions_audios"]
                wandb_media_dict.update(
                    {
                        "audios":
                            {
                                self.set_identifier:
                                    [
                                        wandb.Audio(c_a[1], caption=c_a[0], sample_rate=16000)
                                        for c_a in captions_audios_tuples
                                    ]
                            }
                    }
                )

            if piano_roll_html is True and key in "piano_rolls":
                wandb_media_dict.update(
                    {
                        "piano_roll_html":
                            {
                                self.set_identifier:
                                    wandb.Html(file_html(
                                        logging_dict["piano_rolls"], CDN, "piano_rolls_" + self.set_identifier))
                            }
                    }
                )

        return wandb_media_dict

    def dump(self, path=None, fname="subset_evaluator"):  # todo implement in comparator
        if path is None:
            path = os.path.join("misc", self.set_identifier)
        if not os.path.exists(path):
            os.makedirs(path)

        ofile = bz2.BZ2File(os.path.join(path, f"{fname}.SubEval.bz2"), 'wb')
        pickle.dump(self, ofile)
        ofile.close()


if __name__ == "__main__":
    print("TEST ---- TEST")