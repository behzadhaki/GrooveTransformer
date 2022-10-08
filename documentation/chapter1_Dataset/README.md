## Chapter 1 - Dataset

----

List of content

# Table of Contents
1. [Example](#example)
2. [Example2](#example2)
3. [Third Example](#third-example)
4. [Fourth Example](#fourth-examplehttpwwwfourthexamplecom)


## Example

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
            gmd_pickle_path,
            list_of_filter_dicts_for_subsets,
            _identifier="Train",
            n_samples_to_use=1024,
            max_hvo_shape=(32, 27),
            n_samples_to_synthesize_visualize_per_subset=20,
            analyze_heatmap=True,
            analyze_global_features=True,
            disable_tqdm=True,
            # todo distance measures KL, Overlap, intra and interset
    ):
        """
        This class will perform a thorough Intra- and Inter- evaluation between ground truth data and predictions

        :param gmd_pickle_path:          "data/gmd/resources/cached/beat_division_factor_[4]/
                                          drum_mapping_label_['ROLAND_REDUCED_MAPPING']/
                                          beat_type_['beat']_time_signature_['4-4']/test.bz2pickle"
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
        self.disable_tqdm = disable_tqdm

        # Create subsets of data
        gt_subsetter_sampler = subsetters.GrooveMidiSubsetterAndSampler(
            gmd_pickle_path,
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
            group_by_minor_keys=True)

        # Empty place holder for predictions, also Placeholder Subset evaluator for predicted data
        self._prediction_tags, self._prediction_subsets = None, None
        self.prediction_SubSet_Evaluator = None
        self._prediction_hvos_array = None

        # Get the index for the samples that will be synthesized and also for which the piano-roll can be generated
        self.audio_sample_locations = self.get_sample_indices(n_samples_to_synthesize_visualize_per_subset)

        self._gt_logged_once = False  # Flag will be set to True when ground truth data is once evaluated
        self._gt_logged_once_wandb = False  # Flag will be set to True when ground truth data is once evaluated
        # for WANDB

    def get_logging_dict(self, velocity_heatmap_html=True, global_features_html=True,
                         piano_roll_html=True, audio_files=True,
                         sf_paths=["hvo_sequence/soundfonts/Standard_Drum_Kit.sf2"],
                         recalculate_ground_truth=True):

        _gt_logging_data = None
        # Get logging data for ground truth data
        if recalculate_ground_truth is True or self._gt_logged_once is False:
            _gt_logging_data = self.gt_SubSet_Evaluator.get_logging_dict(
                velocity_heatmap_html=velocity_heatmap_html,
                global_features_html=global_features_html,
                piano_roll_html=piano_roll_html,
                audio_files=audio_files,
                sf_paths=sf_paths,
                use_specific_samples_at=self.audio_sample_locations
            )

        _predicted_logging_data = self.prediction_SubSet_Evaluator.get_logging_dict(
            velocity_heatmap_html=velocity_heatmap_html,
            global_features_html=global_features_html,
            piano_roll_html=piano_roll_html,
            audio_files=audio_files,
            sf_paths=sf_paths,
            use_specific_samples_at=self.audio_sample_locations
        ) if self.prediction_SubSet_Evaluator is not None else None

        return _gt_logging_data, _predicted_logging_data

    def get_wandb_logging_media(self, velocity_heatmap_html=True, global_features_html=True,
                                piano_roll_html=True, audio_files=True,
                                sf_paths=["hvo_sequence/soundfonts/Standard_Drum_Kit.sf2"],
                                recalculate_ground_truth=True):

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
## Example2
## Third Example
## [Fourth Example](http://www.fourthexample.com) 