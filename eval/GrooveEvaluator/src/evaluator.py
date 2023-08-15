import sys

sys.path.insert(1, "../../../")
sys.path.insert(1, "../../")
import numpy as np
import bz2
import wandb
from eval.GrooveEvaluator.src.feature_extractor import Feature_Extractor_From_HVO_SubSets
from eval.GrooveEvaluator.src.plotting_utils import global_features_plotter, velocity_timing_heatmaps_scatter_plotter
from eval.GrooveEvaluator.src.plotting_utils import separate_figues_by_tabs, tabulated_violin_plot
from eval.GrooveEvaluator.utilities import subsetters

from bokeh.embed import file_html
from bokeh.resources import CDN
import pickle
import os
from tqdm import tqdm
import copy
import pandas as pd
from scipy.io.wavfile import write

from bokeh.models.widgets import Panel, Tabs
from bokeh.io import save

from eval.MultiSetEvaluator import MultiSetEvaluator
from copy import deepcopy


def flatten(t):
    if len(t) >=1:
        if isinstance(t[0], list):
            return [item for sublist in t for item in sublist]
        else:
            return t


def get_stats_from_samples_dict(feature_value_dict, trim_decimals=None):
    stats = []  # list of lists stats[i] corresponds to [mean, std, min, max, median, q1, q3]
    labels = []

    for key in feature_value_dict.keys():
        # Compile all genre data together
        data = []
        if isinstance(feature_value_dict[key], dict):
            for key2 in feature_value_dict[key].keys():
                data.extend(feature_value_dict[key][key2])
        else:
            data = flatten(feature_value_dict[key])

        # Calc stats
        stats.append(
            [np.mean(data), np.std(data), np.min(data), np.max(data), np.percentile(data, 50),
             np.percentile(data, 25),
             np.percentile(data, 75)])
        labels.append(key)

    # trim dataframe values to have trim_decimals decimal places
    if trim_decimals is not None:
        stats = [[f"{x:.{trim_decimals}f}" if x is not None else x for x in data] for data in stats]

    df2 = pd.DataFrame(np.array(stats).transpose(),
                       ["mean", "std", "min", "max", "median", "q1", "q3"],
                       labels)

    if trim_decimals is not None:
        df2 = df2.round(trim_decimals)

    return df2


# ======================================================================================================================
#  Evaluator Class
# ======================================================================================================================
class Evaluator:

    def __init__(
            self,
            hvo_sequences_list_,
            list_of_filter_dicts_for_subsets=None,
            _identifier="Train",
            n_samples_to_use=-1,
            max_hvo_shape=(32, 27),
            need_hit_scores=True,
            need_velocity_distributions=True,
            need_offset_distributions=True,
            need_rhythmic_distances=True,
            need_heatmap=True,
            need_global_features=True,
            need_piano_roll=True,
            need_audio=True,
            need_kl_oa=True,
            n_samples_to_synthesize_and_draw="all",

            disable_tqdm=False,
            # todo distance measures KL, Overlap, intra and interset
    ):
        """
        This class will perform a thorough Intra- and Inter- evaluation between ground truth data and predictions

        :param hvo_sequences_list_: A 1D list of HVO_Sequence objects corresponding to ground truth data
        :param list_of_filter_dicts_for_subsets: (Default: None, means use all data without subsetting) The filter dictionaries using which the dataset will be subsetted into different groups. Note that the HVO_Sequence objects must contain `metadata` attributes with the keys specified in the filter dictionaries.
        :param _identifier: A string label to identify the set of HVO_Sequence objects. This is used to name the output files.
        :param n_samples_to_use: (Default: -1, means use all data) The number of samples to use for evaluation in case you don't want to use all the samples. THese are randomly selected.
                 (it is recommended to use the entirety of the dataset, if smaller subset is needed, process them externally prior to Evaluator initialization)
        :param max_hvo_shape: (Default: (32, 27)) The maximum shape of the HVO array. This is used to trim/pad the HVO arrays to the same shape.
        :param need_heatmap: (Default: True) Whether to generate velocity timing heatmaps
        :param need_global_features: (Default: True) Whether to generate global features plots
        :param need_piano_roll: (Default: True) Whether to generate piano roll plots
        :param need_audio: (Default: True) Whether to generate audio files
        :param n_samples_to_synthesize: (Default: "all") The number of samples to synthesize audio files for. If "all", all samples will be synthesized.
        :param n_samples_to_draw_pianorolls: (Default: "all") The number of samples to draw piano rolls for. If "all", all samples will be drawn.
        :param disable_tqdm: (Default: False) Whether to disable tqdm progress bars
        """

        self.need_hit_scores = need_hit_scores
        self.need_velocity_distributions = need_velocity_distributions
        self.need_offset_distributions = need_offset_distributions
        self.need_rhythmic_distances = need_rhythmic_distances
        self.need_heatmap, self.need_global_features = need_heatmap, need_global_features
        self.need_piano_roll, self.need_audio = need_piano_roll, need_audio
        self.need_kl_oa = need_kl_oa

        self.disable_tqdm = disable_tqdm
        self.num_voices = int(max_hvo_shape[-1] / 3)

        hvo_sequences_list = deepcopy(hvo_sequences_list_)

        n_samples_to_use = len(hvo_sequences_list) if n_samples_to_use == -1 else n_samples_to_use

        # Create subsets of data
        gt_subsetter_sampler = subsetters.SubsetterAndSampler(
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

        self._gt_hvos_array = np.stack(self._gt_hvos_array) if len(self._gt_hvos_array) > 1\
            else self._gt_hvos_array

        # a text to identify the evaluator (exp "Train_Epoch1", "Test_Epoch1")
        self._identifier = _identifier

        # Subset evaluator for ground_truth data
        self.gt_SubSet_Evaluator = HVOSeq_SubSet_Evaluator(
            self._gt_subsets,  # Ground Truth typically
            self._gt_tags,
            "{}_Ground_Truth".format(self._identifier),  # a name for the subset
            disable_tqdm=self.disable_tqdm,
            group_by_minor_keys=True,
            need_heatmap=need_heatmap,
            need_global_features=need_global_features
        )

        # Empty place holder for predictions, also Placeholder Subset evaluator for predicted data
        self._prediction_tags, self._prediction_subsets = None, None
        self.prediction_SubSet_Evaluator = None
        self._prediction_hvos_array = None

        # Get the index for the samples that will be synthesized and also for which the piano-roll can be generated
        self.audio_sample_locations = self.get_sample_indices(n_samples_to_synthesize_and_draw)
        self.piano_roll_sample_locations = self.audio_sample_locations

        self._cached_gt_logging_data = None  # Cached data for ground truth logging
        self._cached_predicted_logging_data = None  # Cached data for predicted logging
        self._cached_gt_logging_data_wandb = None  # Cached data for ground truth logging
        self._cached_predicted_logging_data_wandb = None  # Cached data for predicted logging

    def __getstate__(self):

        state = {
            "need_hit_scores": self.need_hit_scores,
            "need_velocity_distributions": self.need_velocity_distributions,
            "need_offset_distributions": self.need_offset_distributions,
            "need_rhythmic_distances": self.need_rhythmic_distances,
            "need_heatmap": self.need_heatmap,
            "need_global_features": self.need_global_features,
            "need_piano_roll": self.need_piano_roll,
            "need_audio": self.need_audio,
            "need_kl_oa": self.need_kl_oa,
            "disable_tqdm": self.disable_tqdm,
            "num_voices": self.num_voices,
            "_gt_tags": self._gt_tags,
            "_gt_subsets": self._gt_subsets,
            "_identifier": self._identifier,
            "audio_sample_locations": self.audio_sample_locations,
            "piano_roll_sample_locations": self.piano_roll_sample_locations,
            "_prediction_hvos_array": self._prediction_hvos_array
        }

        return state

    def __setstate__(self, state):
        # reload the states
        self.need_hit_scores = state["need_hit_scores"]
        self.need_velocity_distributions = state["need_velocity_distributions"]
        self.need_offset_distributions = state["need_offset_distributions"]
        self.need_rhythmic_distances = state["need_rhythmic_distances"]
        self.need_heatmap = state["need_heatmap"]
        self.need_global_features = state["need_global_features"]
        self.need_piano_roll = state["need_piano_roll"]
        self.need_audio = state["need_audio"]
        self.need_kl_oa = state["need_kl_oa"]
        self.disable_tqdm = state["disable_tqdm"]
        self.num_voices = state["num_voices"]
        self._gt_tags = state["_gt_tags"]
        self._gt_subsets = state["_gt_subsets"]
        self._identifier = state["_identifier"]
        self.audio_sample_locations = state["audio_sample_locations"]
        self.piano_roll_sample_locations = state["piano_roll_sample_locations"]
        self._prediction_hvos_array = state["_prediction_hvos_array"]

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

        self._gt_hvos_array = np.stack(self._gt_hvos_array) if len(self._gt_hvos_array) > 1 \
            else self._gt_hvos_array

        # Subset evaluator for ground_truth data
        self.gt_SubSet_Evaluator = HVOSeq_SubSet_Evaluator(
            self._gt_subsets,  # Ground Truth typically
            self._gt_tags,
            "{}_Ground_Truth".format(self._identifier),  # a name for the subset
            disable_tqdm=self.disable_tqdm,
            group_by_minor_keys=True,
            need_heatmap=self.need_heatmap,
            need_global_features=self.need_global_features
        )

        # Empty place holder for predictions, also Placeholder Subset evaluator for predicted data
        self._prediction_tags, self._prediction_subsets = None, None
        self.prediction_SubSet_Evaluator = None

        self._cached_gt_logging_data = None  # Cached data for ground truth logging
        self._cached_predicted_logging_data = None  # Cached data for predicted logging
        self._cached_gt_logging_data_wandb = None  # Cached data for ground truth logging
        self._cached_predicted_logging_data_wandb = None  # Cached data for predicted logging

        if self._prediction_hvos_array is not None:
            self.add_predictions(self._prediction_hvos_array)

    # ==================================================================================================================
    #  Get Logging dict or WandB Artifacts for ground truth data and/or predictions
    #  The logging dict contains:
    #   - Heatmap Bokeh Plots
    #   - Global Feature Bokeh Plots
    #   - Piano Roll Bokeh Plots
    #   - Audio Files synthesized from HVO_Sequences
    # ==================================================================================================================
    def get_logging_media(self, prepare_for_wandb=True, save_directory=None, **kwargs):
        logging_media = dict()

        # use default constructor values if a value is not provided
        need_hit_scores = kwargs["need_hit_scores"] if "need_hit_scores" in kwargs.keys() else self.need_hit_scores
        need_velocity_distributions = kwargs["need_velocity_distributions"] if "need_velocity_distributions" in kwargs.keys() else self.need_velocity_distributions
        need_offset_distributions = kwargs["need_offset_distributions"] if "need_offset_distributions" in kwargs.keys() else self.need_offset_distributions
        need_rhythmic_distances = kwargs["need_rhythmic_distances"] if "need_rhythmic_distances" in kwargs.keys() else self.need_rhythmic_distances
        need_heatmap = kwargs["need_heatmap"] if "need_heatmap" in kwargs.keys() else self.need_heatmap
        need_global_features = kwargs["need_global_features"] if "need_global_features" in kwargs.keys() else self.need_global_features
        need_piano_roll = kwargs["need_piano_roll"] if "need_piano_roll" in kwargs.keys() else self.need_piano_roll
        need_audio = kwargs["need_audio"] if "need_audio" in kwargs.keys() else self.need_audio
        need_kl_oa = kwargs["need_kl_oa"] if "need_kl_oa" in kwargs.keys() else self.need_kl_oa

        if need_hit_scores is True:
            print("Preparing Hit Score Plots for Logging")
            save_path = os.path.join(save_directory, f"Hit_Score_Plots_{self._identifier}.html") if save_directory is not None else None
            logging_media["hit_score_plots"] = \
                {self._identifier: self.get_pos_neg_hit_plots(prepare_for_wandb=prepare_for_wandb, save_path=save_path)}

        if need_velocity_distributions is True:
            print("Preparing Velocity Distribution Plots for Logging")
            save_path = os.path.join(
                save_directory, f"Velocity_Distribution_Plots_{self._identifier}.html") if save_directory is not None else None
            logging_media["velocity_distribution_plots"] = \
                {self._identifier:
                     self.get_velocity_distribution_plots(prepare_for_wandb=prepare_for_wandb, save_path=save_path)
                 }


        if need_offset_distributions is True:
            print("Preparing Offset Distribution Plots for Logging")
            save_path = os.path.join(
                save_directory, f"Offset_Distribution_Plots_{self._identifier}.html") if save_directory is not None else None
            logging_media["offset_distribution_plots"] = \
                {self._identifier:
                     self.get_offset_distribution_plots(prepare_for_wandb=prepare_for_wandb, save_path=save_path)
                 }

        if need_rhythmic_distances is True:
            print("Preparing Rhythmic Distance Plots for Logging")
            if self._prediction_hvos_array is None:
                raise Warning("Cannot compute rhythmic distances without predictions")
            else:
                save_path = os.path.join(
                    save_directory, f"Rhythmic_Distance_Plots_{self._identifier}.html") if save_directory is not None else None
                logging_media["rhythmic_distance_plots"] = \
                    {self._identifier:
                         self.get_rhythmic_distances_of_pred_to_gt_plot(
                             prepare_for_wandb=prepare_for_wandb, save_path=save_path)
                    }

        if need_heatmap is True:
            print("Preparing Heatmap Plots for Logging")
            save_path = os.path.join(save_directory, f"Heatmap_Plots_{self._identifier}.html") if save_directory is not None else None
            logging_media["heatmap_plots"] = \
                {self._identifier: self.get_velocity_heatmaps(prepare_for_wandb=prepare_for_wandb, save_path=save_path)}

        if need_global_features is True:
            print("Preparing Global Feature Plots for Logging")
            save_path = os.path.join(save_directory, f"Global_Feature_Plots_{self._identifier}.html") if save_directory is not None else None
            logging_media["global_feature_plots"] = \
                {self._identifier: self.get_global_features_plot(
                    prepare_for_wandb=prepare_for_wandb, save_path=save_path)}

        if need_piano_roll is True:
            save_path = os.path.join(save_directory, f"Piano_Roll_Plots_{self._identifier}.html") if save_directory is not None else None
            logging_media["piano_roll_plots"] = \
                {self._identifier: self.get_piano_rolls(prepare_for_wandb=prepare_for_wandb, save_path=save_path)}

        if need_audio is True:
            print("Preparing Audio Files for Logging")
            save_path = os.path.join(save_directory, f"Audio_Files_{self._identifier}.html") if save_directory is not None else None
            logging_media["audios"] = \
                {self._identifier: self.get_audio_tuples(prepare_for_wandb=prepare_for_wandb, save_directory=save_path)}

        if need_kl_oa is True:
            print("Preparing KL-OA Plots for Logging")
            save_path = os.path.join(save_directory, f"KL-OA_Plots_{self._identifier}.html") if save_directory is not None else None
            logging_media["kl_oa_plots"] = \
                {self._identifier: self.get_kl_oa_inter_intra_plots(
                    identifier=self._identifier, prepare_for_wandb=prepare_for_wandb, save_path=save_path)}

        return logging_media

    # ==================================================================================================================
    #  KL OA Inter/Intra Plots
    # ==================================================================================================================
    def get_kl_oa_inter_intra_plots(self, identifier=None, save_path=None, prepare_for_wandb=False):
        if self._prediction_hvos_array is None:
            raise Warning("Cannot compute KL OA Inter/Intra plots without predictions")
        identifier = self._identifier if identifier is None else identifier
        ms_evaluator = MultiSetEvaluator(groove_evaluator_sets={identifier: self})

        kl_oa_plot =  ms_evaluator.get_kl_oa_plots()

        # make sure save_path ends in html
        if save_path is not None:
            if not save_path.endswith(".html"):
                save_path += ".html"

            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            save(kl_oa_plot, save_path)

        return kl_oa_plot if prepare_for_wandb is False else wandb.Html(file_html(kl_oa_plot, CDN, f"KL_OA_Inter_Intra_{identifier}"))

    # ==================================================================================================================
    #  Export to Midi
    # ==================================================================================================================
    def export_to_midi(self, need_gt=False, need_pred=False, directory="misc"):
        '''

        :param need_gt: (default False) if True, will export the ground truth to midi
        :param need_pred: (default False) if True, will export the predictions to midi
        :param directory: (default "misc") the parent directory to save the midi files to
        :return:
        '''

        def subset_to_midi(subset_tag, subsets_tags, subsets, path):
            '''

            :param subset_tag: (str) label used for grouping the subsets
            :param subsets_tags: (list) list of tags for each subset
            :param subsets: (list) list of subsets
            :param path: directory to save the midi files
            :return:
            '''
            subset_path = os.path.join(path, subset_tag)
            metadata = dict()

            for subsetix, tag in enumerate(subsets_tags):
                # Reinitialize metadata
                tag_path = os.path.join(subset_path, tag)
                os.makedirs(tag_path, exist_ok=True)

                for ix, _hvo in enumerate(subsets[subsetix]):
                    # add metadata
                    if ix == 0:
                        metadata = {ix: _hvo.metadata}
                    else:
                        metadata.update({ix: _hvo.metadata})

                    # export midi
                    _hvo.save_hvo_to_midi(os.path.join(tag_path, f"{ix}.mid"))

                metadata_for_samples = pd.DataFrame(metadata).transpose()
                metadata_for_samples.to_csv(os.path.join(tag_path, "metadata.csv"))

        directory = os.path.join(directory, self._identifier)
        os.makedirs(directory, exist_ok=True)

        # extract and export subsets for gt
        if need_gt:
            gt_subsets = self._gt_subsets
            gt_subsets_tags = self._gt_tags
            gt_subset_tag = "gt"
            subset_to_midi(gt_subset_tag, gt_subsets_tags, gt_subsets, directory)

        # extract and export subsets for predictions
        if need_pred:
            prediction_subsets = self._prediction_subsets
            prediction_subsets_tags = self._prediction_tags
            prediction_subset_tag = "prediction"
            subset_to_midi(prediction_subset_tag, prediction_subsets_tags, prediction_subsets, directory)

    # ==================================================================================================================
    #  Export to Audio
    # ==================================================================================================================
    def get_audio_tuples(self, prepare_for_wandb=False, sf_path=None, save_directory=None, concatenate_gt_and_pred=True):
        """
        :param sf_path: path to soundfont
        :param save_directory: directory to save audio files
        :param concatenate_gt_and_pred: concatenate ground truth and prediction audio files
        :return: list of tuples of (filename, audio array)
        """

        if sf_path is None:
            sf_path = "hvo_sequence/soundfonts/Standard_Drum_Kit.sf2"
            if os.path.exists(sf_path) is False:
                sf_path = "../" + sf_path
                if os.path.exists(sf_path) is False:
                    sf_path = "../../" + sf_path
                else:
                    raise Exception("Soundfont not found. Please provide a valid path to a soundfont.")

        gt_tuples = self.gt_SubSet_Evaluator.get_audios([sf_path])
        pred_tuples = self.prediction_SubSet_Evaluator.get_audios([sf_path])

        compiled_tuples = []
        for i, _ in enumerate(gt_tuples):
            if concatenate_gt_and_pred is False:
                title = gt_tuples[i][0].replace(".wav", "").replace("Ground_Truth", "")
                compiled_tuples.append((title + "_gt.wav", gt_tuples[i][1]))
                if self._prediction_subsets is not None:
                    compiled_tuples.append((title + "_pred.wav", pred_tuples[i][1]))
            else:
                title = gt_tuples[i][0].replace(".wav", "").replace("Ground_Truth", "") + "_gt_silence_pred.wav"
                gt_audio = gt_tuples[i][1]
                pred_audio = pred_tuples[i][1]
                silence = np.zeros(44100)
                np.concatenate((gt_audio, silence, pred_audio), axis=0)
                compiled_tuples.append((title, np.concatenate((gt_tuples[i][1], np.zeros(44100), pred_tuples[i][1]))))

        if save_directory is not None:
            def save_audio(path, array, sr=44100):
                write(path, sr, array)

            os.makedirs(save_directory, exist_ok=True)

            for title, audio in compiled_tuples:
                save_audio(os.path.join(save_directory, title), audio)

        if prepare_for_wandb is True:
            wandb_audios = [wandb.Audio(c_a[1], caption=c_a[0], sample_rate=16000) for c_a in compiled_tuples]
            return wandb_audios
        else:
            return compiled_tuples

    # ==================================================================================================================
    #  Evaluation of Hits
    # ==================================================================================================================
    def get_pos_neg_hit_scores(self, hit_weight=1, return_as_pandas_df=False):
        hit_scores_dict = dict()

        Actual_P_array = []
        Total_predicted_array = []
        TP_array = []
        FP_array = []
        PPV_array = []
        FDR_array = []
        TPR_array = []
        FPR_array = []
        FP_over_N = []
        FN_over_P = []
        DICE = []
        Accuracy_array = []
        Precision_array = []
        Recall_array = []
        F1_Score_array = []

        n_samples = len(self._gt_hvos_array)
        prediction_hvos_array = self._prediction_hvos_array if self._prediction_hvos_array is not None else [None] * n_samples

        for (true_values, predictions) in zip(self._gt_hvos_array, prediction_hvos_array):
            true_values = np.array(flatten(true_values[:, :self.num_voices]))
            predictions = np.array(flatten(predictions[:, :self.num_voices])) if predictions is not None else None
            Actual_P = np.count_nonzero(true_values)
            Actual_N = true_values.size - Actual_P

            Actual_P_array.append(Actual_P)

            if predictions is not None:
                Total_predicted_array.append((predictions == 1).sum())
                TP = ((predictions == 1) & (true_values == 1)).sum()
                FP = ((predictions == 1) & (true_values == 0)).sum()
                FN = ((predictions == 0) & (true_values == 1)).sum()
                TN = ((predictions == 0) & (true_values == 0)).sum()

                # https://en.wikipedia.org/wiki/Precision_and_recall
                PPV_array.append(TP / (TP + FP) if (TP + FP) > 0 else 0)
                FDR_array.append(FP / (TP + FP) if (TP + FP) > 0 else 0)
                TPR_array.append(TP / Actual_P)
                FPR_array.append(FP / Actual_N)
                TP_array.append(TP)
                FP_array.append(FP)
                FP_over_N.append(FP / Actual_N)
                FN_over_P.append(FN / Actual_P)
                DICE.append(2 * TP / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 0)
                Accuracy_array.append((TP + TN) / (Actual_P + Actual_N))
                Precision_array.append(TP / (TP + FP) if (TP + FP) > 0 else 0)
                Recall_array.append(TP / Actual_P)
                F1_Score_array.append(2 * TP / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 0)

        if predictions is not None:
            hit_scores_dict.update({
                "Relative - DICE": DICE,
                "Relative - Accuracy": Accuracy_array,
                "Relative - Precision": Precision_array,
                "Relative - Recall": Recall_array,
                "Relative - F1_Score": F1_Score_array,
                "Relative - TPR": TPR_array,
                "Relative - FPR": FPR_array,
                "Relative - PPV": PPV_array,
                "Relative - FDR": FDR_array,
                "Relative - Ratio of Silences Predicted as Hits": FP_over_N,
                "Relative - Ratio of Hits Predicted as Silences": FN_over_P,
                "Hit Count - Ground Truth": Actual_P_array,
                "Hit Count - Total Predictions": Total_predicted_array,
                "Hit Count - True Predictions (Matching GMD)": TP_array,
                "Hit Count - False Predictions (Different from GMD)": FP_array,
            })
        else:
            hit_scores_dict.update({
                "Hit Count - Ground Truth": Actual_P_array,
            })

        if return_as_pandas_df:
            return pd.DataFrame(hit_scores_dict)
        else:
            return hit_scores_dict

    def get_statistics_of_pos_neg_hit_scores(self, hit_weight=1, csv_file=None, trim_decimals=3, return_as_pandas_df=False):
        # make sure that the csv file ends in .csv
        if csv_file is not None:
            if not csv_file.endswith(".csv"):
                csv_file = csv_file + ".csv"

        # make directories for file path
        if csv_file is not None:
            os.makedirs(os.path.dirname(csv_file), exist_ok=True)

        hit_scores_dict = self.get_pos_neg_hit_scores(hit_weight=hit_weight)
        df2 = get_stats_from_samples_dict(hit_scores_dict, trim_decimals=trim_decimals)

        if csv_file is not None:
            df2.to_csv(csv_file)

        if return_as_pandas_df:
            return df2
        else:
            return df2.to_dict()

    def get_pos_neg_hit_plots(self, save_path=None, prepare_for_wandb=False, kernel_bandwidth=0.05, plot_width=1200, plot_height=800):
        hit_scores_dict = self.get_pos_neg_hit_scores()

        p = tabulated_violin_plot(hit_scores_dict, save_path=save_path, kernel_bandwidth=kernel_bandwidth,
                                     width=plot_width, height=plot_height)
        return p if not prepare_for_wandb else wandb.Html(file_html(p, CDN, "Pos Neg Hit Scores"))

    # ==================================================================================================================
    #  Evaluation of Velocities
    # ==================================================================================================================
    def get_velocity_distributions(self, return_as_pandas_df=False):
        vel_all_Hits = np.array([])
        vel_TP = np.array([])
        vel_FP = np.array([])
        vel_actual_mean = np.array([])
        vel_actual_std = np.array([])
        vel_all_Hits_mean = np.array([])
        vel_all_Hits_std = np.array([])
        vel_TP_mean = np.array([])
        vel_TP_std = np.array([])
        vel_FP_mean = np.array([])
        vel_FP_std = np.array([])

        n_samples = len(self._gt_hvos_array)
        prediction_hvos_array = self._prediction_hvos_array if self._prediction_hvos_array is not None \
            else [None] * n_samples

        for (true_values, predictions) in zip(self._gt_hvos_array, prediction_hvos_array):
            vel_actual_mean=np.append(vel_actual_mean, np.nanmean(true_values[:, self.num_voices: 2*self.num_voices][np.nonzero(true_values[:, :self.num_voices])]))
            vel_actual_std=np.append(vel_actual_std, np.nanstd(true_values[:, self.num_voices: 2*self.num_voices][np.nonzero(true_values[:, :self.num_voices])]))
            actual_hits = np.array(true_values[:, :self.num_voices]).flatten()


            if predictions is not None:
                vels_predicted = np.array(predictions[:, self.num_voices: 2 * self.num_voices]).flatten()
                predicted_hits = np.array(predictions[:, :self.num_voices]).flatten()
                all_predicted_hit_indices, = (predicted_hits==1).nonzero()
                vel_all_Hits = np.append(vel_all_Hits, vels_predicted[all_predicted_hit_indices])
                vel_all_Hits_mean = np.append(vel_all_Hits_mean, np.nanmean(vels_predicted[all_predicted_hit_indices]))
                vel_all_Hits_std = np.append(vel_all_Hits_std, np.nanstd(vels_predicted[all_predicted_hit_indices]))
                true_hit_indices, = np.logical_and(actual_hits==1, predicted_hits==1).nonzero()
                vel_TP = np.append(vel_TP, vels_predicted[true_hit_indices])
                vel_TP_mean = np.append(vel_TP_mean, np.nanmean(vels_predicted[true_hit_indices]))
                vel_TP_std = np.append(vel_TP_std, np.nanstd(vels_predicted[true_hit_indices]))
                false_hit_indices, = np.logical_and(actual_hits==0, predicted_hits==1).nonzero()
                vel_FP = np.append(vel_FP, vels_predicted[false_hit_indices])
                vel_FP_mean = np.append(vel_FP_mean, np.nanmean(vels_predicted[false_hit_indices]))
                vel_FP_std = np.append(vel_FP_std, np.nanstd(vels_predicted[false_hit_indices]))

        velocity_distributions = {
            "Means Per Sample - Total Ground Truth Hits": np.nan_to_num(vel_actual_mean),
            "STD Per Sample - Total Ground Truth Hits": np.nan_to_num(vel_actual_std)
        }

        if self._prediction_hvos_array is not None:
            velocity_distributions.update(
                {
                    "Means Per Sample - Total Predicted Hits": np.nan_to_num(vel_all_Hits_mean),
                    "Means Per Sample - True Predicted Hits": np.nan_to_num(vel_TP_mean),
                    "Means Per Sample - False Predicted Hits": np.nan_to_num(vel_FP_mean),
                    "STD Per Sample - Total Predicted Hits": np.nan_to_num(vel_all_Hits_std),
                    "STD Per Sample - True Predicted Hits": np.nan_to_num(vel_TP_std),
                    "STD Per Sample - False Predicted Hits": np.nan_to_num(vel_FP_std),
                }
            )

        if return_as_pandas_df:
            return pd.DataFrame(velocity_distributions)
        else:
            return velocity_distributions

    def get_statistics_of_velocity_distributions(self, csv_file=None, trim_decimals=3):
        # make sure that the csv file ends in .csv
        if csv_file is not None:
            if not csv_file.endswith(".csv"):
                csv_file = csv_file + ".csv"

        # make directories for file path
        if csv_file is not None:
            os.makedirs(os.path.dirname(csv_file), exist_ok=True)

        velocity_distributions = self.get_velocity_distributions()
        df2 = get_stats_from_samples_dict(velocity_distributions, trim_decimals=trim_decimals)

        if csv_file is not None:
            df2.to_csv(csv_file)

        return df2

    def get_velocity_distribution_plots(self, save_path=None, prepare_for_wandb=False,
                                        kernel_bandwidth=0.5, plot_width=800, plot_height=400):
        velocity_distributions = self.get_velocity_distributions()
        p = tabulated_violin_plot(velocity_distributions, save_path=save_path, kernel_bandwidth=kernel_bandwidth,
                                  width=plot_width, height=plot_height)
        return p if not prepare_for_wandb else wandb.Html(file_html(p, CDN, "Velocity Distribution"))

    def get_velocity_MSE(self, ignore_correct_silences=True):
        if self._prediction_hvos_array is None:
            print("Can't calculate velocity MSE as no predictions were given. Returning None.")
            return None
        non_silence_indices = (self._gt_hvos_array[:, :, :self.num_voices] + self._prediction_hvos_array[:, :, :self.num_voices]) > 0
        gt_vels = self._gt_hvos_array[:, :, self.num_voices:2 * self.num_voices]
        pred_vels = self._prediction_hvos_array[:, :, self.num_voices:2 * self.num_voices]
        n_examples = gt_vels.shape[0]
        if ignore_correct_silences:
            gt = gt_vels[non_silence_indices]
            pred = pred_vels[non_silence_indices]
        else:
            gt = gt_vels.reshape((n_examples, -1))
            pred = pred_vels.reshape((n_examples, -1))

        return ((gt - pred) ** 2).mean(axis=-1).mean()

    # ==================================================================================================================
    #  Evaluation of Microtiming
    # ==================================================================================================================
    def get_offset_distributions(self, return_as_pandas_df=False):

        offset_all_Hits = np.array([])
        offset_TP = np.array([])
        offset_FP = np.array([])
        offset_actual_mean = np.array([])
        offset_actual_std = np.array([])
        offset_all_Hits_mean = np.array([])
        offset_all_Hits_std = np.array([])
        offset_TP_mean = np.array([])
        offset_TP_std = np.array([])
        offset_FP_mean = np.array([])
        offset_FP_std = np.array([])

        n_samples = len(self._gt_hvos_array)
        prediction_hvos_array = self._prediction_hvos_array if self._prediction_hvos_array is not None \
            else [None] * n_samples

        for (true_values, predictions) in zip(self._gt_hvos_array, prediction_hvos_array):
            offset_actual_mean = np.append(offset_actual_mean, np.nanmean(
                true_values[:, 2 * self.num_voices:][np.nonzero(true_values[:, :self.num_voices])]))
            offset_actual_std = np.append(offset_actual_std, np.nanstd(
                true_values[:, 2 * self.num_voices:][np.nonzero(true_values[:, :self.num_voices])]))
            actual_hits = np.array(true_values[:, :self.num_voices]).flatten()

            if predictions is not None:
                offsets_predicted = np.array(predictions[:, 2 * self.num_voices:]).flatten()
                predicted_hits = np.array(predictions[:, :self.num_voices]).flatten()
                all_predicted_hit_indices, = (predicted_hits == 1).nonzero()
                offset_all_Hits = np.append(offset_all_Hits, offsets_predicted[all_predicted_hit_indices])
                offset_all_Hits_mean = np.append(offset_all_Hits_mean,
                                                 np.nanmean(offsets_predicted[all_predicted_hit_indices]))
                offset_all_Hits_std = np.append(offset_all_Hits_std,
                                                np.nanstd(offsets_predicted[all_predicted_hit_indices]))
                true_hit_indices, = np.logical_and(actual_hits == 1, predicted_hits == 1).nonzero()
                offset_TP = np.append(offset_TP, offsets_predicted[true_hit_indices])
                offset_TP_mean = np.append(offset_TP_mean, np.nanmean(offsets_predicted[true_hit_indices]))
                offset_TP_std = np.append(offset_TP_std, np.nanstd(offsets_predicted[true_hit_indices]))
                false_hit_indices, = np.logical_and(actual_hits == 0, predicted_hits == 1).nonzero()
                offset_FP = np.append(offset_FP, offsets_predicted[false_hit_indices])
                offset_FP_mean = np.append(offset_FP_mean, np.nanmean(offsets_predicted[false_hit_indices]))
                offset_FP_std = np.append(offset_FP_std, np.nanstd(offsets_predicted[false_hit_indices]))

        offset_distributions = {
            "Means Per Sample - Total Ground Truth Hits": np.nan_to_num(offset_actual_mean),
            "STD Per Sample - Total Ground Truth Hits": np.nan_to_num(offset_actual_std)
        }

        if self._prediction_hvos_array is not None:
            offset_distributions.update(
                {
                    "Means Per Sample - Total Predicted Hits": np.nan_to_num(offset_all_Hits_mean),
                    "Means Per Sample - True Predicted Hits": np.nan_to_num(offset_TP_mean),
                    "Means Per Sample - False Predicted Hits": np.nan_to_num(offset_FP_mean),
                    "STD Per Sample - Total Predicted Hits": np.nan_to_num(offset_all_Hits_std),
                    "STD Per Sample - True Predicted Hits": np.nan_to_num(offset_TP_std),
                    "STD Per Sample - False Predicted Hits": np.nan_to_num(offset_FP_std),
                }
            )

        if return_as_pandas_df:
            return pd.DataFrame(offset_distributions)
        else:
            return offset_distributions

    def get_statistics_of_offset_distributions(self, csv_file=None, trim_decimals=3):
        # make sure that the csv file ends in .csv
        if csv_file is not None:
            if not csv_file.endswith(".csv"):
                csv_file = csv_file + ".csv"

        # make directories for file path
        if csv_file is not None:
            os.makedirs(os.path.dirname(csv_file), exist_ok=True)

        offset_distributions = self.get_offset_distributions()
        df2 = get_stats_from_samples_dict(offset_distributions, trim_decimals=trim_decimals)

        if csv_file is not None:
            df2.to_csv(csv_file)

        return df2

    def get_offset_distribution_plots(self, save_path=None, prepare_for_wandb=False,
                                      kernel_bandwidth=0.5, plot_width=800, plot_height=400):
        offset_distributions = self.get_velocity_distributions()
        p = tabulated_violin_plot(offset_distributions, save_path=save_path, kernel_bandwidth=kernel_bandwidth,
                                     width=plot_width, height=plot_height)
        return p if not prepare_for_wandb else wandb.Html(file_html(p, CDN, "Offset Distribution"))

    def get_offset_MSE(self, ignore_correct_silences=True):
        if self._prediction_hvos_array is None:
            print("Can't calculate offset MSE as no predictions were given. Returning None.")
            return None

        non_silence_indices = (self._gt_hvos_array[:, :, :self.num_voices] + self._prediction_hvos_array[:, :,
                                                                             :self.num_voices]) > 0
        gt_offsets = self._gt_hvos_array[:, :, 2 * self.num_voices:]
        pred_offsets = self._prediction_hvos_array[:, :, 2 * self.num_voices:]
        n_examples = gt_offsets.shape[0]
        if ignore_correct_silences:
            gt = gt_offsets[non_silence_indices]
            pred = pred_offsets[non_silence_indices]
        else:
            gt = gt_offsets.reshape((n_examples, -1))
            pred = pred_offsets.reshape((n_examples, -1))

        return ((gt - pred) ** 2).mean(axis=-1).mean()

    # ==================================================================================================================
    #   Velocity Heatmaps
    # ==================================================================================================================
    def get_velocity_heatmaps(self, prepare_for_wandb=False, s=(2, 4), bins=None,
                              regroup_by_drum_voice=True, save_path=None):
        '''

        :param s: used for heatmap smoothing
        :param bins: [x_bins, y_bins] used for the heatmap bins (default [32 * 4, 64])
        :param regroup_by_drum_voice: if True, the heatmap will be grouped by drum voice, otherwise it will display voices in each figure
        :param save_path: if not None, the heatmap will be saved to this path
        :return: final bokeh figure with all the tabs
        '''

        bins = [32 * 4, 64] if bins is None else bins

        # get vel/timing heatmaps

        gt_heatmaps_dict, gt_scatter_velocities_and_timings_dict = \
            self.gt_SubSet_Evaluator.feature_extractor.get_velocity_timing_heatmap_dicts(
                s=s, bins=bins, regroup_by_drum_voice=regroup_by_drum_voice)
        gt_number_of_loops_per_subset_dict = self.gt_SubSet_Evaluator.feature_extractor.number_of_loops_in_sets
        gt_number_of_unique_performances_in_sets = self.gt_SubSet_Evaluator.feature_extractor.number_of_unique_performances_in_sets

        if self.prediction_SubSet_Evaluator is not None:
            pred_heatmaps_dict, pred_scatter_velocities_and_timings_dict = \
                self.prediction_SubSet_Evaluator.feature_extractor.get_velocity_timing_heatmap_dicts(
                    s=s, bins=bins, regroup_by_drum_voice=regroup_by_drum_voice)
            prediction_number_of_loops_per_subset_dict = self.prediction_SubSet_Evaluator.feature_extractor.number_of_loops_in_sets
            prediction_number_of_unique_performances_in_sets = self.prediction_SubSet_Evaluator.feature_extractor.number_of_unique_performances_in_sets

        gt_figs = velocity_timing_heatmaps_scatter_plotter(
            heatmaps_dict=gt_heatmaps_dict,
            scatters_dict=gt_scatter_velocities_and_timings_dict,
            number_of_loops_per_subset_dict=gt_number_of_loops_per_subset_dict,
            number_of_unique_performances_per_subset_dict=gt_number_of_unique_performances_in_sets,
            organized_by_drum_voice=regroup_by_drum_voice
        )

        prediction_figs = [None] * len(gt_figs)
        if self.prediction_SubSet_Evaluator is not None:
            prediction_figs = velocity_timing_heatmaps_scatter_plotter(
                heatmaps_dict=pred_heatmaps_dict,
                scatters_dict=pred_scatter_velocities_and_timings_dict,
                number_of_loops_per_subset_dict=prediction_number_of_loops_per_subset_dict,
                number_of_unique_performances_per_subset_dict=prediction_number_of_unique_performances_in_sets,
                organized_by_drum_voice=regroup_by_drum_voice
            )

        tabs = []
        for gt_fig, pred_fig in zip(gt_figs, prediction_figs):
            if pred_fig is not None:
                pred_fig.x_range = gt_fig.x_range
                pred_fig.y_range = gt_fig.y_range
                p2 = Panel(child=pred_fig, title="Prediction")
                p1 = Panel(child=gt_fig, title="Ground Truth")
            else:
                p1 = Panel(child=gt_fig, title=gt_fig.title.text)

            tabs.append(
                Panel(child=Tabs(tabs=[p1, p2]), title=gt_fig.title.text)) if pred_fig is not None else tabs.append(p1)

        final_fig = Tabs(tabs=tabs)

        if save_path is not None:
            # make sure save path ends with .html
            if not save_path.endswith(".html"):
                save_path += ".html"

            # make directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            save(final_fig, save_path)

        return final_fig if not prepare_for_wandb else wandb.Html(file_html(final_fig, CDN, "Velocity Heatmaps"))

    # ==================================================================================================================
    #  Evaluation using Global Features Implemented in HVO_Sequence
    # ==================================================================================================================
    def get_global_features_values(self, return_as_pandas_df=False):
        ''' Returns a dictionary with the global features values for each loop in the ground truth and predicted dataset (if available).
        :return: '''

        values_dict = {
            "Ground Truth": self.gt_SubSet_Evaluator.feature_extractor.get_global_features_dicts(True),
            "Predictions": self.prediction_SubSet_Evaluator.feature_extractor.get_global_features_dicts(True) if self._prediction_hvos_array is not None else None
        }

        if return_as_pandas_df:
            return pd.DataFrame(values_dict)
        else:
            return values_dict

    def get_statistics_of_global_features(self, calc_gt=True, calc_pred=True, csv_file=None, trim_decimals=3):
        '''
        Calculates the mean, median, Q1, Q3 and std of the global features of the ground truth and the prediction.
        :param calc_gt: If True, the statistics of the ground truth will be calculated.
        :param calc_pred: If True, the statistics of the prediction will be calculated.
        :param csv_file: If not None, the statistics will be saved to the given csv file.
        :param trim_decimals: If not None, The number of decimals to which the statistics will be trimmed.
        :return:
        '''
        # make sure that the csv file ends in .csv
        if csv_file is not None:
            if not csv_file.endswith(".csv"):
                csv_file = csv_file + ".csv"

        # make directories for file path
        if csv_file is not None:
            os.makedirs(os.path.dirname(csv_file), exist_ok=True)

        gt_df = get_stats_from_samples_dict(
            self.gt_SubSet_Evaluator.feature_extractor.get_global_features_dicts(True), trim_decimals=trim_decimals
        ) if (calc_gt and self.gt_SubSet_Evaluator is not None) else None

        pd_df = get_stats_from_samples_dict(
            self.prediction_SubSet_Evaluator.feature_extractor.get_global_features_dicts(True), trim_decimals=trim_decimals
        ) if (calc_pred and self.prediction_SubSet_Evaluator is not None) else None

        keys = []
        if gt_df is not None:
            keys.extend(gt_df.columns)
        if pd_df is not None:
            keys.extend(pd_df.columns)

        datas = []
        labels = []

        for key in keys:
            if gt_df is not None:
                data = gt_df.iloc[:][key].values if key in gt_df.columns else [None] * 7
            else:
                data = [None] * 7

            labels.append(key + "__Ground_Truth")
            datas.append(data)

            if pd_df is not None:
                data = pd_df.iloc[:][key].values if key in pd_df.columns else [None] * 7
            else:
                data = [None] * 7

            datas.append(data)
            labels.append(key + "__Prediction")

        df2 = pd.DataFrame(np.array(datas).transpose(),
                           ["mean", "std", "min", "max", "median", "q1", "q3"],
                           labels
                           )

        df2 = df2.loc[:, ~df2.columns.duplicated()]  # cols are duplicated

        if csv_file is not None:
            df2.to_csv(csv_file)

        return df2

    def get_global_features_plot(self, only_combined_data_needed=True, prepare_for_wandb=False,
                                 save_path=None, kernel_bandwidth=0.5, plot_width=800, plot_height=400):
        global_features = self.get_global_features_values()
        new_dict = {}
        for set_name, feature_dicts in global_features.items():
            if feature_dicts is not None:
                for feature_name, feature_dict in feature_dicts.items():
                    for genre, value in feature_dict.items():
                        if only_combined_data_needed is False:
                            G = genre.replace("[", "").replace("]", "")
                            new_dict[f"{feature_name.replace('-', ' ')} - {G} || {set_name} "] = value[~np.isnan(value)]

                        if f"{feature_name.replace('-', ' ')} - Combined || {set_name}" not in new_dict.keys():
                            new_dict[f"{feature_name.replace('-', ' ')} - Combined || {set_name}"] = value[~np.isnan(value)]
                        else:
                            new_dict[f"{feature_name.replace('-', ' ')} - Combined || {set_name}"] = \
                                np.append(new_dict[f"{feature_name.replace('-', ' ')} - Combined || {set_name}"], value[~np.isnan(value)])

        # sort dictionary by key
        new_dict = {k: new_dict[k] for k in sorted(new_dict.keys())}

        p = tabulated_violin_plot(new_dict, save_path=save_path, kernel_bandwidth=kernel_bandwidth,
                                  width=plot_width, height=plot_height, font_size=10)
        return p if prepare_for_wandb is False else wandb.Html(file_html(p, CDN, "Global Features"))

    # ==================================================================================================================
    #  Evaluation by comparing the rhythmic distances between the ground truth and the prediction
    #  (using multiple distance measures implemented in HVO_Sequence)
    # ==================================================================================================================
    def get_rhythmic_distances_of_pred_to_gt(self, return_as_pandas_df=False):
        '''
        returns a dictionary with the rhythmic distances between the ground truth and the prediction for each of the samples
        calculated
        :return:
        '''
        if self._prediction_hvos_array is None:
            print("Can't calculate rhythmic distances as no predictions were given. Returning None.")
            return None

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

        if return_as_pandas_df:
            return pd.DataFrame(distances_dict)
        else:
            return distances_dict

    def get_statistics_of_rhythmic_distances_of_pred_to_gt(self, tag_by_identifier=False, csv_dir=None, trim_decimals=None):
        '''
        Calculates the mean, median, Q1, Q3 and std of the rhythmic distances between the ground truth and the prediction.
        :param tag_by_identifier:
        :param csv_dir:
        :param trim_decimals:
        :return:
        '''

        if self._prediction_hvos_array is None:
            print("Can't calculate rhythmic distances as no predictions were given. Returning None.")
            return None

        distances_dict = self.get_rhythmic_distances_of_pred_to_gt()

        for key in distances_dict.keys():
            summary = {"mean": np.mean(distances_dict[key]),
                       "min": np.min(distances_dict[key]), "max": np.max(distances_dict[key]),
                       "median": np.percentile(distances_dict[key], 50),
                       "q1": np.percentile(distances_dict[key], 25),
                       "q3": np.percentile(distances_dict[key], 75)}

            if tag_by_identifier:
                distances_dict[key] = {self._identifier: summary}
            else:
                distances_dict[key] = summary

        # write dict to csv
        if csv_dir is not None:
            csv_dir = os.path.join(csv_dir, self._identifier)
            os.makedirs(csv_dir, exist_ok=True)

            if tag_by_identifier:
                for key in distances_dict.keys():
                    df = pd.DataFrame(distances_dict[key])
                    # round dataframe values to have 3 decimal places
                    if trim_decimals is not None:
                        df = df.round(trim_decimals)

                    df.to_csv(os.path.join(csv_dir, key + ".csv"))
                    return df

            else:
                df = pd.DataFrame(distances_dict)
                if trim_decimals is not None:
                    df = df.round(trim_decimals)
                df.to_csv(os.path.join(csv_dir, self._identifier + ".csv"))
                return df

    def get_rhythmic_distances_of_pred_to_gt_plot(self, save_path=None, prepare_for_wandb=False,
                                                  kernel_bandwidth=0.5, plot_width=800, plot_height=400):
        if self._prediction_hvos_array is None:
            print("Can't calculate rhythmic distances as no predictions were given. Returning None.")
            return None

        data_dict = self.get_rhythmic_distances_of_pred_to_gt()
        p = tabulated_violin_plot(data_dict, save_path=save_path, kernel_bandwidth=kernel_bandwidth,
                                  width=plot_width, height=plot_height)
        return p if prepare_for_wandb is False else wandb.Html(file_html(p, CDN, "Rhythmic Distances"))

    # ==================================================================================================================
    #  Get ground truth samples in HVO_Sequence format or as a numpy array
    # ==================================================================================================================
    def get_ground_truth_hvo_sequences(self):
        return copy.deepcopy(self._gt_hvo_sequences)

    def get_ground_truth_hvos_array(self):
        return copy.deepcopy(self._gt_hvos_array)

    # ==================================================================================================================
    #  Get Piano Roll HTML Figures
    def get_piano_rolls(self, save_path=None, prepare_for_wandb=False, x_range_pad=0.5, y_range_pad=None):
        gt = {k: v for k, v in sorted(zip(self._gt_tags, self._gt_subsets))}
        pred = {k: v for k, v in sorted(
            zip(self._prediction_tags, self._prediction_subsets))} if self._prediction_subsets is not None else None

        full_figure = []
        for k in gt.keys():
            subset_panels = []
            for sample_i in range(len(gt[k])):
                title = f"Sample {sample_i} - {gt[k][sample_i].metadata['master_id']}" if "master_id" in gt[k][
                    sample_i].metadata.keys() else "Sample {}".format(sample_i)
                sample_panels = list()
                gt_proll = gt[k][sample_i].to_html_plot(filename=title)
                gt_proll.x_range.range_padding = x_range_pad
                if y_range_pad is not None:
                    gt_proll.y_range.range_padding = y_range_pad
                sample_panels.append(Panel(child=gt_proll, title="GT"))
                if pred is not None:
                    pred_proll = pred[k][sample_i].to_html_plot()
                    pred_proll.x_range = gt_proll.x_range
                    pred_proll.y_range = gt_proll.y_range
                    sample_panels.append(Panel(child=pred_proll, title="Pred"))
                sample_tabs = Tabs(tabs=sample_panels)
                subset_panels.append(Panel(child=sample_tabs, title=title))
            full_figure.append(Panel(child=Tabs(tabs=subset_panels), title=k))

        final_tabs = Tabs(tabs=full_figure)

        if save_path is not None:
            # make sure file ends with .html
            if not save_path.endswith(".html"):
                save_path += ".html"

            # make sure directory exists
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path), exist_ok=True)

            save(final_tabs, save_path)

        return final_tabs if prepare_for_wandb is False else wandb.Html(file_html(final_tabs, CDN, "Piano Rolls"))

    # ==================================================================================================================
    #  Add predictions to the evaluator
    # ==================================================================================================================
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
            need_heatmap=self.need_heatmap,
            need_global_features=self.need_global_features
        )

    # ==================================================================================================================
    #  Save Evaluator
    # ==================================================================================================================
    def dump(self, path=None, fname="evaluator"):  # todo implement in comparator
        if path is None:
            path = os.path.join("misc")

        if not os.path.exists(path):
            os.makedirs(path)

        if not fname.endswith(".Eval.bz2"):
            fname += ".Eval.bz2"

        ofile = bz2.BZ2File(os.path.join(path, f"{self._identifier}_{fname}"), 'wb')
        pickle.dump(self, ofile)
        ofile.close()

        print(f"Dumped Evaluator to {os.path.join(path, f'{self._identifier}_{fname}')}")

    # ==================================================================================================================
    #  Utils
    # ==================================================================================================================
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


# ======================================================================================================================
#  HVOSeq_SubSet_Evaluator
# ======================================================================================================================

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
            need_heatmap=True,
            need_global_features=True
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
        :param need_heatmap:             True/False if velocity heatmap analysis is needed
        :param need_global_features:     True/False if global feature  analysis is needed
        """
        self.__version__ = "0.0.0"

        self.set_identifier = set_identifier

        self.disable_tqdm = disable_tqdm
        self.group_by_minor_keys = group_by_minor_keys

        self.feature_extractor = None  # instantiates in

        self.max_samples_in_subset = max_samples_in_subset
        self.need_heatmap = need_heatmap
        self.need_global_features = need_global_features
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

        if self.need_global_features:
            self.feature_extractor.extract(use_tqdm=not self.disable_tqdm)
            self.global_features_dict = self.feature_extractor.get_global_features_dicts(
                regroup_by_feature=self.group_by_minor_keys)

        if self.need_heatmap:
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
            number_of_loops_per_subset_dict=self.feature_extractor.number_of_loops_in_setsnumber_of_loops_in_sets,
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
                        desc='Synthesizing samples - {}'.format(self.set_identifier),
                        disable=self.disable_tqdm):
            for sample_idx, sample_hvo in enumerate(self._sampled_hvos[key]):
                # randomly select a sound font
                sf_path = sf_paths[np.random.randint(0, len(sf_paths))]
                audios.append(sample_hvo.synthesize(sf_path=sf_path))
                captions.append("subset {} sample {} _{}_{}_{}.wav".format(
                    key, sample_idx, self.set_identifier, sample_hvo.metadata["style_primary"],
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
            minor_tab_titles = []
            for idx, sample_hvo in enumerate(self._sampled_hvos[tag]):
                title = "subset {} sample {} _{}_{}_{}".format(
                    tag, idx, self.set_identifier, sample_hvo.metadata["style_primary"],
                    sample_hvo.metadata["master_id"].replace("/", "_"))
                piano_rolls.append(sample_hvo.to_html_plot(filename=title))
                minor_tab_titles.append(title)

            piano_roll_tabs.append(separate_figues_by_tabs(piano_rolls, tab_titles=minor_tab_titles))
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

    def get_logging_dict(self, need_velocity_heatmap=True, need_global_features=True,
                         need_piano_rolls=True, need_audio_files=True, sf_paths=None,
                         for_audios_use_specific_samples_at=None, for_piano_rolls_use_specific_samples_at=None):

        if need_audio_files is True:
            assert sf_paths is not None, "Provide sound_file path(s) for synthesizing samples"

        logging_dict = {}
        if need_velocity_heatmap is True:
            logging_dict.update({"velocity_heatmaps": self.get_vel_heatmap_bokeh_figures()})
        if need_global_features is True:
            logging_dict.update({"global_feature_pdfs": self.get_global_features_bokeh_figure()})
        if need_audio_files is True:
            captions_audios_tuples = self.get_audios(sf_paths, for_audios_use_specific_samples_at)
            captions_audios = [(c_a[0], c_a[1]) for c_a in captions_audios_tuples]
            logging_dict.update({"captions_audios": captions_audios})
        if need_piano_rolls is True:
            logging_dict.update({"piano_rolls": self.get_piano_rolls(for_piano_rolls_use_specific_samples_at)})

        return logging_dict

    def get_wandb_logging_media(self, need_velocity_heatmap=True, need_global_features=True,
                                need_piano_rolls=True, need_audio_files=True, sf_paths=None,
                                for_audios_use_specific_samples_at=None, for_piano_rolls_use_specific_samples_at=None):

        logging_dict = self.get_logging_dict(need_velocity_heatmap, need_global_features,
                                             need_piano_rolls, need_audio_files, sf_paths,
                                             for_audios_use_specific_samples_at,
                                             for_piano_rolls_use_specific_samples_at)

        wandb_media_dict = {}
        for key in logging_dict.keys():
            if need_velocity_heatmap is True and key in "velocity_heatmaps":
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

            if need_global_features is True and key in "global_feature_pdfs":
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

            if need_audio_files is True and key in "captions_audios":
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

            if need_piano_rolls is True and key in "piano_rolls":
                wandb_media_dict.update(
                    {
                        "need_piano_rolls":
                            {
                                self.set_identifier:
                                    wandb.Html(file_html(
                                        logging_dict["piano_rolls"], CDN, "piano_rolls_" + self.set_identifier))
                            }
                    }
                )

        return wandb_media_dict

    def dump(self, path=None, fname=""):
        if path is None:
            path = os.path.join("misc")

        if not os.path.exists(path):
            os.makedirs(path)

        fpath = os.path.join(path, f"{fname}_{self.set_identifier}.SubEval.bz2")
        ofile = bz2.BZ2File(fpath, 'wb')
        pickle.dump(self, ofile)
        ofile.close()

        print(f"Dumped {self.set_identifier}  evaluator to {fpath}")

        return fpath

def load_evaluator (full_path):
    ifile = bz2.BZ2File(full_path, 'rb')
    evaluator = pickle.load(ifile)
    ifile.close()
    return evaluator

if __name__ == "__main__":
    # For testing use demos/GrooveEvaluator/ .py files
    pass