import numpy as np
from tqdm import tqdm
from eval.GrooveEvaluator.src.settings import FEATURES_TO_EXTRACT

from scipy.ndimage.filters import gaussian_filter

import warnings
warnings.filterwarnings("ignore")


class Feature_Extractor_From_HVO_SubSets:
    def __init__(
            self,
            hvo_subsets,
            tags=None,
            features_to_extract=FEATURES_TO_EXTRACT,
            auto_extract=False,
            subsets_from_gmd=True,      # todo change this later on to incorporate other datasets
                                        # used for getting number of unique performances in each set
            max_samples_in_subset=None,
    ):
        """
        Extracts features for each sample within a list of lists containing hvo_sequences
        each interior list corresponds to a hvo subset
        the tags should have the same length as hvo_datasets
        (compatible with HVO_Sequence Version >= 0.1.0)

        :param hvo_subsets:                 List of List of HVO_Sequence Sets [[Set 1 HVO_Seqs], [Set 2 HVO_Seqs]]
        :param tags:                        tags corresponding to subsets
        :param features_to_extract:         Dictionary of features (for which extractors are implemented in hvo_sequence)
                                            Must be formatted same way as GrooveEvaluator.settings.FEATURES_TO_EXTRACT
        :param auto_extract:                If True, features are extracted as soon as the class is instantiated
        """

        if tags is not None:
            assert len(tags) == len(hvo_subsets), "the length tags MUST be the same as the length of hvo_subsets"

        # Ignore empty sets
        self.tags = []
        self.hvo_subsets = []

        for ix, (tag, hvo_subset) in enumerate(zip(tags, hvo_subsets)):
            if len(hvo_subset) >= 1:
                if max_samples_in_subset is None:
                    # Grab all samples
                    self.hvo_subsets.append(hvo_subset)
                else:
                    max_samples = min(max_samples_in_subset, len(hvo_subset))
                    # Randomly grab max_samples_in_subset samples
                    self.hvo_subsets.append(np.random.choice(hvo_subset, max_samples, replace=False))

                self.tags.append(tag)

        self.features_to_extract = features_to_extract

        # each extracted_features_dict is a dictionary with same structure as FEATURES_TO_EXTRACT dict
        # the extracted features will be stored here upon calling the extract() method
        # will be formatted as dict{"tag0": {features for subset 1}, "tag1": {features for subset 2}}
        self.Feature_Extractors_for_each_subset = [
            Feature_Extractor_From_HVO_Set(
                self.hvo_subsets[ix], name=tags[ix]
            ) for ix in range(len(self.hvo_subsets))
        ]

        if auto_extract is True:
            for Feature_Extractor_for_subset in self.Feature_Extractors_for_each_subset:
                Feature_Extractor_for_subset.extract()

        # Get the number of loops in each set
        self.number_of_loops_in_sets = {}
        self.number_of_loops_in_sets.update(
            {
                set_name: len(hvo_subset) for set_name, hvo_subset in zip(self.tags, self.hvo_subsets)
            }
        )

        self.number_of_unique_performances_in_sets = None
        # Get the number of unique performances in each dataset if gmd dataset
        self.number_of_unique_performances_in_sets = {}
        if subsets_from_gmd is True:
            self.number_of_unique_performances_in_sets.update(
                {
                    set_name: len(
                        list(set(([hvo_seq.metadata["master_id"] for hvo_seq in hvo_subset])))
                    ) for set_name, hvo_subset in zip(self.tags, self.hvo_subsets)
                }
            )

    def extract(self, use_tqdm=True, force_extract=False):
        # Extract Features
        for Feature_Extractor_for_each_subset in self.Feature_Extractors_for_each_subset:
            Feature_Extractor_for_each_subset.extract(use_tqdm, force_extract)

    def get_global_features_dicts(self, use_tqdm=True, force_extract=False, regroup_by_feature=True):
        # If extract() hasn't been called previously, do so to extract features
        # If features extracted, No need to do so anymore
        self.extract(use_tqdm, force_extract)

        features_dicts_per_subset_list = []
        feature_keys = []

        for FE_for_SubSet in self.Feature_Extractors_for_each_subset:
            feature_dict = FE_for_SubSet.extracted_features_dict
            features_dicts_per_subset_list.append(feature_dict)
            for feature_key in feature_dict.keys():
                if feature_key not in feature_keys:
                    feature_keys.append(feature_key)

        if regroup_by_feature is True:
            feature_dicts_grouped = {
                feature_key: {tag: []} for feature_key in feature_keys for tag in self.tags}
            for feature_key in feature_keys:
                for tag_ix, tag in enumerate(self.tags):
                    feature_dicts_grouped[feature_key][tag] = features_dicts_per_subset_list[tag_ix][feature_key]
        else:
            feature_dicts_grouped = {
                tag: {feature_key: []} for tag in self.tags for feature_key in feature_keys}
            for tag_ix, tag in enumerate(self.tags):
                for feature_key in feature_keys:
                    feature_dicts_grouped[tag][feature_key] = features_dicts_per_subset_list[tag_ix][
                        feature_key]

        return feature_dicts_grouped

    def get_scatter_velocities_and_timings_dict(self, regroup_by_drum_voice=True):
        # Gets velocity profiles for each subset
        # returns
        velocity_profiles = []
        voice_keys = []
        for FE_for_SubSet in self.Feature_Extractors_for_each_subset:
            # each velocity_profile is a dictionary where keys are the drum voice ("kick" "snare" ...)
            # and the corresponding value is a tuple of (time_array, velocity_array)
            velocity_profile = FE_for_SubSet.extract_velocity_profile()
            velocity_profiles.append(velocity_profile)
            for voice_key in velocity_profile.keys():
                if voice_key not in voice_keys:
                    voice_keys.append(voice_key)

        if regroup_by_drum_voice is True:
            velocity_profiles_dict = {
                voice_key: {tag: []} for voice_key in voice_keys for tag in self.tags}
            for voice_key in voice_keys:
                for tag_ix, tag in enumerate(self.tags):
                    velocity_profiles_dict[voice_key][tag] = velocity_profiles[tag_ix][voice_key]
        else:
            velocity_profiles_dict = {
                tag: {voice_key: []} for tag in self.tags for voice_key in voice_keys}
            for tag_ix, tag in enumerate(self.tags):
                for voice_key in voice_keys:
                    velocity_profiles_dict[tag][voice_key] = velocity_profiles[tag_ix][voice_key]

        return velocity_profiles_dict

    def get_velocity_timing_heatmap_dicts(self, regroup_by_drum_voice=True, s=8, bins=None):

        if bins is None:
            bins = [32*10, 127]

        scatter_velocities_and_timings_dict = self.get_scatter_velocities_and_timings_dict(
            regroup_by_drum_voice=regroup_by_drum_voice)

        heatmaps_dict = {
            major_key: {
                minor_key: []
                        }
            for major_key in scatter_velocities_and_timings_dict.keys()
            for minor_key in scatter_velocities_and_timings_dict[major_key].keys()
        }

        for major_key in scatter_velocities_and_timings_dict.keys():
            for minor_key in scatter_velocities_and_timings_dict[major_key].keys():
                (times, vels) = scatter_velocities_and_timings_dict[major_key][minor_key]
                heatmap, xedges, yedges = np.histogram2d(times, vels, bins=bins)
                # check this for parameter selection
                # https://stackoverflow.com/questions/25216382/gaussian-filter-in-scipy
                heatmap = gaussian_filter(heatmap, sigma=s)     # , truncate=t)
                extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
                heatmaps_dict[major_key][minor_key] = (heatmap.T, extent)

        return heatmaps_dict, scatter_velocities_and_timings_dict

    def get_few_hvo_samples(self, num_samples):
        n_sets = len(self.hvo_subsets)
        sample_hvos = list()
        corresponding_subset_indices, corresponding_sample_indices_in_subsets = list(), list()

        for ix in range(num_samples):
            # Grab a subset randomly (uniformly possible regardless of number of samples within)
            random_subset_ix = np.random.randint(0, n_sets)
            corresponding_subset_indices.append(random_subset_ix)
            # grab a random sample from the selected subset
            random_sample_ix = np.random.randint(0, len(self.hvo_subsets[random_subset_ix]))
            corresponding_sample_indices_in_subsets.append(random_sample_ix)
            sample_hvos.append(self.hvo_subsets[random_subset_ix][random_sample_ix])

        return sample_hvos, corresponding_subset_indices, corresponding_sample_indices_in_subsets


class Feature_Extractor_From_HVO_Set:

    def __init__(
            self,
            hvo_dataset,
            features_to_extract=FEATURES_TO_EXTRACT,
            name=None
    ):
        """
        Extracts features for each sample within a list containing hvo_sequences
        (compatible with HVO_Sequence Version >= 0.1.0)

        :param hvo_dataset:                 List of HVO_Sequences
        :param features_to_extract:         Dictionary of features (for which extractors are implemented in hvo_sequence)
                                            Must be formatted same way as GrooveEvaluator.settings.FEATURES_TO_EXTRACT
        :param name:                        Name of dataset
        """

        self.__name = name

        # Add the dataset to self
        self.hvo_dataset = hvo_dataset

        self.features_to_extract = features_to_extract

        # __extracted_features_dict is a dictionary with same structure as FEATURES_TO_EXTRACT dict
        # the extracted features will be stored here upon calling the extract() method
        self.__extracted_features_dict = None
        self.__velocity_profile = None

    @property
    def name(self):
        return self.__name

    @name.setter
    def name(self, name_):
        self.__name = name_

    @property
    def extracted_features_dict(self):
        # If extract() hasn't been called previously, do so to extract features
        # If features extracted, No need to do so anymore
        return self.__extracted_features_dict

    @property
    def velocity_profile(self):
        # returns (and extracts if not done already) the velocity profile
        if self.__velocity_profile is None:
            self.__velocity_profile = self.extract_velocity_profile()
        return self.__velocity_profile

    def extract(self, use_tqdm=True, force_extract=False):

        should_extract = [False]
        if self.extracted_features_dict is None:
            should_extract.append(True)
            self.__extracted_features_dict = self.get_empty_extracted_features_dict_dict(self.features_to_extract)

        if force_extract is True:
            should_extract.append(True)

        if any(should_extract):
            for ix in tqdm(range(len(self.hvo_dataset)), desc="Extracting Features from {}".format(self.name),
                           disable=(not use_tqdm)):
                sample_hvo = self.hvo_dataset[ix]
                self.update_statistical_features(sample_hvo)        # 593.00it/s
                self.update_syncopation_features(sample_hvo)         # 167.00it/s
                self.update_autocorrelation_features(sample_hvo)        # 1320.64it/s
                self.update_microtiming_features(sample_hvo)            # 6.11it/s
                                                                        # 3675.27it/s for swingness
                                                                        # 15.00 it/s for laidbackness
                                                                        # improved 15.00 it/s to 4101 for Accuracy

    def extract_pdf(self, use_tqdm=True, force_extract=False):
        self.extract(use_tqdm=use_tqdm, force_extract=force_extract)

    def extract_velocity_profile(self):

        drum_voices = self.hvo_dataset[0].drum_mapping

        # create a dictionary where keys are the drum voice ("kick" "snare" ...)
        # and the corresponding value is a tuple of (time_array, velocity_array)
        velocity_profile_dict = {
            voice_key: (np.array([]), np.array([])) for voice_key in drum_voices
        }

        # Compile all the velocities together
        for hvo_seq in self.hvo_dataset:
            v = hvo_seq.get("v")
            o = hvo_seq.get("o")
            t = np.multiply(np.arange(o.shape[0]).reshape(-1, 1), np.ones((1, o.shape[1])))
            t = t + o
            for voice_ix, drum_voice in enumerate(drum_voices):
                non_zero_indices = np.argwhere(v[:, voice_ix] > 0)
                if non_zero_indices.size > 0:
                    velocity_profile_dict[drum_voice] = (
                        np.append(velocity_profile_dict[drum_voice][0], t[non_zero_indices, voice_ix]),  # timing
                        np.append(velocity_profile_dict[drum_voice][1], 127 * v[non_zero_indices, voice_ix]),
                    )

        for voice_ix, drum_voice in enumerate(drum_voices):
            sorted_time_indices = np.argsort(velocity_profile_dict[drum_voice][0])
            velocity_profile_dict[drum_voice] = (
                velocity_profile_dict[drum_voice][0][sorted_time_indices],
                velocity_profile_dict[drum_voice][1][sorted_time_indices]
            )

        return velocity_profile_dict

    def update_statistical_features(self, sample_hvo):

        statistical_keys = self.__extracted_features_dict.keys()

        if "Statistical::NoI" in statistical_keys:
            self.__extracted_features_dict["Statistical::NoI"] = np.append(
                self.__extracted_features_dict["Statistical::NoI"],
                sample_hvo.get_number_of_active_voices()
            )

        if "Statistical::Total Step Density" in statistical_keys:
            self.__extracted_features_dict["Statistical::Total Step Density"] = np.append(
                self.__extracted_features_dict["Statistical::Total Step Density"],
                sample_hvo.get_total_step_density()
            )

        if "Statistical::Avg Voice Density" in statistical_keys:
            self.__extracted_features_dict["Statistical::Avg Voice Density"] = np.append(
                self.__extracted_features_dict["Statistical::Avg Voice Density"],
                sample_hvo.get_average_voice_density()
            )

        if any(x in statistical_keys for x in ["Statistical::Lowness", "Statistical::Midness", "Statistical::Hiness"]):
            lowness, midness, hiness = sample_hvo.get_lowness_midness_hiness()
            if "Statistical::Lowness" in statistical_keys:
                self.__extracted_features_dict["Statistical::Lowness"] = np.append(
                    self.__extracted_features_dict["Statistical::Lowness"],
                    lowness
                )
            if "Statistical::Midness" in statistical_keys:
                self.__extracted_features_dict["Statistical::Midness"] = np.append(
                    self.__extracted_features_dict["Statistical::Midness"],
                    midness
                )
            if "Statistical::Hiness" in statistical_keys:
                self.__extracted_features_dict["Statistical::Hiness"] = np.append(
                    self.__extracted_features_dict["Statistical::Hiness"],
                    hiness
                )

        if "Statistical::Vel Similarity Score" in statistical_keys:
            self.__extracted_features_dict["Statistical::Vel Similarity Score"] = np.append(
                self.__extracted_features_dict["Statistical::Vel Similarity Score"],
                sample_hvo.get_velocity_score_symmetry()
            )

        if "Statistical::Weak to Strong Ratio" in statistical_keys:
            self.__extracted_features_dict["Statistical::Weak to Strong Ratio"] = np.append(
                self.__extracted_features_dict["Statistical::Weak to Strong Ratio"],
                sample_hvo.get_total_weak_to_strong_ratio()
            )

        if any(x in statistical_keys for x in ["Statistical::Poly Velocity Mean", "Statistical::Poly Velocity std"]):
            mean, std = sample_hvo.get_polyphonic_velocity_mean_stdev()
            if "Statistical::Poly Velocity Mean" in statistical_keys:
                self.__extracted_features_dict["Statistical::Poly Velocity Mean"] = np.append(
                    self.__extracted_features_dict["Statistical::Poly Velocity Mean"],
                    mean
                )
            if "Statistical::Poly Velocity std" in statistical_keys:
                self.__extracted_features_dict["Statistical::Poly Velocity std"] = np.append(
                    self.__extracted_features_dict["Statistical::Poly Velocity std"],
                    std
                )

        if any(x in statistical_keys for x in ["Statistical::Poly Offset Mean", "Statistical::Poly Offset std"]):
            mean, std = sample_hvo.get_polyphonic_offset_mean_stdev()
            if "Statistical::Poly Offset Mean" in statistical_keys:
                self.__extracted_features_dict["Statistical::Poly Offset Mean"] = np.append(
                    self.__extracted_features_dict["Statistical::Poly Offset Mean"],
                    mean
                )
            if "Statistical::Poly Offset std" in statistical_keys:
                self.__extracted_features_dict["Statistical::Poly Offset std"] = np.append(
                    self.__extracted_features_dict["Statistical::Poly Offset std"],
                    std
                )

    def update_syncopation_features(self, sample_hvo):
        sync_keys = self.__extracted_features_dict.keys()

        if "Syncopation::Combined" in sync_keys:
            self.__extracted_features_dict["Syncopation::Combined"] = np.append(
                self.__extracted_features_dict["Syncopation::Combined"],
                sample_hvo.get_combined_syncopation()
            )

        if "Syncopation::Polyphonic" in sync_keys:
            self.__extracted_features_dict["Syncopation::Polyphonic"] = np.append(
                self.__extracted_features_dict["Syncopation::Polyphonic"],
                sample_hvo.get_witek_polyphonic_syncopation()
            )

        if any(shared_feats in sync_keys for shared_feats in ["Syncopation::Lowsync", "Syncopation::Midsync",
                                                              "Syncopation::Hisync","Syncopation::Lowsyness",
                                                              "Syncopation::Midsyness", "Syncopation::Hisyness"]):

            lmh_sync_info = sample_hvo.get_low_mid_hi_syncopation_info()

            for feat in ["Syncopation::Lowsync", "Syncopation::Midsync", "Syncopation::Hisync", "Syncopation::Lowsyness",
                         "Syncopation::Midsyness", "Syncopation::Hisyness"]:
                if feat.split("::")[-1].lower() in lmh_sync_info.keys():
                    self.__extracted_features_dict[feat] = np.append(
                        self.__extracted_features_dict[feat],
                        lmh_sync_info[feat.split("::")[-1].lower()]
                    )

        if "Syncopation::Complexity" in sync_keys:
            self.__extracted_features_dict["Syncopation::Complexity"] = np.append(
                self.__extracted_features_dict["Syncopation::Complexity"],
                sample_hvo.get_total_complexity()
            )

    def update_autocorrelation_features(self, sample_hvo):
        autocorrelation_keys = self.__extracted_features_dict.keys()

        if any(shared_feats in autocorrelation_keys for shared_feats in [
            "Auto-Correlation::Skewness", "Auto-Correlation::Max",
            "Auto-Correlation::Centroid", "Auto-Correlation::Harmonicity"]
               ):
            autocorrelation_features = sample_hvo.get_velocity_autocorrelation_features()

            for feat in ["Auto-Correlation::Skewness", "Auto-Correlation::Max",
                         "Auto-Correlation::Centroid", "Auto-Correlation::Harmonicity"]:
                self.__extracted_features_dict[feat] = np.append(
                        self.__extracted_features_dict[feat],
                        autocorrelation_features[feat.split("::")[-1].lower()]
                )

    def update_microtiming_features(self, sample_hvo):

        if "Micro-Timing::Swingness" in self.__extracted_features_dict.keys():
            self.__extracted_features_dict["Micro-Timing::Swingness"] = np.append(
                self.__extracted_features_dict["Micro-Timing::Swingness"],
                sample_hvo.swingness()
            )

        if "Micro-Timing::Laidbackness" in self.__extracted_features_dict.keys():
            self.__extracted_features_dict["Micro-Timing::Laidbackness"] = np.append(
                self.__extracted_features_dict["Micro-Timing::Laidbackness"],
                sample_hvo.laidbackness()
            )

        if "Micro-Timing::Accuracy" in self.__extracted_features_dict.keys():
            self.__extracted_features_dict["Micro-Timing::Accuracy"] = np.append(
                self.__extracted_features_dict["Micro-Timing::Accuracy"],
                sample_hvo.get_timing_accuracy()
            )

    def get_empty_extracted_features_dict_dict(self, _features_to_extract):
        '''
        creates an empty dictionary for
        :param _features_to_extract:
        :return:
        '''
        # Create a new dictionary with same structure as _features_to_extract
        # Initialize to an empty np array and remove non required features

        extracted_features_dict = {}

        for type_key in _features_to_extract.keys():
            for feat_key in _features_to_extract[type_key].keys():
                if _features_to_extract[type_key][feat_key] is True:
                    extracted_features_dict.update({"{}::{}".format(type_key, feat_key) :  np.array([], dtype=np.float16) })

        return extracted_features_dict

    def get_major_minor_field_keys(self):
        # returns top level and specific feature levels
        # exp:  major   ["statistical", "micro-timing"]
        #       minor   ["noi", "lowness", "swing"]
        major_keys = list()
        minor_keys = list()

        for major_key in self.features_to_extract.keys():
            major_keys.append(major_key)
            for minor_key in self.features_to_extract[major_key].keys():
                minor_keys.append(minor_key)

        return list(set(major_keys)), list(set(minor_keys))


class Intraset_Distance_Calculator:
    def __init__(self, feature_dictionary, name=None):

        self.feature_dictionary = feature_dictionary
        self.name = name
        self.__intraset_distances_per_feat = None

    @property
    def intraset_distances_per_feat(self):
        if self.__intraset_distances_per_feat is None:
            self.calculate_distances()
        return self.__intraset_distances_per_feat

    def calculate_distances(self):

        # For each feature distance matrix will be a symmetrical MxM matrix with zeros on diagonal
        for feature, feature_values in zip(self.feature_dictionary.keys(), self.feature_dictionary.values()):
            n_samples = len(feature_values)
            distance_matrix_for_feature = np.zeros((n_samples, n_samples))
            for sample_ix, feature_value in enumerate(feature_values):
                dist_from_val_in_feature_set = np.linalg.norm(feature_values - feature_value)
                distance_matrix_for_feature[sample_ix,:] = dist_from_val_in_feature_set

            if self.__intraset_distances_per_feat is None:
                self.__intraset_distances_per_feat = {feature: distance_matrix_for_feature}
            else:
                self.__intraset_distances_per_feat.update({feature: distance_matrix_for_feature})

        return self.__intraset_distances_per_feat


class Interset_Distance_Calculator:
    def __init__(self, feature_dictionary_a, feature_dictionary_b, name_a=None, name_b=None):

        self.feature_dictionary_a = feature_dictionary_a
        self.name_a = name_a
        self.feature_dictionary_b = feature_dictionary_b
        self.name_b = name_b

        self.__inter_distances_per_feat = None

    @property
    def interset_distances_per_feat(self):
        if self.__inter_distances_per_feat is None:
            self.calculate_distances()
        return self.__inter_distances_per_feat

    def calculate_distances(self):

        # For each feature distance matrix will be a symmetrical MxM matrix with zeros on diagonal
        for feature, feature_values_a in zip(self.feature_dictionary_a.keys(), self.feature_dictionary_a.values()):
            if feature in self.feature_dictionary_b.keys():
                # Get the values in second set corresponding to feature
                feature_values_b = self.feature_dictionary_b[feature]

                # Calculate number of samples in both sets
                n_samples_a = len(feature_values_a)
                n_samples_b = len(feature_values_b)

                # Create an empty distance matrix of shape n_samples_a, n_samples_b
                distance_matrix_for_feature = np.zeros((n_samples_a, n_samples_b))

                # Fill in the distance matrix
                for sample_ix_a, feature_value_a in enumerate(feature_values_a):
                    dist_from_val_in_feature_set = np.linalg.norm(feature_values_b - feature_value_a)
                    distance_matrix_for_feature[sample_ix_a, :] = dist_from_val_in_feature_set

                if self.__inter_distances_per_feat is None:
                    self.__inter_distances_per_feat = {feature: distance_matrix_for_feature}
                else:
                    self.__inter_distances_per_feat.update({feature: distance_matrix_for_feature})

        return self.__inter_distances_per_feat


def convert_distances_dict_to_pdf_histograms_dict (distances_dict):
    distances_histograms = dict()
    for feature in distances_dict.keys():
        # Find number of bins using Scott's Rule of Thumb
        # https://en.wikipedia.org/wiki/Histogram#Scott's_normal_reference_rule
        distances_for_feat = distances_dict[feature]
        if np.count_nonzero(distances_for_feat) > 0:
            hist, bin_edges = np.histogram(
                distances_for_feat, bins="scott",
                density=True
            )
            distances_histograms.update({feature: (hist, bin_edges)})
    return distances_histograms


class Distance_to_PDF_Converter:
    def __init__(self, intraset_distances_per_features, interset_distances_per_features):
        self.intraset_distances_per_features = intraset_distances_per_features
        self.interset_distances_per_features = interset_distances_per_features

        self.__intraset_pdfs_per_features = None
        self.__interset_pdfs_per_features = None

    @property
    def intraset_pdfs_per_features(self):
        if self.__intraset_pdfs_per_features is None:
            self.calculate_pdfs()
        return self.__intraset_pdfs_per_features

    @property
    def interset_pdfs_per_features(self):
        if self.__interset_pdfs_per_features is None:
            self.calculate_pdfs()
        return self.__interset_pdfs_per_features

    def calculate_pdfs(self):
        # Calculate pdfs for interset distances per each feature
        self.__intraset_pdfs_per_features = convert_distances_dict_to_pdf_histograms_dict(
            self.intraset_distances_per_features)
        self.__interset_pdfs_per_features = convert_distances_dict_to_pdf_histograms_dict(
            self.interset_distances_per_features)
        return self.__intraset_pdfs_per_features, self.__interset_pdfs_per_features
