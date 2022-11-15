import pickle
import os
import pandas as pd
import warnings

import sys
sys.path.append("../../..")
sys.path.append("../..")
sys.path.append("..")

from hvo_sequence.hvo_seq import HVO_Sequence                           # required for loading pickles
import note_seq
import math
import numpy as np
import bz2

from copy import deepcopy
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("eval/GrooveEvaluator/utilities/subsetters.py")


class SubsetterAndSampler(object):
    def __init__(
            self,
            hvo_sequences_list,
            list_of_filter_dicts_for_subsets=None,
            number_of_samples=1024,
            max_hvo_shape=(32, 27),
            at_least_one_hit_in_voices=None         # should be a list of voices where at least 1 hit is required
                                                    # example:  [0, 1, 2]
    ):
        tags_all, subsets_all = HVOSetSubsetter(
            hvo_sequences_list,
            list_of_filter_dicts_for_subsets=list_of_filter_dicts_for_subsets,
            max_len=max_hvo_shape[0],
            at_least_one_hit_in_voices=at_least_one_hit_in_voices).create_subsets()

        set_sampler = Set_Sampler(
            tags_all, subsets_all,
            number_of_samples=number_of_samples,
            max_hvo_shape=max_hvo_shape)

        self.sampled_tags, self.sampled_subsets = set_sampler.get_sampled_tags_subsets()

        self.hvos_array_tags, self.hvos_array, self.hvo_seq_templates = set_sampler.get_hvos_array()

    def get_subsets(self):
        return self.sampled_tags, self.sampled_subsets

    def get_hvos_array(self):
        return self.hvos_array_tags, self.hvos_array, self.hvo_seq_templates


class HVOSetSubsetter(object):
    def __init__(
            self,
            hvo_sequences_list,
            list_of_filter_dicts_for_subsets=None,
            max_len=None,
            at_least_one_hit_in_voices=None
    ):
        '''
        Uses a list of filter dictionaries to create subsets of the hvo_sequences_list

        :param hvo_sequences_list: a flat list of all hvo_sequences in the set
        :param list_of_filter_dicts_for_subsets: uses this to divide the dataset into subsets
        :param max_len: makes sure that all sequences in the subset are of length == max_len
        :param at_least_one_hit_in_voices: (default None: no requirement) A list of voice indices, where at least one hit is needed
                                            for example, if [0, 1, 2] there should be at least
                                            a kick, snare OR hat hit in each sample. In other words,
                                            if there is no hit in these voices, the sample is discarded
        '''
        self.list_of_filter_dicts_for_subsets = list_of_filter_dicts_for_subsets

        self.full_hvo_set_pre_filters = hvo_sequences_list

        self.at_least_one_hit_in_voices = at_least_one_hit_in_voices

        if max_len is not None:
            for hvo_seq in self.full_hvo_set_pre_filters:
                # pad with zeros or trim to match max_len
                pad_count = max(max_len - hvo_seq.hvo.shape[0], 0)
                hvo_seq.hvo = np.pad(hvo_seq.hvo, ((0, pad_count), (0, 0)), 'constant')
                hvo_seq.hvo = hvo_seq.hvo[:max_len, :]  # In case, sequence exceeds max_len

        self.subset_tags = None
        self.hvo_subsets = None

    def create_subsets(self, force_create=False):
        """
        Creates a set of subsets from a hvo_sequence dataset using a set of filters specified in constructor

        :param      force_create: If True, this method re-creates subsets even if it has already done so
        :return:    subset_tags, hvo_subsets
        """

        # Don't recreate if already haven't done so ( and if force_create is false)
        if self.subset_tags is not None and self.hvo_subsets is not None:
            if len(self.subset_tags) == len(self.hvo_subsets) and force_create is False:
                return self.subset_tags, self.hvo_subsets

        # if no filters, return a SINGLE dataset containing all hvo_seq sequences
        if self.list_of_filter_dicts_for_subsets is None or self.list_of_filter_dicts_for_subsets == [None]:
            hvo_subsets = [self.full_hvo_set_pre_filters]
            subset_tags = ['CompleteSet']

        else:
            hvo_subsets = []
            subset_tags = []
            for i in range(len(self.list_of_filter_dicts_for_subsets)):
                hvo_subsets.append([])
                subset_tags.append([''])

            for subset_ix, filter_dict_for_subset in enumerate(self.list_of_filter_dicts_for_subsets):
                # if current filter is None or a dict with None values
                # add all the dataset in its entirety to current subset
                if filter_dict_for_subset is None:
                    hvo_subsets[subset_ix] = self.full_hvo_set_pre_filters
                elif isinstance(filter_dict_for_subset, dict) and \
                        all(value is None for value in filter_dict_for_subset.values()):
                    hvo_subsets[subset_ix] = self.full_hvo_set_pre_filters

                else:
                    # Check which samples meet all filter specifications and add them to the current subset
                    subset_tags[subset_ix] = '_AND_'.join(str(x) for x in filter_dict_for_subset.values())
                    for hvo_sample in self.full_hvo_set_pre_filters:
                        if self.does_pass_filter(hvo_sample, filter_dict_for_subset):
                            if self.at_least_one_hit_in_voices is not None:
                                # Check that there is at least one hit in the required subset of voices
                                if 1 in hvo_sample.hvo[:, self.at_least_one_hit_in_voices]:
                                    hvo_subsets[subset_ix].append(hvo_sample)
                            else:
                                hvo_subsets[subset_ix].append(hvo_sample)


        return subset_tags, hvo_subsets

    def does_pass_filter(self, hvo_sample, filter_dict):   # FIXME THERE IS AN ISSUE HERE
        passed_conditions = [True]  # initialized with true in case no filters are required
        for fkey_, fval_ in filter_dict.items():
            if fval_ is not None:
                passed_conditions.append(True if hvo_sample.metadata[fkey_] in fval_ else False)

        return all(passed_conditions)


class Set_Sampler(object):

    def __init__(self, tags_, hvo_subsets_, number_of_samples = None, max_hvo_shape=(32, 27)):
        '''
        samples randomly from a set of hvo_sequence subsets
        :param tags_: list of tags for each subset
        :param hvo_subsets_:  list of hvo_sequences for each subset
        :param number_of_samples: (default None --> means all samples)
                                    number of samples *RANDOMLY* to take from each subset
        :param max_hvo_shape: (default (32, 27)) max shape of hvo_sequences to be returned.
                            If samples are longer, they will be trimmed to this length.
        '''
        tags = []
        hvo_subsets = []
        self.subsets_dict = {}

        total_samples = sum([len(x) for x in hvo_subsets_])
        if number_of_samples is None or number_of_samples > total_samples or number_of_samples <= 0:
            logger.warning('All the samples are being used for GrooveEvaluator initialization')
            number_of_samples = total_samples

        # delete empty sets
        for tag, hvo_subset in zip(tags_, hvo_subsets_):
            if hvo_subset:
                tags.append(tag)
                hvo_subsets.append(hvo_subset)

        # remove empty subsets
        self.hvos_array_tags = []
        self.hvos_array = np.zeros((number_of_samples, max_hvo_shape[0], max_hvo_shape[1]))
        self.hvo_seqs = []
        self.empty_hvo_seqs = []

        sample_count = 0
        while sample_count<number_of_samples:
            # Sample a subset
            subset_ix = int(np.random.choice(range(len(tags)), 1))
            tag = tags[subset_ix]

            # Sample an example if subset is not fully emptied out
            if hvo_subsets[subset_ix]:
                sample_ix = int(np.random.choice(range(len(hvo_subsets[subset_ix])), 1))
                hvo_seq = hvo_subsets[subset_ix][sample_ix]
                if tag not in self.subsets_dict.keys():
                    self.subsets_dict.update({tag: [deepcopy(hvo_seq)]})
                else:
                    self.subsets_dict[tag].append(hvo_seq)

                hvo = hvo_seq.get("hvo")
                max_time = min(max_hvo_shape[0], hvo.shape[0])

                self.hvos_array[sample_count, :max_time, :] = hvo
                self.hvos_array_tags.append(tag)
                self.hvo_seqs.append(hvo_seq)
                self.empty_hvo_seqs.append(hvo_seq.copy_empty())
                del (hvo_subsets[subset_ix][sample_ix])  # remove the sample from future selections

                sample_count += 1

        del tags
        del hvo_subsets

    def get_hvos_array(self):
        return self.hvos_array_tags, self.hvos_array, self.empty_hvo_seqs

    def get_sampled_tags_subsets(self):
        return list(self.subsets_dict.keys()), list(self.subsets_dict.values())


def convert_hvos_array_to_subsets(hvos_array_tags, hvos_array_predicted, hvo_seqs_templates_):
    hvo_seqs_templates = deepcopy(hvo_seqs_templates_)

    tags = list(set(hvos_array_tags))
    temp_dict = {tag: [] for tag in tags}

    for i in range(hvos_array_predicted.shape[0]):
        hvo_seqs_templates[i].hvo = hvos_array_predicted[i, :, :]
        temp_dict[hvos_array_tags[i]].append(hvo_seqs_templates[i])

    tags = list(temp_dict.keys())
    subsets = list(temp_dict.values())

    return tags, subsets
