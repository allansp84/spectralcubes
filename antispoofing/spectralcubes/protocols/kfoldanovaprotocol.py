# -*- coding: utf-8 -*-

import os
import itertools
import numpy as np

from antispoofing.spectralcubes.utils import *
from antispoofing.spectralcubes.classification import Classification
from antispoofing.spectralcubes.lowlevelfeatures import LowLevelFeatures
from antispoofing.spectralcubes.midlevelfeatures import MidLevelFeatures
from antispoofing.spectralcubes.protocols.protocol import Protocol


class KFoldANOVAProtocol(Protocol):
    """docstring for ANOVAProtocol"""

    def __init__(self, data, k_fold=10):
        self.data = data
        self.k_fold = k_fold
        self.llf_trials = None
        self.dict_trials = None
        self.mlf_trials = None
        self.n_jobs = N_JOBS
        self.realizations = range(1, 2)
        self.random_state = None

    def build_search_space(self):
        """ docstring """

        llf_search_space = {'NTV': [1200],
                            'LGF': [1],  # [key for key in rois_available],
                            'M': [7],  # [key for key in measures_available],
                            }

        build_dicts_search_space = {'CS': ["kmeans"],
                                    'SDD': ["class_based", "unified"],
                                    'DS': range(80, 361, 40),
                                    }

        mlf_search_space = {'CP': ["softmax"]}

        build_dicts_search_space.update(llf_search_space)

        mlf_search_space.update(build_dicts_search_space)

        trials = []
        for values in itertools.product(*llf_search_space.values()):
            trial = [{key: value} for (key, value) in zip(llf_search_space, values)]
            trials += [dict(kv for d in trial for kv in d.iteritems())]
        self.llf_trials = trials

        trials = []
        for values in itertools.product(*build_dicts_search_space.values()):
            trial = [{key: value} for (key, value) in zip(build_dicts_search_space, values)]
            trials += [dict(kv for d in trial for kv in d.iteritems())]
        self.dict_trials = trials

        trials = []
        for values in itertools.product(*mlf_search_space.values()):
            trial = [{key: value} for (key, value) in zip(mlf_search_space, values)]
            trials += [dict(kv for d in trial for kv in d.iteritems())]
        self.mlf_trials = trials

    def extract_low_level_features(self):
        """ docstring """

        start = get_time()

        np.random.seed(7)

        seeds = []
        for _ in self.realizations:
            seeds += [int(np.random.rand(1, 1)[0, 0] * 1e9)]

        fnames = self.data.metainfo['all_fnames']

        tasks = []
        for realization, sd in zip(self.realizations, seeds):
            print "realization, sd", realization, sd
            for fname in fnames:
                for trial in self.llf_trials:
                    path = 'realization_{0}/low_level_features/{1}/{2}/{3}'.format(realization,
                                                                                   trial["NTV"], rois_available[trial["LGF"]],
                                                                                   measures_available[trial["M"]])

                    output_path = os.path.join(self.data.output_path, path)

                    tasks += [LowLevelFeatures(self.data.dataset_path, output_path, fname,
                                               trial["LGF"],
                                               trial["M"],
                                               n_cuboids=trial["NTV"],
                                               seed=sd)]

        print "running %d tasks in parallel" % len(tasks)
        RunInParallel(tasks, self.n_jobs).run()

        total_time_elapsed(start, get_time())

    def extract_mid_level_features(self):
        """ docstring """

        start = get_time()

        for realization in self.realizations:
            for trial in self.dict_trials:
                self.random_state = np.random.RandomState(7)
                for k in xrange(self.k_fold):
                    path = 'realization_{0}/low_level_features/{1}/{2}/{3}'.format(realization,
                                                                                   trial["NTV"], rois_available[trial["LGF"]],
                                                                                   measures_available[trial["M"]])
                    input_path = os.path.join(self.data.output_path, path)

                    metainfo_feats = self.data.metainfo_feats(input_path, ['npy'], random_state=self.random_state)

                    folds = "fold_{0}/low_level_features".format(k)
                    path = path.replace('low_level_features', folds)

                    output_path = os.path.join(self.data.output_path,
                                               path,
                                               trial["CS"],
                                               trial["SDD"],
                                               str(trial["DS"]))

                    output_path = output_path.replace('low_level_features', 'mid_level_features')

                    print output_path

                    midlevelfeats = MidLevelFeatures(metainfo_feats, input_path, output_path,
                                                     codebook_selection=trial["CS"],
                                                     codebook_build=trial["SDD"],
                                                     codebook_size=trial["DS"])

                    midlevelfeats.build_codebook()

        for realization in self.realizations:
            for trial in self.mlf_trials:

                self.random_state = np.random.RandomState(7)

                for k in xrange(self.k_fold):
                    path = 'realization_{0}/low_level_features/{1}/{2}/{3}'.format(
                        realization, trial["NTV"], rois_available[trial["LGF"]], measures_available[trial["M"]])

                    input_path = os.path.join(self.data.output_path, path)

                    metainfo_feats = self.data.metainfo_feats(input_path, ['npy'], random_state=self.random_state)

                    folds = "fold_{0}/low_level_features".format(k)
                    path = path.replace('low_level_features', folds)
                    output_path = os.path.join(self.data.output_path,
                                               path,
                                               trial["CS"],
                                               trial["SDD"],
                                               str(trial["DS"]))

                    output_path = output_path.replace('low_level_features', 'mid_level_features')

                    print output_path

                    midlevelfeats = MidLevelFeatures(metainfo_feats, input_path, output_path,
                                                     codebook_selection=trial["CS"],
                                                     codebook_build=trial["SDD"],
                                                     codebook_size=trial["DS"],
                                                     coding_poling=trial["CP"])

                    midlevelfeats.run()

        total_time_elapsed(start, get_time())

    def classification(self):
        """ docstring """

        start = get_time()

        algos = ["svm"]
        for realization in self.realizations:
            for trial in self.mlf_trials:
                self.random_state = np.random.RandomState(7)
                for algo in algos:
                    mean_hter = 0
                    for k in xrange(self.k_fold):
                        path = 'realization_{0}/fold_{1}/mid_level_features/{2}/{3}/{4}/{5}/{6}/{7}/{8}'.format(
                            realization, k, trial["NTV"], rois_available[trial["LGF"]], measures_available[trial["M"]],
                            trial["CS"], trial["SDD"], trial["DS"], trial["CP"])

                        input_path = os.path.join(self.data.output_path, path)
                        metainfo_feats = self.data.metainfo_feats(input_path, ['npy'], random_state=self.random_state)

                        output_path = os.path.join(input_path, algo)
                        output_path = output_path.replace('mid_level_features', 'classifiers')

                        print "output_path", output_path

                        hter = Classification(output_path, metainfo_feats, algo=algo).anova_protocol()
                        mean_hter += hter

                    mean_hter /= float(self.k_fold)
                    print '\tMEAN HTER = {0:.4f}'.format(mean_hter)

        total_time_elapsed(start, get_time())

    def execute_protocol(self):
        """ docstring """

        print "building search space ..."
        self.build_search_space()

        print "computing low level features ..."
        self.extract_low_level_features()

        print "computing mid level features ..."
        self.extract_mid_level_features()

        print "building classifiers ..."
        self.classification()
