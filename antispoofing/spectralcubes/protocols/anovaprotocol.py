# -*- coding: utf-8 -*-

import os
import itertools
import numpy as np

from antispoofing.spectralcubes.utils import *
from antispoofing.spectralcubes.classification import Classification
from antispoofing.spectralcubes.lowlevelfeatures import LowLevelFeatures
from antispoofing.spectralcubes.midlevelfeatures import MidLevelFeatures
from antispoofing.spectralcubes.protocols.protocol import Protocol


class ANOVAProtocol(Protocol):

    def __init__(self, data):
        self.data = data
        self.llf_trials = None
        self.dict_trials = None
        self.mlf_trials = None
        self.n_jobs = N_JOBS
        self.realizations = range(1, 4)

    def build_search_space(self):
        """ docstring """

        llf_search_space = {'NTV': [300],
                            'LGF': [key for key in rois_available],
                            'M': [key for key in measures_available],
                            }

        build_dicts_search_space = {'CS': ["kmeans", "random"],
                                    'SDD': ["class_based", "unified"],
                                    'DS': range(80, 361, 40),
                                    }

        mlf_search_space = {'CP': ["hardsum"]}

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

        np.random.seed(7)
        seeds = []
        for _ in self.realizations:
            seeds += [int(np.random.rand(1, 1)[0, 0] * 1e9)]

        start = get_time()

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

        # tasks = []
        # for realization in self.realizations:
        #     for trial in self.dict_trials:
        #         path = 'realization_{0}/low_level_features/{1}/{2}/{3}'.format(realization,
        #                     trial["NTV"], rois_available[trial["LGF"]],
        #                     measures_available[trial["M"]])
        #         input_path = os.path.join(self.data.output_path, path)

        #         metainfo_feats = self.data.metainfo_feats(input_path, ['npy'])

        #         output_path = os.path.join(input_path, trial["CS"], trial["SDD"], str(trial["DS"]))
        #         output_path = output_path.replace('low_level_features','mid_level_features')

        #         print output_path

        #         midlevelfeats = MidLevelFeatures(metainfo_feats, input_path, output_path,
        #                                          codebook_selection=trial["CS"],
        #                                          codebook_build=trial["SDD"],
        #                                          codebook_size=trial["DS"])

        #         midlevelfeats.build_codebook()

        for realization in self.realizations:
            for trial in self.mlf_trials:
                path = 'realization_{0}/low_level_features/{1}/{2}/{3}'.format(realization,
                                                                               trial["NTV"], rois_available[trial["LGF"]],
                                                                               measures_available[trial["M"]])
                input_path = os.path.join(self.data.output_path, path)

                metainfo_feats = self.data.metainfo_feats(input_path, ['npy'])

                output_path = os.path.join(input_path, trial["CS"], trial["SDD"], str(trial["DS"]))
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
                for algo in algos:
                    path = 'realization_{0}/mid_level_features/{1}/{2}/{3}/{4}/{5}/{6}/{7}'.format(realization,
                                                                                                   trial["NTV"],
                                                                                                   rois_available[trial["LGF"]],
                                                                                                   measures_available[trial["M"]],
                                                                                                   trial["CS"], trial["SDD"],
                                                                                                   trial["DS"], trial["CP"])

                    import pdb
                    pdb.set_trace()
                    input_path = os.path.join(self.data.output_path, path)
                    metainfo_feats = self.data.metainfo_feats(input_path, ['npy'])

                    output_path = os.path.join(input_path, algo)
                    output_path = output_path.replace('mid_level_features', 'classifiers')

                    print "output_path", output_path

                    Classification(output_path, metainfo_feats, algo=algo).anova_protocol()

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
