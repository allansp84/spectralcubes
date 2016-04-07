# -*- coding: utf-8 -*-

import os
import sys
import numpy as np

from antispoofing.spectralcubes.utils import *
from antispoofing.spectralcubes.lowlevelfeatures import LowLevelFeatures
from antispoofing.spectralcubes.midlevelfeatures import MidLevelFeatures
from antispoofing.spectralcubes.classification import Classification
from antispoofing.spectralcubes.protocols.protocol import Protocol


class KFoldProtocol(Protocol):
    """docstring for KFoldProtocol"""

    def __init__(self, data, k_fold=10, frame_numbers=0):
        self.data = data
        self.llf_trials = None
        self.mlf_trials = None
        self.n_jobs = int(3 * N_JOBS / 4)
        self.k_fold = k_fold
        self.random_state = None
        self.frame_numbers = frame_numbers

    def build_search_space(self):
        """ docstring """

        if self.data.__class__.__name__.lower() == 'maskattack':

            llf_search_space = {'NTV': 1200, 'LGF': 1, 'M': 7}
            mlf_search_space = {'CS': 'kmeans', 'SDD': 'class_based', 'DS': 80, 'CP': 'softmax'}

        else:
            print 'This Protocol is not defined for this dataset!'
            sys.exit(0)

        mlf_search_space.update(llf_search_space)

        self.llf_trials = [llf_search_space]
        self.mlf_trials = [mlf_search_space]

    def extract_low_level_features(self):
        """ docstring """

        start = get_time()

        np.random.seed(7)

        if self.data.__class__.__name__.lower() == 'maskattack':
            fnames = self.data.metainfo['all_fnames']
        else:
            print 'This Protocol is not defined for this dataset!'
            sys.exit(0)

        tasks = []
        sd = int(np.random.rand(1, 1)[0, 0] * 1e9)
        for fname in fnames:
            for trial in self.llf_trials:
                path = 'low_level_features/{0}/{1}/{2}'.format(trial["NTV"],
                                                               rois_available[trial["LGF"]],
                                                               measures_available[trial["M"]])

                output_path = os.path.join(self.data.output_path, path)

                tasks += [LowLevelFeatures(self.data.dataset_path, output_path, fname,
                                           trial["LGF"],
                                           trial["M"],
                                           n_cuboids=trial["NTV"],
                                           frame_numbers=self.frame_numbers,
                                           seed=sd)]

        print "running %d tasks in parallel" % len(tasks)
        RunInParallel(tasks, self.n_jobs).run()

        total_time_elapsed(start, get_time())

    def extract_mid_level_features(self):
        """ docstring """

        start = get_time()

        self.random_state = np.random.RandomState(7)

        for k in xrange(self.k_fold):
            for trial in self.mlf_trials:
                path = 'low_level_features/{0}/{1}/{2}'.format(trial["NTV"],
                                                               rois_available[trial["LGF"]],
                                                               measures_available[trial["M"]])

                input_path = os.path.join(self.data.output_path, path)

                metainfo_feats = self.data.metainfo_feats(input_path, ['npy'], random_state=self.random_state)

                folds = "fold_{0}".format(k)
                output_path = os.path.join(self.data.output_path,
                                           folds,
                                           path,
                                           trial["CS"],
                                           trial["SDD"],
                                           str(trial["DS"]))

                output_path = output_path.replace('low_level_features', 'mid_level_features')

                print "{0}/{1}".format(output_path, trial["CP"])

                midlevelfeats = MidLevelFeatures(metainfo_feats, input_path, output_path,
                                                 codebook_selection=trial["CS"],
                                                 codebook_build=trial["SDD"],
                                                 codebook_size=trial["DS"],
                                                 coding_poling=trial["CP"],
                                                 n_jobs=self.n_jobs)
                midlevelfeats.build_codebook()
                midlevelfeats.run()

            total_time_elapsed(start, get_time())

    def classification(self):
        """ docstring """

        start = get_time()

        algos = ["svm"]

        for trial in self.mlf_trials:
            self.random_state = np.random.RandomState(7)
            for algo in algos:
                mean_hter = 0
                for k in xrange(self.k_fold):
                    folds = "fold_{0}".format(k)
                    path = '{0}/mid_level_features/{1}/{2}/{3}/{4}/{5}/{6}/{7}'.format(folds,
                                                                                       trial["NTV"], rois_available[trial["LGF"]],
                                                                                       measures_available[trial["M"]],
                                                                                       trial["CS"], trial["SDD"], trial["DS"],
                                                                                       trial["CP"])

                    input_path = os.path.join(self.data.output_path, path)
                    metainfo_feats = self.data.metainfo_feats(input_path, ['npy'],
                                                              random_state=self.random_state)

                    output_path = os.path.join(input_path, algo)
                    output_path = output_path.replace('mid_level_features', 'classifiers')

                    print "output_path", output_path

                    hter = Classification(output_path, metainfo_feats, algo=algo).tdt_protocol()
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
