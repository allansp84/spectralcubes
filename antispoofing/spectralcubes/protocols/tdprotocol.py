# -*- coding: utf-8 -*-

import os
import sys
import numpy as np

from antispoofing.spectralcubes.utils import *
from antispoofing.spectralcubes.lowlevelfeatures import LowLevelFeatures
from antispoofing.spectralcubes.midlevelfeatures import MidLevelFeatures
from antispoofing.spectralcubes.classification import Classification
from antispoofing.spectralcubes.protocols.protocol import Protocol


class TDProtocol(Protocol):
    """docstring for TDProtocol"""

    def __init__(self, data, frame_numbers=0, only_face=False):
        self.data = data
        self.llf_trials = None
        self.mlf_trials = None
        self.n_jobs = int(2 * N_JOBS / 4)
        self.frame_numbers = frame_numbers
        self.only_face = only_face

    def build_search_space(self):
        """ docstring """

        # llf_search_space = {'NTV': 1200, 'LGF': 0, 'M': 7}
        # mlf_search_space = {'CS': 'kmeans', 'SDD': 'class_based', 'DS': 160, 'CP': 'softmax'}

        llf_search_space = {'NTV': 1200, 'LGF': 1, 'M': 7}
        mlf_search_space = {'CS': 'kmeans', 'SDD': 'class_based', 'DS': 160, 'CP': 'softmax'}

        if self.data.__class__.__name__.lower() == 'uvad':
            llf_search_space = {'NTV': 1200, 'LGF': 1, 'M': 7}
            mlf_search_space = {'CS': 'random', 'SDD': 'class_based', 'DS': 320, 'CP': 'softmax'}

        mlf_search_space.update(llf_search_space)

        self.llf_trials = [llf_search_space]
        self.mlf_trials = [mlf_search_space]

    def extract_low_level_features(self):
        """ docstring """

        start = get_time()

        np.random.seed(7)

        fnames = self.data.metainfo['all_fnames']

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
                                           only_face=self.only_face,
                                           frame_numbers=self.frame_numbers,
                                           seed=sd,
                                           get_faceloc=self.data.get_faceloc)]

        if self.n_jobs > 1:
            print "running %d tasks in parallel" % len(tasks)
            RunInParallel(tasks, self.n_jobs).run()
        else:
            print "running %d tasks in sequence" % len(tasks)
            for idx in range(len(fnames)):
                tasks[idx].run()
                progressbar('-- RunInSequence', idx, len(fnames))

        elapsed = total_time_elapsed(start, get_time())
        print 'spent time: {0}!'.format(elapsed)
        sys.stdout.flush()

    def extract_mid_level_features(self):
        """ docstring """

        start = get_time()

        for trial in self.mlf_trials:
            path = 'low_level_features/{0}/{1}/{2}'.format(trial["NTV"],
                                                           rois_available[trial["LGF"]],
                                                           measures_available[trial["M"]])

            input_path = os.path.join(self.data.output_path, path)
            metainfo_feats = self.data.metainfo_feats(input_path, ['npy'])

            output_path = os.path.join(input_path, trial["CS"], trial["SDD"], str(trial["DS"]))
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

        elapsed = total_time_elapsed(start, get_time())
        print 'spent time: {0}!'.format(elapsed)
        sys.stdout.flush()

    def classification(self):
        """ docstring """

        start = get_time()

        algos = ["svm"]
        for trial in self.mlf_trials:
            for algo in algos:
                path = 'mid_level_features/{0}/{1}/{2}/{3}/{4}/{5}/{6}/'.format(trial["NTV"],
                                                                                rois_available[trial["LGF"]],
                                                                                measures_available[trial["M"]],
                                                                                trial["CS"], trial["SDD"], trial["DS"],
                                                                                trial["CP"])

                input_path = os.path.join(self.data.output_path, path)
                metainfo_feats = self.data.metainfo_feats(input_path, ['npy'])

                output_path = os.path.join(input_path, algo)
                output_path = output_path.replace('mid_level_features', 'classifiers')

                print "output_path", output_path

                Classification(output_path, metainfo_feats, algo=algo).td_protocol()

        elapsed = total_time_elapsed(start, get_time())
        print 'spent time: {0}!'.format(elapsed)
        sys.stdout.flush()

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
