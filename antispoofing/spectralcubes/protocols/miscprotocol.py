# -*- coding: utf-8 -*-

import os
import numpy as np

from antispoofing.spectralcubes.utils import *
from antispoofing.spectralcubes.lowlevelfeatures import LowLevelFeatures
from antispoofing.spectralcubes.protocols.protocol import Protocol


class MISCProtocol(Protocol):

    """docstring for AnalizeProtocol"""

    def __init__(self, data, sample):
        self.data = data
        self.sample = sample
        self.llf_trials = None
        self.mlf_trials = None

    def build_search_space(self):
        """ docstring """

        llf_search_space = {'NTV': 300, 'LGF': 1, 'M': 7}
        mlf_search_space = {'CS': 'kmeans', 'SDD': 'class_based',
                            'DS': 320, 'CP': 'softmax'}

        mlf_search_space.update(llf_search_space)

        self.llf_trials = [llf_search_space]
        self.mlf_trials = [mlf_search_space]

    def extract_low_level_features(self):
        """ docstring """

        start = get_time()

        np.random.seed(7)

        fnames = []

        if self.data.__class__.__name__.lower() == 'replayattack':

            fnames = self.data.metainfo['all_fnames']
            fnames = [fname for fname in fnames if (self.sample in fname)and(('hand/' in fname)or('real' in fname))]

        elif self.data.__class__.__name__.lower() == 'casia':
            fnames = self.data.metainfo['all_fnames']

        elif self.data.__class__.__name__.lower() == 'maskattack':
            fnames = self.data.metainfo['all_fnames']

        elif self.data.__class__.__name__.lower() == 'uvad':
            fnames = self.data.metainfo['all_fnames']

        else:
            pass

        tasks = []
        for realization in [1]:
            sd = int(np.random.rand(1, 1)[0, 0] * 1e9)
            for fname in fnames:
                for trial in self.llf_trials:
                    path = 'realization_{0}/low_level_features/{1}/{2}/{3}'.format(realization, trial["NTV"],
                                                                                   rois_available[trial["LGF"]],
                                                                                   measures_available[trial["M"]])

                    output_path = os.path.join(self.data.output_path, path)

                    tasks += [LowLevelFeatures(self.data.dataset_path, output_path, fname, trial["LGF"], trial["M"],
                                               n_cuboids=trial["NTV"],
                                               analize=False,
                                               frame_numbers=0,
                                               seed=sd).run()]

        total_time_elapsed(start, get_time())

    def extract_mid_level_features(self):
        """ docstring """
        print 'It isn\'t necessary in this protocol'
        print 'Bye'

    def classification(self):
        """ docstring """
        print 'It isn\'t necessary in this protocol'
        print 'Bye'
