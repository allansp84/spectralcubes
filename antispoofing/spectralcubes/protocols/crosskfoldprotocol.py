# -*- coding: utf-8 -*-

import os
import sys
import numpy as np

from antispoofing.spectralcubes.utils import *
from antispoofing.spectralcubes.classification import Classification
from antispoofing.spectralcubes.lowlevelfeatures import LowLevelFeatures
from antispoofing.spectralcubes.midlevelfeatures import MidLevelFeatures
from antispoofing.spectralcubes.protocols.protocol import Protocol


class CrossKFoldProtocol(Protocol):
    """docstring for CrossKFoldProtocol"""

    def __init__(self, data_a, data_b, k_fold=10):
        self.data_a = data_a
        self.data_b = data_b
        self.llf_trials_a = None
        self.mlf_trials_a = None
        self.llf_trials_b = None
        self.mlf_trials_b = None
        self.n_jobs = N_JOBS
        self.k_fold = k_fold
        self.random_state = None

    def build_search_space(self):
        """ docstring """

        if self.data_a.__class__.__name__.lower() == 'replayattack':
            llf_search_space_a = {'NTV': 300, 'LGF': 1, 'M': 7}
            mlf_search_space_a = {'CS': 'kmeans', 'SDD': 'class_based', 'DS': 320, 'CP': 'softmax'}
            # llf_search_space_a = {'NTV': 300, 'LGF': 0, 'M': 7}
            # mlf_search_space_a = {'CS': 'kmeans', 'SDD': 'class_based','DS': 160, 'CP': 'softmax'}
        elif self.data_a.__class__.__name__.lower() == 'casia':
            llf_search_space_a = {'NTV': 1200, 'LGF': 1, 'M': 7}
            mlf_search_space_a = {'CS': 'kmeans', 'SDD': 'class_based', 'DS': 320, 'CP': 'softmax'}
            # llf_search_space_a = {'NTV': 1200, 'LGF': 0, 'M': 7}
            # mlf_search_space_a = {'CS': 'kmeans', 'SDD': 'class_based','DS': 160, 'CP': 'softmax'}
        elif self.data_a.__class__.__name__.lower() == 'maskattack':
            # llf_search_space_a = {'NTV': 1200, 'LGF': 0, 'M': 7}
            # mlf_search_space_a = {'CS': 'kmeans', 'SDD': 'unified', 'DS': 160, 'CP': 'softmax'}
            llf_search_space_a = {'NTV': 1200, 'LGF': 1, 'M': 7}
            mlf_search_space_a = {'CS': 'kmeans', 'SDD': 'class_based', 'DS': 320, 'CP': 'softmax'}
        elif self.data_a.__class__.__name__.lower() == 'uvad':
            # llf_search_space_a = {'NTV': 2, 'LGF': 1, 'M': 7}
            # mlf_search_space_a = {'CS': 'random', 'SDD': 'class_based', 'DS': 360, 'CP': 'softmax'}
            llf_search_space_a = {'NTV': 1200, 'LGF': 1, 'M': 7}
            mlf_search_space_a = {'CS': 'kmeans', 'SDD': 'class_based', 'DS': 320, 'CP': 'softmax'}
        else:
            print 'Put in TDTProtocol.build_search_space()\
                    the best cofiguration found for your dataset !'
            sys.exit(0)

        if self.data_b.__class__.__name__.lower() == 'replayattack':
            llf_search_space_b = {'NTV': 300, 'LGF': 1, 'M': 7}
            mlf_search_space_b = {'CS': 'kmeans', 'SDD': 'class_based', 'DS': 320, 'CP': 'softmax'}
            # llf_search_space_b = {'NTV': 300, 'LGF': 0, 'M': 7}
            # mlf_search_space_b = {'CS': 'kmeans', 'SDD': 'class_based','DS': 160, 'CP': 'softmax'}
        elif self.data_b.__class__.__name__.lower() == 'casia':
            llf_search_space_b = {'NTV': 1200, 'LGF': 1, 'M': 7}
            mlf_search_space_b = {'CS': 'kmeans', 'SDD': 'class_based', 'DS': 320, 'CP': 'softmax'}
            # llf_search_space_b = {'NTV': 1200, 'LGF': 0, 'M': 7}
            # mlf_search_space_b = {'CS': 'kmeans', 'SDD': 'class_based','DS': 160, 'CP': 'softmax'}
        elif self.data_b.__class__.__name__.lower() == 'maskattack':
            # llf_search_space_b = {'NTV': 1200, 'LGF': 0, 'M': 7}
            # mlf_search_space_b = {'CS': 'kmeans', 'SDD': 'unified', 'DS': 160, 'CP': 'softmax'}
            llf_search_space_b = {'NTV': 1200, 'LGF': 1, 'M': 7}
            mlf_search_space_b = {'CS': 'kmeans', 'SDD': 'class_based', 'DS': 320, 'CP': 'softmax'}
        elif self.data_b.__class__.__name__.lower() == 'uvad':
            # llf_search_space_b = {'NTV': 2, 'LGF': 1, 'M': 7}
            # mlf_search_space_b = {'CS': 'random', 'SDD': 'class_based', 'DS': 360, 'CP': 'softmax'}
            llf_search_space_b = {'NTV': 1200, 'LGF': 1, 'M': 7}
            mlf_search_space_b = {'CS': 'kmeans', 'SDD': 'class_based', 'DS': 320, 'CP': 'softmax'}
        else:
            print 'Put in TDTProtocol.build_search_space()\
                    the best cofiguration found for your dataset !'
            sys.exit(0)

        # -- Best for UVAD
        # mlf_search_space_a = {'CS': 'random', 'SDD': 'unified','DS': 320, 'CP': 'softmax'}
        # mlf_search_space_b = {'CS': 'random', 'SDD': 'unified','DS': 320, 'CP': 'softmax'}

        # mlf_search_space_a = {'CS': 'kmeans', 'SDD': 'class_based', 'DS': 160, 'CP': 'softmax'}
        # mlf_search_space_b = {'CS': 'kmeans', 'SDD': 'class_based', 'DS': 160, 'CP': 'softmax'}

        mlf_search_space_a.update(llf_search_space_a)
        self.llf_trials_a = [llf_search_space_a]
        self.mlf_trials_a = [mlf_search_space_a]

        mlf_search_space_b.update(llf_search_space_b)
        self.llf_trials_b = [llf_search_space_b]
        self.mlf_trials_b = [mlf_search_space_b]

    @staticmethod
    def building_metainfo(metainfo_a):

        metainfo_train = {'all_fnames': metainfo_a['all_fnames'],
                          'all_labels': metainfo_a['all_labels'],
                          'train_idxs': metainfo_a['train_idxs_for_cross'],
                          }

        return metainfo_train

    def extract_low_level_features(self):

        start = get_time()

        np.random.seed(7)
        sd = int(np.random.rand(1, 1)[0, 0] * 1e9)
        self.n_jobs = N_JOBS

        # if self.data_a.__class__.__name__.lower() == 'replayattack':
        #     # fnames = [fname for fname in self.data_a.metainfo['all_fnames']
        #     #                 if ('train/' in fname)or('devel/' in fname)or('test/' in fname)]
        #     fnames_a = self.data_a.metainfo['all_fnames']
        # elif self.data_a.__class__.__name__.lower() == 'casia':
        #     fnames = self.data_a.metainfo['all_fnames']
        # elif self.data_a.__class__.__name__.lower() == 'maskattack':
        #     fnames = self.data_a.metainfo['all_fnames']
        # elif self.data_a.__class__.__name__.lower() == 'uvad':
        #     fnames = self.data_a.metainfo['all_fnames']
        #     # # search substrings
        #     # fnames_camera = fnames[np.array(["attack/nikon" in s
        #     #     for s in fnames.flat]).reshape(fnames.shape)]
        #     # fnames = [fname for fname in fnames_camera if ('monitor7' in fname)]
        # else:
        #     pass

        # if self.data_b.__class__.__name__.lower() == 'replayattack':
        #     # fnames = [fname for fname in self.data_b.metainfo['all_fnames']
        #     #                 if ('train/' in fname)or('devel/' in fname)or('test/' in fname)]
        #     fnames_a = self.data_b.metainfo['all_fnames']
        # elif self.data_b.__class__.__name__.lower() == 'casia':
        #     fnames = self.data_b.metainfo['all_fnames']
        # elif self.data_b.__class__.__name__.lower() == 'maskattack':
        #     fnames = self.data_b.metainfo['all_fnames']
        # elif self.data_b.__class__.__name__.lower() == 'uvad':
        #     fnames = self.data_b.metainfo['all_fnames']
        #     # # search substrings
        #     # fnames_camera = fnames[np.array(["attack/nikon" in s
        #     #     for s in fnames.flat]).reshape(fnames.shape)]
        #     # fnames = [fname for fname in fnames_camera if ('monitor7' in fname)]
        # else:
        #     pass

        fnames_a = self.data_a.metainfo['all_fnames']

        tasks = []
        for fname in fnames_a:
            for trial in self.llf_trials_a:
                path = 'low_level_features/{0}/{1}/{2}'.format(trial["NTV"],
                                                               rois_available[trial["LGF"]],
                                                               measures_available[trial["M"]])

                output_path = os.path.join(self.data_a.output_path, path)

                tasks += [LowLevelFeatures(self.data_a.dataset_path, output_path, fname,
                                           trial["LGF"],
                                           trial["M"],
                                           n_cuboids=trial["NTV"],
                                           seed=sd,
                                           )]

        print "running %d tasks in parallel" % len(tasks)
        sys.stdout.flush()
        RunInParallel(tasks, self.n_jobs).run()

        fnames_b = self.data_b.metainfo['all_fnames']

        tasks = []
        for fname in fnames_b:
            for trial in self.llf_trials_b:
                path = 'low_level_features/{0}/{1}/{2}'.format(trial["NTV"],
                                                               rois_available[trial["LGF"]],
                                                               measures_available[trial["M"]])

                output_path = os.path.join(self.data_b.output_path, path)

                tasks += [LowLevelFeatures(self.data_b.dataset_path, output_path, fname,
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
        codebook_path = ''

        for trial in self.mlf_trials_a:
            path = 'low_level_features/{0}/{1}/{2}'.format(trial["NTV"],
                                                           rois_available[trial["LGF"]],
                                                           measures_available[trial["M"]])

            input_path_a = os.path.join(self.data_a.output_path, path)

            metainfo_feats_a = self.data_a.metainfo_feats(input_path_a, ['npy'])

            metainfo_train = self.building_metainfo(metainfo_feats_a)

            output_path = os.path.join(input_path_a, trial["CS"], trial["SDD"], str(trial["DS"]))
            output_path = output_path.replace('low_level_features', 'mid_level_features')

            print 'building dictionary', output_path
            sys.stdout.flush()

            midlevelfeats = MidLevelFeatures(metainfo_train, input_path_a, output_path,
                                             codebook_selection=trial["CS"],
                                             codebook_build=trial["SDD"],
                                             codebook_size=trial["DS"],
                                             n_jobs=self.n_jobs)

            midlevelfeats.build_codebook()

        for trial in self.mlf_trials_a:
            path = 'low_level_features/{0}/{1}/{2}'.format(trial["NTV"],
                                                           rois_available[trial["LGF"]],
                                                           measures_available[trial["M"]])

            input_path_a = os.path.join(self.data_a.output_path, path)
            metainfo_feats_a = self.data_a.metainfo_feats(input_path_a, ['npy'])

            codebook_path = os.path.join(input_path_a, trial["CS"], trial["SDD"], str(trial["DS"]))
            codebook_path = codebook_path.replace('low_level_features', 'mid_level_features')

            output_path_a = os.path.join(input_path_a, trial["CS"], trial["SDD"], str(trial["DS"]))
            output_path_a = output_path_a.replace('low_level_features', 'mid_level_features')

            print 'coding and pooling features', output_path_a
            sys.stdout.flush()

            midlevelfeats = MidLevelFeatures(metainfo_feats_a, input_path_a, output_path_a,
                                             codebook_path=codebook_path,
                                             codebook_selection=trial["CS"],
                                             codebook_build=trial["SDD"],
                                             codebook_size=trial["DS"],
                                             coding_poling=trial["CP"],
                                             n_jobs=self.n_jobs)

            midlevelfeats.run()

        self.random_state = np.random.RandomState(7)
        for k in xrange(self.k_fold):
            for trial in self.mlf_trials_b:
                path = 'low_level_features/{0}/{1}/{2}'.format(trial["NTV"],
                                                               rois_available[trial["LGF"]],
                                                               measures_available[trial["M"]])
                input_path_b = os.path.join(self.data_b.output_path, path)
                metainfo_feats_b = self.data_b.metainfo_feats(input_path_b, ['npy'], random_state=self.random_state)

                folds = "fold_{0}".format(k)
                output_path_b = os.path.join(self.data_b.output_path,
                                             folds,
                                             path,
                                             trial["CS"],
                                             trial["SDD"],
                                             str(trial["DS"]))

                output_path_b = output_path_b.replace('low_level_features', 'mid_level_features')

                print 'coding and pooling features', output_path_b
                sys.stdout.flush()

                midlevelfeats = MidLevelFeatures(metainfo_feats_b, input_path_b, output_path_b,
                                                 codebook_path=codebook_path,
                                                 codebook_selection=trial["CS"],
                                                 codebook_build=trial["SDD"],
                                                 codebook_size=trial["DS"],
                                                 coding_poling=trial["CP"],
                                                 n_jobs=self.n_jobs)

                midlevelfeats.run()

        total_time_elapsed(start, get_time())

    def classification(self):
        """ docstring """

        algos = ["svm"]
        for algo in algos:

            metainfo_feats_a = []
            input_path_a = ""

            for trial in self.mlf_trials_a:
                path = 'mid_level_features/{0}/{1}/{2}/{3}/{4}/{5}/{6}'.format(trial["NTV"],
                                                                               rois_available[trial["LGF"]],
                                                                               measures_available[trial["M"]],
                                                                               trial["CS"], trial["SDD"], trial["DS"],
                                                                               trial["CP"])

                input_path_a = os.path.join(self.data_a.output_path, path)
                metainfo_feats_a = self.data_a.metainfo_feats(input_path_a, ['npy'])

            self.random_state = np.random.RandomState(7)
            for k in xrange(self.k_fold):
                for trial in self.mlf_trials_b:
                    folds = "fold_{0}".format(k)
                    path = '{0}/mid_level_features/{1}/{2}/{3}/{4}/{5}/{6}/{7}'.format(folds,
                                                                                       trial["NTV"], rois_available[trial["LGF"]],
                                                                                       measures_available[trial["M"]],
                                                                                       trial["CS"], trial["SDD"], trial["DS"],
                                                                                       trial["CP"])

                    input_path_b = os.path.join(self.data_b.output_path, path)
                    metainfo_feats_b = self.data_b.metainfo_feats(input_path_b, ['npy'], random_state=self.random_state)

                    output_model = os.path.join(input_path_a, algo)
                    output_model = output_model.replace('mid_level_features', 'classifiers')

                    output_path = os.path.join(input_path_b, algo)
                    output_path = output_path.replace('mid_level_features', 'classifiers')

                    print "output_path", output_path

                    Classification(output_path, metainfo_feats_a,
                                   output_model=output_model,
                                   metafeat_b=metainfo_feats_b,
                                   algo=algo).cross_dataset_protocol()

        print '\t- done!'

    def execute_protocol(self):

        print "building search space ..."
        self.build_search_space()

        print "computing low level features ..."
        self.extract_low_level_features()

        print "computing mid level features ..."
        self.extract_mid_level_features()

        print "building classifiers ..."
        self.classification()
