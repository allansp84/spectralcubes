# -*- coding: utf-8 -*-

import os
import operator
import itertools
import numpy as np
from glob import glob
from antispoofing.spectralcubes.datasets.dataset import Dataset


class Casia(Dataset):
    def __init__(self, dataset_path, output_path='./working', file_types=['avi'], face_locations_path=None):
        super(Casia, self).__init__(dataset_path, output_path)
        self.file_types = file_types
        self.face_locations_path = face_locations_path

    def split_data(self, all_labels, all_idxs, training_rate):

        rstate = np.random.RandomState(7)

        pos_idxs = np.where(all_labels[all_idxs] == 1)[0]
        neg_idxs = np.where(all_labels[all_idxs] == 0)[0]

        # -- cross dataset idxs
        n_samples_pos = int(len(all_idxs[pos_idxs]) * training_rate)
        n_samples_neg = int(len(all_idxs[neg_idxs]) * training_rate)

        rand_idxs_pos = rstate.permutation(all_idxs[pos_idxs])
        rand_idxs_neg = rstate.permutation(all_idxs[neg_idxs])

        train_idxs_rand_pos = rand_idxs_pos[:n_samples_pos]
        train_idxs_rand_neg = rand_idxs_neg[:n_samples_neg]
        test_idxs_rand_pos = rand_idxs_pos[n_samples_pos:]
        test_idxs_rand_neg = rand_idxs_neg[n_samples_neg:]

        train_idxs_for_cross = np.concatenate((train_idxs_rand_neg, train_idxs_rand_pos))
        devel_idxs_for_cross = np.concatenate((test_idxs_rand_neg, test_idxs_rand_pos))

        return train_idxs_for_cross, devel_idxs_for_cross

    def _build_meta(self, inpath, filetypes):

        img_idx = 0
        training_rate = 0.8

        all_fnames = []
        all_labels = []
        all_idxs = []

        train_idxs = []

        test_idxs_1 = []
        test_idxs_2 = []
        test_idxs_3 = []
        test_idxs_4 = []
        test_idxs_5 = []
        test_idxs_6 = []
        test_idxs_7 = []

        scenario_1 = {"L1", "L2", "L3", "L4"}
        scenario_2 = {"N1", "N2", "N3", "N4"}
        scenario_3 = {"H1", "H2", "H3", "H4"}
        scenario_4 = {"L1", "N1", "H1", "L2", "N2", "H2"}
        scenario_5 = {"L1", "N1", "H1", "L3", "N3", "H3"}
        scenario_6 = {"L1", "N1", "H1", "L4", "N4", "H4"}

        pos_samples = ["1", "2", "HR_1"]
        pos_samples = ["{0}.{1}".format(s, filetype) for filetype in filetypes for s in pos_samples]

        # folders = np.array(sorted(self._list_dirs(inpath, filetype)))
        folders = [self._list_dirs(inpath, filetype) for filetype in filetypes]
        # flat and sort list of fnames
        folders = itertools.chain.from_iterable(folders)
        folders = sorted(list(folders))

        for i, folder in enumerate(folders):
            fnames = [glob(os.path.join(inpath, folder, '*' + filetype)) for filetype in filetypes]
            fnames = itertools.chain.from_iterable(fnames)
            fnames = sorted(list(fnames))

            for fname in fnames:

                filename = os.path.basename(fname)
                positive_class = [s == filename for s in pos_samples]
                name_video = os.path.splitext(filename)[0]

                if 'H' in name_video:
                    video_id = 'H' + name_video.split("_")[1]
                else:
                    if int(name_video) % 2 == 0:
                        idx = np.where(np.arange(2, 9, 2) == int(name_video))[0]
                        idx += 1
                        video_id = 'L%d' % idx[0]

                    else:
                        idx = np.where(np.arange(1, 9, 2) == int(name_video))[0]
                        idx += 1
                        video_id = 'N%d' % idx[0]

                if 'train_release/' in os.path.relpath(fname, inpath):

                    all_fnames += [fname]
                    all_labels += [int(reduce(operator.or_, positive_class))]
                    all_idxs += [img_idx]
                    train_idxs += [img_idx]
                    img_idx += 1

                else:

                    if video_id in scenario_1:
                        test_idxs_1 += [img_idx]

                    if video_id in scenario_2:
                        test_idxs_2 += [img_idx]

                    if video_id in scenario_3:
                        test_idxs_3 += [img_idx]

                    if video_id in scenario_4:
                        test_idxs_4 += [img_idx]

                    if video_id in scenario_5:
                        test_idxs_5 += [img_idx]

                    if video_id in scenario_6:
                        test_idxs_6 += [img_idx]

                    test_idxs_7 += [img_idx]

                    all_fnames += [fname]
                    all_labels += [int(reduce(operator.or_, positive_class))]
                    all_idxs += [img_idx]

                    img_idx += 1

        all_fnames = np.array(all_fnames)
        all_labels = np.array(all_labels)
        all_idxs = np.array(all_idxs)

        train_idxs = np.array(train_idxs)

        test_idxs_1 = np.array(test_idxs_1)
        test_idxs_2 = np.array(test_idxs_2)
        test_idxs_3 = np.array(test_idxs_3)
        test_idxs_4 = np.array(test_idxs_4)
        test_idxs_5 = np.array(test_idxs_5)
        test_idxs_6 = np.array(test_idxs_6)
        test_idxs_7 = np.array(test_idxs_7)

        train_idxs_for_cross, devel_idxs_for_cross = self.split_data(all_labels,
                                                                     all_idxs,
                                                                     training_rate)

        r_dict = {'all_fnames': all_fnames,
                  'all_labels': all_labels,
                  'all_idxs': all_idxs,
                  'train_idxs_for_cross': train_idxs_for_cross,
                  'devel_idxs_for_cross': devel_idxs_for_cross,
                  'train_idxs': train_idxs,
                  'test_idxs': {'test_scenario_1': test_idxs_1,
                                'test_scenario_2': test_idxs_2,
                                'test_scenario_3': test_idxs_3,
                                'test_scenario_4': test_idxs_4,
                                'test_scenario_5': test_idxs_5,
                                'test_scenario_6': test_idxs_6,
                                'test_scenario_7': test_idxs_7,
                                },
                  }

        return r_dict

    @property
    def metainfo(self):
        try:
            return self.__metainfo
        except AttributeError:
            self.__metainfo = self._build_meta(self.dataset_path, self.file_types)
            return self.__metainfo

    def metainfo_feats(self, output_path, file_types):
        return self._build_meta(output_path, file_types)

    def metainfo_facelocations(self, file_types):
        if self.face_locations_path is None:
            return None
        else:
            return self._build_meta(self.face_locations_path, file_types)

    def metainfo_images(self, output_path, file_types):
        return self._build_meta(output_path, file_types)
