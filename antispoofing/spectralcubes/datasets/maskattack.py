# -*- coding: utf-8 -*-

import os
import itertools
import numpy as np
from glob import glob
from antispoofing.spectralcubes.datasets.dataset import Dataset


class MaskAttack(Dataset):

    def __init__(self, dataset_path, output_path='./working', file_types=['avi'], face_locations_path=None):

        super(MaskAttack, self).__init__(dataset_path, output_path)
        self.file_types = file_types
        self.face_locations_path = face_locations_path

    @staticmethod
    def get_fold(k_fold):
        rstate = np.random.RandomState(42)

        k = 0
        fold = []
        while k < k_fold:
            fold = rstate.permutation(17)
            k += 1

        return fold

    @staticmethod
    def split_for_cross(all_labels, all_idxs, training_rate):

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

    def _build_meta(self, inpath, filetypes, random_state=None, fold=None):

        img_idx = 0
        training_rate = 0.8

        all_fnames = []
        all_labels = []
        all_idxs = []

        train_idxs = []
        devel_idxs = []
        test_idxs = []

        if fold is None:
            if random_state is None:
                rstate = np.random.RandomState(7)
                idxs = rstate.permutation(17)
            else:
                idxs = random_state.permutation(17)
            idxs += 1
        else:
            idxs = self.get_fold(fold)
            idxs += 1

        # train_range = idxs[:7]
        # devel_range = idxs[7:12]
        # test_range = idxs[12:]

        train_range = range(1, (7+1))
        devel_range = range(8, (12+1))
        test_range = range(13, (17+1))

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

                user_id, session, n_video, data_type = os.path.basename(fname).split("_")

                if "C" in data_type:
                    if int(session) != 1:

                        pos_class = False if int(session) == 3 else True

                        if int(user_id) in train_range:
                            all_fnames += [fname]
                            all_labels += [int(pos_class)]
                            train_idxs += [img_idx]
                            all_idxs += [img_idx]
                            img_idx += 1

                        elif int(user_id) in devel_range:
                            all_fnames += [fname]
                            all_labels += [int(pos_class)]
                            devel_idxs += [img_idx]
                            all_idxs += [img_idx]
                            img_idx += 1

                        elif int(user_id) in test_range:
                            all_fnames += [fname]
                            all_labels += [int(pos_class)]
                            test_idxs += [img_idx]
                            all_idxs += [img_idx]
                            img_idx += 1

                        else:
                            pass

                    else:
                        pass

                else:
                    pass

        all_fnames = np.array(all_fnames)
        all_labels = np.array(all_labels)
        all_idxs = np.array(all_idxs)

        train_idxs = np.array(train_idxs)
        devel_idxs = np.array(devel_idxs)
        test_idxs = np.array(test_idxs)

        train_idxs_for_cross, devel_idxs_for_cross = self.split_for_cross(all_labels,
                                                                          all_idxs,
                                                                          training_rate)

        r_dict = {'all_fnames': all_fnames,
                  'all_labels': all_labels,
                  'all_idxs': all_idxs,
                  'train_idxs': train_idxs,
                  'devel_idxs': devel_idxs,
                  'test_idxs': {'test': test_idxs}
                  }

        return r_dict

    @property
    def metainfo(self):
        try:
            return self.__metainfo
        except AttributeError:
            self.__metainfo = self._build_meta(self.dataset_path, self.file_types)
            return self.__metainfo

    def metainfo_feats(self, output_path, file_types, random_state=None):
        return self._build_meta(output_path, file_types, random_state)

    def metainfo_facelocations(self, file_types):
        if self.face_locations_path is None:
            return None
        else:
            return self._build_meta(self.face_locations_path, file_types)

    def metainfo_images(self, output_path, file_types):
        return self._build_meta(output_path, file_types)
