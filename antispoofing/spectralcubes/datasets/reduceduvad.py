# -*- coding: utf-8 -*-

import os
from glob import glob
import numpy as np
import itertools

from dataset import Dataset


class ReducedUVAD(Dataset):

    def __init__(self, dataset_path, output_path='./working', file_types=['MOV', 'MP4'], face_locations_path=None):
        super(ReducedUVAD, self).__init__(dataset_path, output_path)
        self.file_types = file_types
        self.face_locations_path = face_locations_path

    @staticmethod
    def split_data(all_labels, all_idxs, training_rate):

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

    def _build_meta(self, inpath, filetypes, to_reduce=False):

        img_idx = 0
        training_rate = 0.8

        all_fnames = []
        all_labels = []
        all_idxs = []

        train_idxs = []
        test_idxs = []

        # canon_idxs = []
        # kodac_idxs = []
        # nikon_idxs = []
        # olympus_idxs = []
        # panasonic_idxs = []
        # sony_idxs = []

        cameras = {'sony': [], 'kodac': [], 'olympus': [], 'nikon': [], 'canon': [], 'panasonic': []}

        train_range = ['sony', 'kodac', 'olympus']
        test_range = ['nikon', 'canon', 'panasonic']

        folders = [self._list_dirs(inpath, filetype) for filetype in filetypes]
        # flat and sort list of fnames
        folders = itertools.chain.from_iterable(folders)
        folders = sorted(list(folders))

        for i, folder in enumerate(folders):
            fnames = [glob(os.path.join(inpath, folder, '*' + filetype)) for filetype in filetypes]
            fnames = itertools.chain.from_iterable(fnames)
            fnames = sorted(list(fnames))

            for fname in fnames:

                rel_path = os.path.relpath(fname, inpath)

                class_name = rel_path.split('/')[0]
                camera_name = rel_path.split('/')[1]

                # filename = os.path.basename(fname)
                # name_video = os.path.splitext(filename)[0]
                # video_number = int(''.join([i for i in name_video if i.isdigit()]))

                if camera_name in train_range:

                    all_fnames += [fname]
                    all_labels += [int('real' in class_name)]
                    all_idxs += [img_idx]
                    train_idxs += [img_idx]
                    img_idx += 1

                    cameras[camera_name] += [img_idx]

                elif camera_name in test_range:

                    all_fnames += [fname]
                    all_labels += [int('real' in class_name)]
                    all_idxs += [img_idx]
                    test_idxs += [img_idx]
                    img_idx += 1

                    cameras[camera_name] += [img_idx]

                else:
                    pass

        all_fnames = np.array(all_fnames)
        all_labels = np.array(all_labels)

        all_idxs = np.array(all_idxs)
        train_idxs = np.array(train_idxs)
        test_idxs = np.array(test_idxs)

        if to_reduce is True:
            # -- Reduce Train set
            rstate = np.random.RandomState(7)
            train_pos_idxs = np.where(all_labels[train_idxs] == 1)[0]
            train_neg_idxs = np.where(all_labels[train_idxs] == 0)[0]
            reduction_rate = 0.1
            n_samples_neg = int(len(all_idxs[train_neg_idxs]) * reduction_rate)
            rand_idxs_neg = rstate.permutation(all_idxs[train_neg_idxs])
            train_idxs_rand_neg = rand_idxs_neg[:n_samples_neg]
            train_idxs_reduced = np.concatenate((train_idxs_rand_neg, train_pos_idxs))

            train_idxs = train_idxs_reduced

        train_idxs_for_cross, devel_idxs_for_cross = self.split_data(all_labels, all_idxs, training_rate)

        r_dict = {'all_fnames': all_fnames,
                  'all_labels': all_labels,
                  'all_idxs': all_idxs,
                  'train_idxs_for_cross': train_idxs_for_cross,
                  'devel_idxs_for_cross': devel_idxs_for_cross,
                  'train_idxs': train_idxs,
                  'test_idxs': {'test': test_idxs,
                                },
                  }

        return r_dict

    @property
    def metainfo(self):
        try:
            return self.__metainfo
        except AttributeError:
            self.__metainfo = self._build_meta(self.dataset_path, self.file_types, to_reduce=True)
            return self.__metainfo

    def metainfo_feats(self, output_path, file_types, to_reduce=True):
        return self._build_meta(output_path, file_types, to_reduce=True)

    def metainfo_facelocations(self, file_types):
        if self.face_locations_path is None:
            return None
        else:
            return self._build_meta(self.face_locations_path, file_types)

    def metainfo_images(self, output_path, file_types):
        return self._build_meta(output_path, file_types)
