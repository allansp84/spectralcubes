# -*- coding: utf-8 -*-

import os
from glob import glob
import numpy as np
import itertools

from dataset import Dataset


class UVAD(Dataset):

    def __init__(self, dataset_path, output_path='./working', file_types=['MOV', 'MP4'], face_locations_path=None):
        super(UVAD, self).__init__(dataset_path, output_path)
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

    def _build_meta(self, inpath, filetypes):

        img_idx = 0

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

                elif camera_name in test_range:

                    all_fnames += [fname]
                    all_labels += [int('real' in class_name)]
                    all_idxs += [img_idx]
                    test_idxs += [img_idx]
                    img_idx += 1

                else:
                    pass

        all_fnames = np.array(all_fnames)
        all_labels = np.array(all_labels)

        all_idxs = np.array(all_idxs)
        train_idxs = np.array(train_idxs)
        test_idxs = np.array(test_idxs)

        # pos_idxs = np.where(all_labels[all_idxs]==1)[0]
        # train_idxs_pos, test_idxs_pos = np.split(all_idxs[pos_idxs], 2)
        # rstate = np.random.RandomState(7)
        # n_train_idxs_pos  = int(len(train_idxs_pos)*1)
        # n_test_idxs_pos  = int(len(test_idxs_pos)*1)
        # n_sony_idxs  = int(len(sony_idxs)*1)
        # n_canon_idxs = int(len(canon_idxs)*1)
        # n_nikon_idxs = int(len(nikon_idxs)*1)
        # train_idxs_pos = rstate.permutation(train_idxs_pos)[:n_train_idxs_pos]
        # test_idxs_pos = rstate.permutation(test_idxs_pos)[:n_test_idxs_pos]
        # sony_idxs = rstate.permutation(sony_idxs)[:n_sony_idxs]
        # canon_idxs = rstate.permutation(canon_idxs)[:n_canon_idxs]
        # nikon_idxs = rstate.permutation(nikon_idxs)[:n_nikon_idxs]
        # sony_camera = sorted(np.concatenate((sony_idxs, test_idxs_pos)))
        # canon_camera = sorted(np.concatenate((canon_idxs, test_idxs_pos)))
        # nikon_camera = sorted(np.concatenate((nikon_idxs, test_idxs_pos)))
        # train_idxs = sorted(np.concatenate((sony_idxs, canon_idxs, train_idxs_pos)))
        # test_idxs = [nikon_camera]

        train_idxs_for_cross, devel_idxs_for_cross = self.split_data(all_labels, all_idxs, training_rate=0.8)

        train_idxs, devel_idxs = self.split_data(all_labels, train_idxs, training_rate=0.8)

        r_dict = {'all_fnames': all_fnames,
                  'all_labels': all_labels,
                  'all_idxs': all_idxs,
                  'train_idxs_for_cross': train_idxs_for_cross,
                  'devel_idxs_for_cross': devel_idxs_for_cross,
                  'train_idxs': train_idxs,
                  'devel_idxs': devel_idxs,
                  'test_idxs': {'test': test_idxs,
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
