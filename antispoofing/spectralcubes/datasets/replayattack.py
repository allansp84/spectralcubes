# -*- coding: utf-8 -*-

import os
import itertools
import numpy as np
from glob import glob
from dataset import Dataset


class ReplayAttack(Dataset):

    def __init__(self, dataset_path, output_path='./working', file_types=['mov'], face_locations_path=None):
        super(ReplayAttack, self).__init__(dataset_path, output_path)
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
        devel_idxs = []
        test_idxs = []
        anon_idxs = []

        attack_highdef = []
        attack_mobile = []
        attack_print = []
        attack_fixed = []
        attack_hand = []
        train_idxs_for_cross = []

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

                if not 'enroll' in fname:
                    if 'train/' in os.path.relpath(fname, inpath):
                        all_idxs += [img_idx]
                        train_idxs_for_cross += [img_idx]
                        train_idxs += [img_idx]
                        all_fnames += [fname]
                        all_labels += [int('real' in (os.path.relpath(fname, inpath)))]
                        img_idx += 1
                    else:
                        if 'devel/' in os.path.relpath(fname, inpath):
                            all_idxs += [img_idx]
                            train_idxs_for_cross += [img_idx]
                            devel_idxs += [img_idx]
                            all_fnames += [fname]
                            all_labels += [int('real' in (os.path.relpath(fname, inpath)))]
                            img_idx += 1

                        elif 'test/' in os.path.relpath(fname, inpath):

                            if 'attack_highdef' in os.path.relpath(fname, inpath):
                                attack_highdef += [img_idx]
                            elif 'attack_mobile' in os.path.relpath(fname, inpath):
                                attack_mobile += [img_idx]
                            elif 'attack_print' in os.path.relpath(fname, inpath):
                                attack_print += [img_idx]
                            else:
                                pass

                            if 'attack/fixed' in os.path.relpath(fname, inpath):
                                attack_fixed += [img_idx]
                            elif 'attack/hand' in os.path.relpath(fname, inpath):
                                attack_hand += [img_idx]
                            else:
                                pass

                            all_idxs += [img_idx]
                            test_idxs += [img_idx]
                            all_fnames += [fname]
                            all_labels += [int('real' in (os.path.relpath(fname, inpath)))]
                            img_idx += 1

                        # elif 'competition_icb2013/' in os.path.relpath(fname, inpath):
                        #     anon_idxs += [img_idx]
                        #     all_fnames += [fname]
                        #     all_labels += [int('real' in (os.path.relpath(fname, inpath)))]
                        #     img_idx += 1

                        else:
                            pass

        all_fnames = np.array(all_fnames)
        all_labels = np.array(all_labels)
        all_idxs = np.array(all_idxs)

        train_idxs = np.array(train_idxs)
        devel_idxs = np.array(devel_idxs)
        test_idxs = np.array(test_idxs)

        train_idxs_for_cross, devel_idxs_for_cross = self.split_data(all_labels, all_idxs, training_rate=0.8)

        pos_idxs = np.where(all_labels[test_idxs] == 1)[0]

        hand_attack = np.concatenate((attack_hand, test_idxs[pos_idxs]))
        fixed_attack = np.concatenate((attack_fixed, test_idxs[pos_idxs]))

        highdef_attack = np.concatenate((attack_highdef, test_idxs[pos_idxs]))
        mobile_attack = np.concatenate((attack_mobile, test_idxs[pos_idxs]))
        print_attack = np.concatenate((attack_print, test_idxs[pos_idxs]))

        r_dict = {'all_fnames': all_fnames,
                  'all_labels': all_labels,
                  'all_idxs': all_idxs,
                  'train_idxs_for_cross': train_idxs_for_cross,
                  'devel_idxs_for_cross': devel_idxs_for_cross,
                  'train_idxs': train_idxs,
                  'devel_idxs': devel_idxs,
                  'test_idxs': {'test': test_idxs,
                                'hand_attack': hand_attack,
                                'fixed_attack': fixed_attack,
                                'highdef_attack': highdef_attack,
                                'mobile_attack': mobile_attack,
                                'print_attack': print_attack,
                                # anon_idxs,
                                }
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
