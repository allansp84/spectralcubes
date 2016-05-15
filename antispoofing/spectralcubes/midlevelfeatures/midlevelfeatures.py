# -*- coding: utf-8 -*-

import os
import sys
import cPickle
import numpy as np

from sklearn.cluster import KMeans
from sklearn.cluster.k_means_ import _init_centroids
from sklearn.metrics.pairwise import chi2_kernel
from sklearn.metrics.pairwise import pairwise_distances

from antispoofing.spectralcubes.utils import N_JOBS


class MidLevelFeatures(object):
    cs_dict = {"random": 0, "kmeans": 1}
    cp_dict = ["hardsum", "hardmax", "softmax"]
    sdd_dict = {"unified": 0, "class_based": 1}
    seed = 42
    debug = True

    def __init__(self, meta, input_path, output_path,
                 codebook_path=None,
                 codebook_selection="random",
                 codebook_build="unified",
                 codebook_size=80,
                 coding_poling="hardsum",
                 file_type="npy",
                 n_jobs=N_JOBS):

        # -- private attributes
        self.__input_path = ""
        self.__output_path = ""
        self.__codebook_selection = {}
        self.__codebook_build = {}
        self.__coding_poling = {}

        # -- public attributes
        self._meta = meta
        self.input_path = input_path
        self.output_path = output_path
        self.codebook_path = codebook_path
        self.codebook_selection = codebook_selection
        self.codebook_build = codebook_build
        self.coding_poling = coding_poling
        self.codebook_size = codebook_size
        self.file_type = file_type
        self.variance = 0.04
        self.n_jobs = n_jobs

        if codebook_path is None:
            self._fname_codebook_pos = "{0}/codebook/positive_class.codebook".format(self.output_path)
            self._fname_codebook_neg = "{0}/codebook/negative_class.codebook".format(self.output_path)
            self._fname_codebook_unified = "{0}/codebook/unified.codebook".format(self.output_path)

        else:

            self._fname_codebook_pos = "{0}/codebook/positive_class.codebook".format(codebook_path)
            self._fname_codebook_neg = "{0}/codebook/negative_class.codebook".format(codebook_path)
            self._fname_codebook_unified = "{0}/codebook/unified.codebook".format(codebook_path)

    @property
    def input_path(self):
        return self.__input_path

    @input_path.setter
    def input_path(self, path):
        self.__input_path = os.path.abspath(path)

    @property
    def output_path(self):
        return self.__output_path

    @output_path.setter
    def output_path(self, path):
        path = os.path.abspath(path)
        self.__output_path = path

    @property
    def codebook_selection(self):
        return self.__codebook_selection

    @codebook_selection.setter
    def codebook_selection(self, value):
        try:
            assert value in self.cs_dict
            self.__codebook_selection = self.cs_dict[value]
        except AssertionError:
            raise AssertionError("Value not found: choose 'random' or 'kmeans'")

    @property
    def coding_poling(self):
        return self.__coding_poling

    @coding_poling.setter
    def coding_poling(self, value):
        try:
            assert value in self.cp_dict
            self.__coding_poling = value
        except AssertionError:
            raise AssertionError("Value not found: choose 'hardsum', 'hardmax', or 'softmax'")

    @property
    def codebook_build(self):
        return self.__codebook_build

    @codebook_build.setter
    def codebook_build(self, value):
        try:
            self.__codebook_build = self.sdd_dict[value]
        except KeyError:
            raise KeyError("Value not found: choose 'unified' or 'class_based'")

    def __load_features(self, fnames):
        feats = []
        for i, fname in enumerate(fnames):
            if 'npy' in self.file_type:
                feats += [np.load(fname)]
            else:
                values = np.loadtxt(fname, delimiter=',')
                values = values[:, np.newaxis, :]
                feats += [values]

        return np.array(feats)

    def __load_train_features(self):

        if self.debug:
            print '\t- loading low level features ...'
            sys.stdout.flush()

        all_labels = self._meta['all_labels']
        all_fnames = self._meta['all_fnames']
        train_idxs = self._meta['train_idxs']

        return all_labels[train_idxs], self.__load_features(all_fnames[train_idxs])

    def __load_all_features(self):

        if self.debug:
            print '\t- loading low level features ...'
            sys.stdout.flush()

        all_fnames = self._meta['all_fnames']
        all_labels = self._meta['all_labels']

        return all_labels, self.__load_features(all_fnames), all_fnames

    def create_codebook(self, features, _class='label'):

        if self.debug:
            print '\t- creating visual codebook for {0} ...'.format(_class)
            print '\t- features.shape', features.shape
            sys.stdout.flush()

        n_feats, n_cuboids, cuboid_depth = features.shape
        features = features.reshape(-1, cuboid_depth)

        if self.codebook_selection == self.cs_dict["kmeans"]:

            codebook = KMeans(init='k-means++', n_clusters=self.codebook_size, n_init=50,
                              tol=1e-10, max_iter=1000, random_state=self.seed, n_jobs=self.n_jobs)

            codebook.fit(features)

            return codebook

        else:

            codebook = KMeans(init='random', n_clusters=self.codebook_size, n_init=1,
                              tol=1e-10, max_iter=1, random_state=self.seed, n_jobs=self.n_jobs)

            codebook.cluster_centers_ = _init_centroids(features, k=self.codebook_size, init='random', random_state=self.seed)

            return codebook

    @staticmethod
    def pickle(fname, data):

        try:
            os.makedirs(os.path.dirname(fname))
        except OSError:
            pass

        fo = open(fname, 'wb')
        cPickle.dump(data, fo)
        fo.close()

    @staticmethod
    def unpickle(fname):
        fo = open(fname, 'rb')
        data = cPickle.load(fo)
        fo.close()
        return data

    def build_unified_codebook(self, feats):

        if not (os.path.exists(self._fname_codebook_unified)):
            codebook_unified = self.create_codebook(feats)
            self.pickle(self._fname_codebook_unified, codebook_unified)

    def build_class_based_codebook(self, labels, feats):

        if not (os.path.exists(self._fname_codebook_pos)):
            train_idxs_pos = np.where(labels == 1)
            feats_train_pos = feats[train_idxs_pos]
            codebook_pos = self.create_codebook(feats_train_pos, _class='pos')
            self.pickle(self._fname_codebook_pos, codebook_pos)

        if not (os.path.exists(self._fname_codebook_neg)):
            train_idxs_neg = np.where(labels == 0)
            feats_train_neg = feats[train_idxs_neg]
            codebook_neg = self.create_codebook(feats_train_neg, _class='neg')
            self.pickle(self._fname_codebook_neg, codebook_neg)

    def coding_class_based(self, codebook_pos_, codebook_neg_, feats_):

        feats = feats_.copy()
        codebook_pos = codebook_pos_.copy()
        codebook_neg = codebook_neg_.copy()

        if self.debug:
            print '\t- coding features ...'
            sys.stdout.flush()

        if 'hard' in self.coding_poling:

            print "\t- feats.shape", feats.shape

            coded_feats = np.zeros((feats.shape[:2] + (self.codebook_size + self.codebook_size,)), dtype=np.int)

            feats = feats.reshape(feats.shape[0], feats.shape[1], -1)
            idxs_cuboid = np.arange(feats.shape[1])

            codebook_pos -= codebook_pos.min(axis=1).reshape(-1, 1)
            codebook_neg -= codebook_neg.min(axis=1).reshape(-1, 1)

            for sample in range(feats.shape[0]):
                feats[sample] -= feats[sample].min(axis=1).reshape(-1, 1)

                dists_pos = pairwise_distances(feats[sample], codebook_pos, metric="cosine")
                dists_neg = pairwise_distances(feats[sample], codebook_neg, metric="cosine")

                dists = np.hstack((dists_neg, dists_pos))
                idxs = np.argmin(dists, axis=1)

                coded_feats[sample, idxs_cuboid, idxs] = 1

        elif 'soft' in self.coding_poling:

            print "\t- feats.shape", feats.shape
            coded_feats = np.zeros((feats.shape[:2] + (self.codebook_size + self.codebook_size,)), dtype=np.float)
            feats = feats.reshape(feats.shape[0], feats.shape[1], -1)

            beta = 1.0 / (2.0 * self.variance)

            codebook_pos -= codebook_pos.min(axis=1).reshape(-1, 1)
            codebook_neg -= codebook_neg.min(axis=1).reshape(-1, 1)

            for sample in range(feats.shape[0]):
                feats[sample] -= feats[sample].min(axis=1).reshape(-1, 1)

                dists_pos = chi2_kernel(feats[sample], codebook_pos, gamma=beta)
                dists_neg = chi2_kernel(feats[sample], codebook_neg, gamma=beta)

                cfnorm = dists_pos.sum(axis=1).reshape(-1, 1)
                cfnorm[cfnorm == 0] = 1.
                dists_pos /= cfnorm

                cfnorm = dists_neg.sum(axis=1).reshape(-1, 1)
                cfnorm[cfnorm == 0] = 1.
                dists_neg /= cfnorm

                coded_feats[sample] = np.hstack((dists_neg, dists_pos))

        else:
            raise ValueError('Coding method not implemented')

        return coded_feats

    def coding_unified(self, codebook_, feats_):

        feats = feats_.copy()
        codebook = codebook_.copy()

        if self.debug:
            print '\t- coding features ...'
            sys.stdout.flush()

        if 'hard' in self.coding_poling:

            coded_feats = np.zeros((feats.shape[:2] + (self.codebook_size,)), dtype=np.int)

            feats = feats.reshape(feats.shape[0], feats.shape[1], -1)
            idxs_cuboid = np.arange(feats.shape[1])

            codebook -= codebook.min(axis=1).reshape(-1, 1)

            for sample in range(feats.shape[0]):
                feats[sample] -= feats[sample].min(axis=1).reshape(-1, 1)
                idxs = np.argmin(pairwise_distances(feats[sample], codebook, metric="cosine"), axis=1)
                coded_feats[sample, idxs_cuboid, idxs] = 1

        elif 'soft' in self.coding_poling:

            coded_feats = np.zeros((feats.shape[:2] + (self.codebook_size,)), dtype=np.float)
            beta = 1.0 / (2.0 * self.variance)
            codebook -= codebook.min(axis=1).reshape(-1, 1)

            for sample in range(feats.shape[0]):
                feats[sample] -= feats[sample].min(axis=1).reshape(-1, 1)
                coded_feats[sample] = chi2_kernel(feats[sample], codebook, gamma=beta)

                cfnorm = coded_feats[sample].sum(axis=1).reshape(-1, 1)
                cfnorm[cfnorm == 0] = 1.
                coded_feats[sample] /= cfnorm

        else:
            raise ValueError('Coding method not implemented')

        return coded_feats

    def pooling(self, coded_feats):

        if self.debug:
            print '\t- pooling features ...'
            sys.stdout.flush()

        if 'sum' in self.coding_poling:

            pooled_feats = []
            for sample in range(coded_feats.shape[0]):
                pooled_feats += [coded_feats[sample].sum(axis=0)]
            pooled_feats = np.array(pooled_feats)

        elif 'max' in self.coding_poling:

            pooled_feats = []
            for sample in range(coded_feats.shape[0]):
                pooled_feats += [coded_feats[sample].max(axis=0)]
            pooled_feats = np.array(pooled_feats)

        else:
            raise ValueError('Pooling method not implemented')

        return pooled_feats

    def feature_extraction_with_unified_codebook(self, feats):

        codebook = self.unpickle(self._fname_codebook_unified)

        coded_feats = self.coding_unified(codebook.cluster_centers_, feats)

        pooled_feats = self.pooling(coded_feats)

        return pooled_feats

    def feature_extraction_with_class_based_dictionary(self, feats):

        codebook_pos = self.unpickle(self._fname_codebook_pos)
        codebook_neg = self.unpickle(self._fname_codebook_neg)

        # coded_feats = self.coding_class_based(codebook_pos.cluster_centers_, codebook_neg.cluster_centers_, feats)
        coded_feats_neg = self.coding_unified(codebook_neg.cluster_centers_, feats)
        coded_feats_pos = self.coding_unified(codebook_pos.cluster_centers_, feats)
        coded_feats = np.concatenate((coded_feats_neg, coded_feats_pos), axis=2)
        del coded_feats_neg
        del coded_feats_pos

        pooled_feats = self.pooling(coded_feats)

        return pooled_feats

    def save_features(self, mid_level_feats, fnames):

        print '\t- saving mid level features ...'
        sys.stdout.flush()

        for fname, feat_vector in zip(fnames, mid_level_feats):
            relfname = os.path.os.path.relpath(fname, self.input_path)
            output_fname = os.path.join(self.output_path, self.coding_poling, relfname)
            # output_fname = output_fname.replace("llf", "mlf")

            try:
                os.makedirs(os.path.dirname(output_fname))
            except OSError:
                pass

            np.save(output_fname, feat_vector[np.newaxis, :])

    def build_codebook(self):

        if self.codebook_build == self.sdd_dict["class_based"]:

            labels, feats = self.__load_train_features()
            self.build_class_based_codebook(labels, feats)

        else:

            labels, feats = self.__load_train_features()
            self.build_unified_codebook(feats)

        return True

    def run(self):

        if self.codebook_build == self.sdd_dict["class_based"]:

            all_labels, all_feats, all_fnames = self.__load_all_features()
            all_mid_level_feats = self.feature_extraction_with_class_based_dictionary(all_feats)
            self.save_features(all_mid_level_feats, all_fnames)

        else:

            all_labels, all_feats, all_fnames = self.__load_all_features()
            all_mid_level_feats = self.feature_extraction_with_unified_codebook(all_feats)
            self.save_features(all_mid_level_feats, all_fnames)

        return True
