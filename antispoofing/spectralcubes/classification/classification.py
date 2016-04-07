# -*- coding: utf-8 -*-

import os
import sys
import time

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc

from antispoofing.spectralcubes.utils import det, det_axis
from antispoofing.spectralcubes.utils import eer_threshold
from antispoofing.spectralcubes.utils import farfrr
# import bob.measure as measure

from .pls import PLSClassifier
from .svm import SVMClassifier

np.seterr(all='raise')


class Classification(object):
    _MAX_FRAME_NUMBERS = 1000
    debug = False

    def __init__(self, output_path, metafeat,
                 output_model=None,
                 metafeat_b=None,
                 file_type="npy",
                 algo='svm',
                 n_components=3,
                 seed=42,
                 fname_report='classification.report',
                 input_path_2=None,
                 frame_numbers=0):

        self.__dataset_path = ""
        self.__output_path = ""
        self.__feat_path = ""
        self.__fname_report = ""
        self.__frame_numbers = 0

        self.output_path = output_path
        self.metafeat = metafeat
        self.output_model = output_model
        self.metafeat_b = metafeat_b
        self.file_type = file_type
        self.algo = algo
        self.n_components = n_components
        self.seed = seed
        self.fname_report = fname_report
        self.input_path_2 = input_path_2
        self.frame_numbers = frame_numbers

    @property
    def dataset_path(self):
        return self.__dataset_path

    @dataset_path.setter
    def dataset_path(self, path):
        self.__dataset_path = os.path.abspath(path)

    @property
    def output_path(self):
        return self.__output_path

    @output_path.setter
    def output_path(self, path):
        self.__output_path = os.path.abspath(path)

    @property
    def feat_path(self):
        return self.__feat_path

    @feat_path.setter
    def feat_path(self, path):
        self.__feat_path = os.path.abspath(path)

    @property
    def fname_report(self):
        return self.__fname_report

    @fname_report.setter
    def fname_report(self, fname):
        self.__fname_report = os.path.join(self.output_path, fname)

    @property
    def frame_numbers(self):
        return self.__frame_numbers

    @frame_numbers.setter
    def frame_numbers(self, frame_numbers):
        if frame_numbers > 0:
            self.__frame_numbers = frame_numbers
        else:
            self.__frame_numbers = self._MAX_FRAME_NUMBERS

    def __load_features(self, fnames):

        n_fnames = len(fnames)
        feat_dimension = np.load(fnames[0]).shape[1]
        feats = np.zeros((n_fnames, feat_dimension), dtype=np.float32)

        fname = ''
        try:
            for i, fname in enumerate(fnames):
                feats[i] = np.load(fname).astype(np.float32)

        except Exception, e:
            import shutil
            shutil.rmtree(os.path.dirname(fname))
            print 'Please, recompute the feature for Video ' + os.path.dirname(fname)
            sys.exit(0)

        return feats

    def __load_all_features(self, fnames, labels, idxs):

        if self.debug:
            print '\t- loading low level features ...'
            sys.stdout.flush()

        return labels[idxs], self.__load_features(fnames[idxs])

    # def __load_all_features(self):

    #     if self.debug:
    #         print '\t- loading low level features ...'
    #         sys.stdout.flush()

    #     all_fnames = self.metafeat['all_fnames']
    #     all_labels = self.metafeat['all_labels']

    #     return all_labels, self.__load_features(all_fnames), all_fnames

    def load_features(self):

        all_fnames = self.metafeat['all_fnames']
        train_idxs = self.metafeat['train_idxs']

        print '\t- loading mid level features ...'
        sys.stdout.flush()

        feats = []
        for i, fname in enumerate(all_fnames):
            feats += [np.load(fname)]

        feats = np.array(feats, dtype=np.float32)

        feats_mean = feats[train_idxs].mean(axis=0)
        feats_std = feats[train_idxs].std(axis=0, ddof=1)
        feats_std[feats_std == 0.] = 1.

        feats -= feats_mean
        feats /= feats_std

        return np.reshape(feats, ((-1,) + feats.shape[2:]))

    def acc_threshold(self, neg_scores, pos_scores, T):

        neg_scores = np.array(neg_scores)
        pos_scores = np.array(pos_scores)

        n_scores = float(len(neg_scores) + len(pos_scores))

        n_fa = (neg_scores >= T).sum()
        n_fr = (pos_scores < T).sum()

        return (1. - ((n_fa + n_fr) / n_scores)) * 100.

    def _plot_score_distributions(self, T, neg_devel, pos_devel, neg_test, pos_test, filename='score_dist.png'):

        plt.clf()
        plt.figure(1)
        plt.subplot(211)
        plt.title("Score distributions (Deve set)")
        n, bins, patches = plt.hist(neg_devel, bins=25, normed=1, histtype='bar', label='Negative class')
        na, binsa, patchesa = plt.hist(pos_devel, bins=25, normed=1, histtype='bar', label='Positive class')
        # add a line showing the expected distribution
        y = mlab.normpdf(bins, np.mean(neg_devel), np.std(neg_devel))
        l = plt.plot(bins, y, 'k--', linewidth=1.5)
        y = mlab.normpdf(binsa, np.mean(pos_devel), np.std(pos_devel))
        l = plt.plot(binsa, y, 'k--', linewidth=1.5)
        plt.axvline(x=T, linewidth=2, color='blue')
        plt.legend()

        plt.subplot(212)
        plt.title("Score distributions (Test set)")
        n, bins, patches = plt.hist(neg_test, bins=25, normed=1, facecolor='green', alpha=0.5, histtype='bar',
                                    label='Negative class')
        na, binsa, patchesa = plt.hist(pos_test, bins=25, normed=1, facecolor='red', alpha=0.5, histtype='bar',
                                       label='Positive class')
        # add a line showing the expected distribution
        y = mlab.normpdf(bins, np.mean(neg_test), np.std(neg_test))
        l = plt.plot(bins, y, 'k--', linewidth=1.5)
        y = mlab.normpdf(binsa, np.mean(pos_test), np.std(pos_test))
        l = plt.plot(binsa, y, 'k--', linewidth=1.5)
        plt.axvline(x=T, linewidth=2, color='blue')
        plt.legend()

        current_dir = os.getcwd()
        saida = '{0}/{1}.png'.format(current_dir, filename)
        plt.savefig(saida)

    def _plot_det_curve(self, neg_test, pos_test, filename, color=0):
        # plotting DET Curve
        plt.clf()

        npoints = 100
        color_graph = (0, 0, 0)

        # if color < 16:
        #     list_chars = list("{0:#b}".format(10)[2:])
        #     color_graph = tuple(list_chars)

        # measure.plot.det(neg_test, pos_test, npoints, color=color_graph, linestyle='-', label='test')
        # # measure.plot.det_axis([0.01, 40, 0.01, 40])
        # measure.plot.det_axis([1, 40, 1, 40])
        # plt.xlabel('FRR (%)')
        # plt.ylabel('FAR (%)')
        # titulo = "DET curve"
        # plt.title(titulo)
        # plt.grid(True)
        # current_dir = os.getcwd()
        # saida = '{0}/{1}.png'.format(current_dir,filename)
        # plt.savefig(saida)

    def __fuse_scores(self, labels, scores):

        n_frames = self.frame_numbers

        # if n_frames%2 == 0:
        #     n_frames -= 1

        r_labels = np.reshape(labels, (-1, n_frames))
        r_scores = np.reshape(scores, (-1, n_frames))

        # # -- max fusion
        # r_labels = r_labels.max(axis=1)
        # r_scores = r_scores.max(axis=1)

        # -- majority-vote fusion
        r_labels = r_labels.sum(axis=1)
        r_labels[r_labels == 0] = -1
        r_labels[r_labels < 0] = 0
        r_labels[r_labels > 0] = 1

        # r_labels = r_labels.sum(axis=1)
        # r_scores = r_scores.sum(axis=1)

        return r_labels, r_scores

    def _get_score_distributions(self, labels, scores):

        # labels, scores = self.__fuse_scores(labels, scores)

        # get the score distributions of positive and negative classes
        pos = [pred for label, pred in zip(labels, scores) if label == 1]  # real
        neg = [pred for label, pred in zip(labels, scores) if label == 0]  # attack

        return np.array(neg), np.array(pos)

    def _calc_hter(self, neg_devel, pos_devel, neg_test, pos_test):

        # calculate threshould upon eer point
        # T = bob.measure.eer_threshold(neg_devel, pos_devel)
        T = eer_threshold(neg_devel, pos_devel)

        # calculate far and frr
        # far, frr = bob.measure.farfrr(neg_test, pos_test, T)
        far, frr = farfrr(neg_test, pos_test, T)

        far *= 100.
        frr *= 100.

        hter = ((far + frr) / 2.)

        return T, far, frr, hter

    def save_descriptors(self, feat_set, train_idxs, devel_idxs, test_idxs, all_labels):

        pos_idxs = np.where(all_labels[train_idxs] == 1)[0]
        neg_idxs = np.where(all_labels[train_idxs] == 0)[0]

        labels = np.reshape(all_labels[train_idxs][pos_idxs], (len(all_labels[train_idxs][pos_idxs]), 1))
        feats = feat_set[train_idxs][pos_idxs]
        fmt = '%d' + ', %f' * feats.shape[1]
        np.savetxt(os.path.join(self.output_path, 'real-train.txt'), np.hstack((labels, feats)), fmt=fmt, delimiter=',',
                   newline='\n')

        labels = np.reshape(all_labels[train_idxs][neg_idxs], (len(all_labels[train_idxs][neg_idxs]), 1))
        labels[labels == 0] = -1
        feats = feat_set[train_idxs][neg_idxs]
        fmt = '%d' + ', %f' * feats.shape[1]
        np.savetxt(os.path.join(self.output_path, 'attack-train.txt'), np.hstack((labels, feats)), fmt=fmt,
                   delimiter=',', newline='\n')

        pos_idxs = np.where(all_labels[devel_idxs] == 1)[0]
        neg_idxs = np.where(all_labels[devel_idxs] == 0)[0]

        labels = np.reshape(all_labels[devel_idxs][pos_idxs], (len(all_labels[devel_idxs][pos_idxs]), 1))
        labels[labels == 0] = -1
        feats = feat_set[devel_idxs][pos_idxs]
        fmt = '%d' + ', %f' * feats.shape[1]
        np.savetxt(os.path.join(self.output_path, 'real-devel.txt'), np.hstack((labels, feats)), fmt=fmt, delimiter=',',
                   newline='\n')

        labels = np.reshape(all_labels[devel_idxs][neg_idxs], (len(all_labels[devel_idxs][neg_idxs]), 1))
        labels[labels == 0] = -1
        feats = feat_set[devel_idxs][neg_idxs]
        fmt = '%d' + ', %f' * feats.shape[1]
        np.savetxt(os.path.join(self.output_path, 'attack-devel.txt'), np.hstack((labels, feats)), fmt=fmt,
                   delimiter=',', newline='\n')

        # pos_idxs = np.where(all_labels[test_idxs]==1)[0]
        # neg_idxs = np.where(all_labels[test_idxs]==0)[0]
        #
        # labels = np.reshape(all_labels[test_idxs][pos_idxs], (len(all_labels[test_idxs][pos_idxs]),1))
        # labels[labels==0] = -1
        # feats = feat_set[test_idxs][pos_idxs]
        # fmt = '%d' + ', %f' * feats.shape[1]
        # np.savetxt(os.path.join(self.output_path, 'real-test.txt'), np.hstack((labels, feats)),
        # fmt=fmt, delimiter=',', newline='\n')
        #
        # labels = np.reshape(all_labels[test_idxs][neg_idxs], (len(all_labels[test_idxs][neg_idxs]),1))
        # labels[labels==0] = -1
        # feats = feat_set[test_idxs][neg_idxs]
        # fmt = '%d' + ', %f' * feats.shape[1]
        # np.savetxt(os.path.join(self.output_path, 'real-attack.txt'), np.hstack((labels, feats)), fmt=fmt, delimiter=',', newline='\n')

    def performance_eval(self, scores, all_results, scores_devel=None):
        class_report = {}
        for score_key in scores:

            if scores_devel is None:
                T, far, frr, hter = self._calc_hter(scores[score_key]['neg'],
                                                    scores[score_key]['pos'],
                                                    scores[score_key]['neg'],
                                                    scores[score_key]['pos'])
            else:
                T, far, frr, hter = self._calc_hter(scores_devel['neg'],
                                                    scores_devel['pos'],
                                                    scores[score_key]['neg'],
                                                    scores[score_key]['pos'])

            acc = self.acc_threshold(scores[score_key]['neg'], scores[score_key]['pos'], T)

            fpr, tpr, thres = roc_curve(all_results[score_key]['gt'], all_results[score_key]['predicted_scores'])
            roc_auc = auc(fpr, tpr) * 100.

            class_report[score_key] = {'acc': acc,
                                       'auc': roc_auc,
                                       'far': far,
                                       'frr': frr,
                                       'hter': hter,
                                       'neg': scores[score_key]['neg'],
                                       'pos': scores[score_key]['pos'],
                                       }

        report = ''
        for k in class_report:
            report += '{0}, acc={1:.4f}, auc={2:.4f}, far=={3:.4f}, frr={4:.4f}, hter={5:.4f}\n'.format(k,
                                                                                                        class_report[k][
                                                                                                            'acc'],
                                                                                                        class_report[k][
                                                                                                            'auc'],
                                                                                                        class_report[k][
                                                                                                            'far'],
                                                                                                        class_report[k][
                                                                                                            'frr'],
                                                                                                        class_report[k][
                                                                                                            'hter'])

        for k in class_report:
            filename_acc = os.path.join(self.output_path, k, 'acc.result')
            filename_auc = os.path.join(self.output_path, k, 'auc.result')
            filename_hter = os.path.join(self.output_path, k, 'hter.result')
            filename_far = os.path.join(self.output_path, k, 'far.result')
            filename_frr = os.path.join(self.output_path, k, 'frr.result')
            filename_pos_scores = os.path.join(self.output_path, k, 'pos.scores')
            filename_neg_scores = os.path.join(self.output_path, k, 'neg.scores')

            if not os.path.exists(os.path.dirname(filename_acc)):
                os.makedirs(os.path.dirname(filename_acc))

            if not os.path.exists(os.path.dirname(filename_auc)):
                os.makedirs(os.path.dirname(filename_auc))

            if not os.path.exists(os.path.dirname(filename_hter)):
                os.makedirs(os.path.dirname(filename_hter))

            if not os.path.exists(os.path.dirname(filename_far)):
                os.makedirs(os.path.dirname(filename_far))

            if not os.path.exists(os.path.dirname(filename_frr)):
                os.makedirs(os.path.dirname(filename_frr))

            if not os.path.exists(os.path.dirname(filename_pos_scores)):
                os.makedirs(os.path.dirname(filename_pos_scores))

            if not os.path.exists(os.path.dirname(filename_neg_scores)):
                os.makedirs(os.path.dirname(filename_neg_scores))

            f = open(filename_acc, 'w')
            f.write("%2.4f" % class_report[k]['acc'])
            f.close()

            f = open(filename_auc, 'w')
            f.write("%2.4f" % class_report[k]['auc'])
            f.close()

            f = open(filename_hter, 'w')
            f.write("%2.4f" % class_report[k]['hter'])
            f.close()

            f = open(filename_far, 'w')
            f.write("%2.4f" % class_report[k]['far'])
            f.close()

            f = open(filename_frr, 'w')
            f.write("%2.4f" % class_report[k]['frr'])
            f.close()

            np.save(filename_neg_scores, class_report[k]['neg'])

            np.save(filename_pos_scores, class_report[k]['pos'])

        print report

        return class_report

    def z_score_norm(self, train_, test_):

        train = train_.copy()
        test_norm = test_.copy()

        train_mean = train.mean(axis=0)
        train_std = train.std(axis=0, ddof=1)
        train_std[train_std == 0.] = 1.

        test_norm -= train_mean
        test_norm /= train_std

        return test_norm

    def testing(self, classifier, test_sets, scores, all_results):

        for i, test_set in enumerate(test_sets):
            print '-- classifying set {0}'.format(test_set['test_id'])

            acc = 0.
            outputs = {}

            classifier.test_set = test_set
            outputs = classifier.testing

            labels = outputs['gt']
            predicted_scores = outputs['predicted_scores']

            neg, pos = self._get_score_distributions(labels, predicted_scores)

            scores[test_set['test_id']] = {'neg': neg, 'pos': pos}
            all_results[test_set['test_id']] = {'gt': labels, 'predicted_scores': predicted_scores}

    def cross_dataset_protocol(self):
        print "cross_dataset_protocol()"

        # -- dataset A is used to train a classifier
        all_fnames_dataset_a = self.metafeat['all_fnames']
        all_labels_dataset_a = self.metafeat['all_labels']
        train_idxs_dataset_a = self.metafeat['train_idxs_for_cross']
        devel_idxs_dataset_a = self.metafeat['devel_idxs_for_cross']

        # -- dataset B is used for testing the built classifier
        all_fnames_dataset_b = self.metafeat_b['all_fnames']
        all_labels_dataset_b = self.metafeat_b['all_labels']
        test_idxs_dataset_b = self.metafeat_b['test_idxs']

        # -- loading features
        train_labels, train_data = self.__load_all_features(all_fnames_dataset_a, all_labels_dataset_a,
                                                            train_idxs_dataset_a)
        train_set = {'data': train_data, 'labels': train_labels}

        devel_labels, devel_data = self.__load_all_features(all_fnames_dataset_a, all_labels_dataset_a,
                                                            devel_idxs_dataset_a)
        devel_set = [{'data': devel_data, 'labels': devel_labels, 'test_id': 'devel'}]

        test_sets = []
        for test_id in test_idxs_dataset_b:
            if test_idxs_dataset_b[test_id].size:
                test_labels, test_data = self.__load_all_features(all_fnames_dataset_b, all_labels_dataset_b,
                                                                  test_idxs_dataset_b[test_id])
                test_sets += [{'data': test_data, 'labels': test_labels, 'test_id': test_id}]

        start = time.time()

        # --  training a classifier
        svm_clf = SVMClassifier()

        if self.output_model:
            svm_clf.output_path = self.output_model
        else:
            svm_clf.output_path = self.output_path

        svm_clf.train_set = train_set
        svm_clf.training()

        elapsed = time.strftime("%j,%H,%M,%S", time.gmtime((time.time() - start))).split(',')
        print "elapsed time: {0} days and {1}h{2}m{3}s".format(int(elapsed[0]) - 1, elapsed[1], elapsed[2], elapsed[3])

        scores = {}
        all_results = {}

        # --  testing the classifier
        self.testing(svm_clf, devel_set, scores, all_results)
        self.testing(svm_clf, test_sets, scores, all_results)

        class_report = self.performance_eval(scores, all_results, scores['devel'])

        return True

    def td_protocol(self):
        print "td_protocol()"

        try:
            os.makedirs(self.output_path)
        except Exception, e:
            pass

        all_fnames = self.metafeat['all_fnames']
        all_labels = self.metafeat['all_labels']
        train_idxs = self.metafeat['train_idxs']
        test_idxs = self.metafeat['test_idxs']

        train_labels, train_data = self.__load_all_features(all_fnames, all_labels, train_idxs)
        train_set = {'data': train_data, 'labels': train_labels}

        test_sets = []
        for test_id in test_idxs:
            if test_idxs[test_id].size:
                test_labels, test_data = self.__load_all_features(all_fnames, all_labels, test_idxs[test_id])
                test_sets += [{'data': test_data, 'labels': test_labels, 'test_id': test_id}]

        start = time.time()

        svm_clf = SVMClassifier()
        svm_clf.output_path = self.output_path
        svm_clf.train_set = train_set
        svm_clf.training()

        elapsed = time.strftime("%j,%H,%M,%S", time.gmtime((time.time() - start))).split(',')
        print "elapsed time: {0} days and {1}h{2}m{3}s".format(int(elapsed[0]) - 1, elapsed[1], elapsed[2], elapsed[3])

        scores = {}
        all_results = {}

        self.testing(svm_clf, test_sets, scores, all_results)

        class_report = self.performance_eval(scores, all_results)

        self.make_det_plots(scores, 'casia')

        return class_report['test']['hter']

    def tdt_protocol(self):
        print "tdt_protocol()"

        try:
            os.makedirs(self.output_path)
        except OSError:
            pass

        all_fnames = self.metafeat['all_fnames']
        all_labels = self.metafeat['all_labels']
        train_idxs = self.metafeat['train_idxs']
        devel_idxs = self.metafeat['devel_idxs']
        test_idxs = self.metafeat['test_idxs']

        train_labels, train_data = self.__load_all_features(all_fnames, all_labels, train_idxs)
        train_set = {'data': train_data, 'labels': train_labels}

        devel_labels, devel_data = self.__load_all_features(all_fnames, all_labels, devel_idxs)
        devel_set = [{'data': devel_data, 'labels': devel_labels, 'test_id': 'devel'}]

        test_sets = []
        for test_id in test_idxs:
            if test_idxs[test_id].size:
                test_labels, test_data = self.__load_all_features(all_fnames, all_labels, test_idxs[test_id])
                test_sets += [{'data': test_data, 'labels': test_labels, 'test_id': test_id}]

        scores = {}
        all_results = {}

        if 'svm' in self.algo:

            start = time.time()

            svm_clf = SVMClassifier()
            svm_clf.output_path = self.output_path
            svm_clf.train_set = train_set
            svm_clf.training()

            elapsed = time.strftime("%j,%H,%M,%S", time.gmtime((time.time() - start))).split(',')
            print "elapsed time: {0} days and {1}h{2}m{3}s".format(int(elapsed[0]) - 1, elapsed[1], elapsed[2],
                                                                   elapsed[3])

            self.testing(svm_clf, devel_set, scores, all_results)
            self.testing(svm_clf, test_sets, scores, all_results)

            # self._plot_det_curve(neg, pos, "det_curve_replay_teste_%d" % i)

        else:

            start = time.time()

            pls_clf = PLSClassifier()
            pls_clf.output_path = self.output_path
            pls_clf.train_set = train_set
            pls_clf.training()

            elapsed = time.strftime("%j,%H,%M,%S", time.gmtime((time.time() - start))).split(',')
            print "elapsed time: {0} days and {1}h{2}m{3}s".format(int(elapsed[0]) - 1, elapsed[1], elapsed[2],
                                                                   elapsed[3])

            self.testing(pls_clf, devel_set, scores, all_results)
            self.testing(pls_clf, test_sets, scores, all_results)

        class_report = self.performance_eval(scores, all_results, scores['devel'])

        self.make_det_plots(scores, 'replay')

        return class_report['test']['hter']

    def make_det_plots(self, scores, dataset):

        title_font = {'size': '18', 'color': 'black', 'weight': 'normal',
                      'verticalalignment': 'bottom'}  # Bottom vertical alignment for more space
        axis_font = {'size': '14'}
        fontsize_axis = 12

        if 'casia' in dataset:

            # -- plot attacks by quality
            fig1 = plt.figure(figsize=(8, 6), dpi=100)
            plt.clf()
            npoints = 300
            det(scores['test_scenario_7']['neg'], scores['test_scenario_7']['pos'], npoints, color=(0, 0, 0),
                marker='o', linestyle='-', linewidth=2, label='Overall test set')
            det(scores['test_scenario_1']['neg'], scores['test_scenario_1']['pos'], npoints, color=(0, 1, 0),
                marker='s', linestyle='-', linewidth=2, label='Low Quality Attacks')
            det(scores['test_scenario_2']['neg'], scores['test_scenario_2']['pos'], npoints, color=(0, 1, 1),
                marker='d', linestyle='-', linewidth=2, label='Normal Quality Attacks')
            det(scores['test_scenario_3']['neg'], scores['test_scenario_3']['pos'], npoints, color=(1, 0, 0),
                marker='*', linestyle='-', linewidth=2, label='High Quality Attacks')
            det_axis([5, 40, 5, 40])
            plt.xlabel('FRR (%)', **axis_font)
            plt.ylabel('FAR (%)', **axis_font)

            plt.xticks(size=fontsize_axis)
            plt.yticks(size=fontsize_axis)

            plt.legend()
            titulo = 'DET Curve'
            plt.title(titulo, **title_font)
            plt.grid(True)

            fig1.savefig('dets/det_plot_casia_qualy.pdf')

            # -- plot attacks by type
            fig2 = plt.figure(figsize=(8, 6), dpi=100)
            plt.clf()
            npoints = 300
            det(scores['test_scenario_7']['neg'], scores['test_scenario_7']['pos'], npoints, color=(0, 0, 0),
                marker='o', linestyle='-', linewidth=2, label='Overall test set')
            det(scores['test_scenario_4']['neg'], scores['test_scenario_4']['pos'], npoints, color=(0, 1, 0),
                marker='s', linestyle='-', linewidth=2, label='Warp Attacks')
            det(scores['test_scenario_5']['neg'], scores['test_scenario_5']['pos'], npoints, color=(0, 1, 1),
                marker='d', linestyle='-', linewidth=2, label='Cut Attack')
            det(scores['test_scenario_6']['neg'], scores['test_scenario_6']['pos'], npoints, color=(1, 0, 0),
                marker='*', linestyle='-', linewidth=2, label='Video Attack')
            det_axis([5, 40, 5, 40])

            plt.xlabel('FRR (%)', **axis_font)
            plt.ylabel('FAR (%)', **axis_font)

            plt.xticks(size=fontsize_axis)
            plt.yticks(size=fontsize_axis)

            plt.legend()
            titulo = "DET Curve"
            plt.title(titulo, **title_font)
            plt.grid(True)

            fig2.savefig("dets/det_plot_casia_type_attacks.pdf")

        elif 'replay' in dataset:

            # -- plot attacks by type
            fig1 = plt.figure(figsize=(8, 6), dpi=100)
            plt.clf()
            npoints = 300

            det(scores['test']['neg'], scores['test']['pos'], npoints, color=(0, 0, 0), marker='o', linestyle='-',
                linewidth=2, label='Overall test set')
            det(scores['highdef_attack']['neg'], scores['highdef_attack']['pos'], npoints, color=(0, 1, 0), marker='s',
                linestyle='-', linewidth=2, label='High-definition attacks')
            det(scores['mobile_attack']['neg'], scores['mobile_attack']['pos'], npoints, color=(0, 1, 1), marker='d',
                linestyle='-', linewidth=2, label='Mobile attacks')
            det(scores['print_attack']['neg'], scores['print_attack']['pos'], npoints, color=(1, 0, 0), marker='*',
                linestyle='-', linewidth=2, label='Print attacks')
            det_axis([1, 40, 1, 40])

            plt.xlabel('FRR (%)', **axis_font)
            plt.ylabel('FAR (%)', **axis_font)

            plt.xticks(size=fontsize_axis)
            plt.yticks(size=fontsize_axis)

            plt.legend()
            titulo = "DET Curve"
            plt.title(titulo, **title_font)
            plt.grid(True)

            fig1.savefig("dets/det_plot_replay_attacks.pdf")

            # -- plot attacks by type
            fig2 = plt.figure(figsize=(8, 6), dpi=100)
            plt.clf()
            npoints = 300
            det(scores['test']['neg'], scores['test']['pos'], npoints, color=(0, 0, 0), marker='o', linestyle='-',
                linewidth=2, label='Overall test set')
            det(scores['fixed_attack']['neg'], scores['fixed_attack']['pos'], npoints, color=(0, 1, 0), marker='s',
                linestyle='-', linewidth=2, label='Fixed-support attacks')
            det(scores['hand_attack']['neg'], scores['hand_attack']['pos'], npoints, color=(1, 0, 0), marker='d',
                linestyle='-', linewidth=2, label='Hand-based attacks')
            det_axis([1, 40, 1, 40])

            plt.xlabel('FRR (%)', **axis_font)
            plt.ylabel('FAR (%)', **axis_font)

            plt.xticks(size=fontsize_axis)
            plt.yticks(size=fontsize_axis)

            plt.legend()
            titulo = "DET Curve"
            plt.title(titulo, **title_font)
            plt.grid(True)

            fig2.savefig("dets/det_plot_replay_hand_fixed.pdf")

            # # -- plot attacks by type
            # fig3 = plt.figure(figsize=(8,8), dpi=100)
            # plt.clf()
            # npoints = 300
            # measure.plot.det(scores['test']['neg'], scores['test']['pos'], npoints, color=(0,0,0), marker = 'o', linestyle='-', linewidth=2, label='Overall test set')
            # measure.plot.det(scores['highdef_attack']['neg'], scores['highdef_attack']['pos'], npoints, color=(0,1,0), marker = 'x', linestyle='-', linewidth=2, label='High-definition attacks')
            # measure.plot.det(scores['mobile_attack']['neg'], scores['mobile_attack']['pos'], npoints, color=(0,1,1), marker = 'd', linestyle='-', linewidth=2, label='Mobile attacks')
            # measure.plot.det(scores['print_attack']['neg'], scores['print_attack']['pos'], npoints, color=(1,0,0), marker = '*', linestyle='-', linewidth=2, label='Print attacks')
            # measure.plot.det(scores['fixed_attack']['neg'], scores['fixed_attack']['pos'], npoints, color=(1,0,1), marker = 's', linestyle='-', linewidth=2, label='Fixed-support attacks')
            # measure.plot.det(scores['hand_attack']['neg'], scores['hand_attack']['pos'], npoints, color=(1,1,1), marker = 'D', linestyle='-', linewidth=2, label='Hand-based attacks')
            # measure.plot.det_axis([1, 40, 1, 40])
            # plt.xlabel('FRR (%)', fontsize=fontsize)
            # plt.ylabel('FAR (%)', fontsize=fontsize)
            # plt.legend()
            # titulo = "DET Curve"
            # plt.title(titulo, fontsize=fontsize)
            # plt.grid(True)
            # fig3.savefig("dets/det_plot_replay_all.pdf")

        return True

    def anova_protocol(self):

        try:
            os.makedirs(self.output_path)
        except Exception, e:
            pass

        all_fnames = self.metafeat['all_fnames']
        all_labels = self.metafeat['all_labels']
        train_idxs = self.metafeat['train_idxs']
        devel_idxs = self.metafeat['devel_idxs']
        test_idxs = self.metafeat['test_idxs']

        train_labels, train_data = self.__load_all_features(all_fnames, all_labels, train_idxs)
        train_set = {'data': train_data, 'labels': train_labels}

        devel_labels, devel_data = self.__load_all_features(all_fnames, all_labels, devel_idxs)
        devel_set = [{'data': devel_data, 'labels': devel_labels, 'test_id': 'devel'}]

        test_sets = []
        for test_id in test_idxs:
            if test_idxs[test_id].size:
                test_labels, test_data = self.__load_all_features(all_fnames, all_labels, test_idxs[test_id])
                test_sets += [{'data': test_data, 'labels': test_labels, 'test_id': test_id}]

        scores = {}
        all_results = {}

        try:
            if 'svm' in self.algo:

                start = time.time()

                svm_clf = SVMClassifier()
                svm_clf.output_path = self.output_path
                svm_clf.train_set = train_set
                svm_clf.training()

                elapsed = time.strftime("%j,%H,%M,%S", time.gmtime((time.time() - start))).split(',')
                print "elapsed time: {0} days and {1}h{2}m{3}s".format(int(elapsed[0]) - 1, elapsed[1], elapsed[2],
                                                                       elapsed[3])

                self.testing(svm_clf, devel_set, scores, all_results)
                self.testing(svm_clf, test_sets, scores, all_results)

                # self._plot_det_curve(neg, pos, "det_curve_replay_teste_%d" % i)

            else:

                start = time.time()

                pls_clf = PLSClassifier()
                pls_clf.output_path = self.output_path
                pls_clf.train_set = train_set
                pls_clf.training()

                elapsed = time.strftime("%j,%H,%M,%S", time.gmtime((time.time() - start))).split(',')
                print "elapsed time: {0} days and {1}h{2}m{3}s".format(int(elapsed[0]) - 1, elapsed[1], elapsed[2],
                                                                       elapsed[3])

                self.testing(pls_clf, devel_set, scores, all_results)
                self.testing(pls_clf, test_sets, scores, all_results)

        except Exception, e:
            print e
            sys.stdout.flush()
            return True

        class_report = self.performance_eval(scores, all_results, scores['devel'])

        return class_report['test']['hter']

    def run(self):
        print "run()"
