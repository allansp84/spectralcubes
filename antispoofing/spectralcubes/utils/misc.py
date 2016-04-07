# -*- coding: utf-8 -*-

# from antispoofing.spectralcubes.utils.constants import *
import os
import sys
import operator
import datetime
import numpy as np
import itertools as it
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from glob import glob
from multiprocessing import Pool, cpu_count


def modification_date(filename):
    t = os.path.getmtime(filename)
    return datetime.datetime.fromtimestamp(t)


def get_time():
    return datetime.datetime.now()


def total_time_elapsed(start, finish):
    elapsed = finish - start

    total_seconds = int(elapsed.total_seconds())
    total_minutes = int(total_seconds // 60)
    hours = int(total_minutes // 60)
    minutes = int(total_minutes % 60)
    seconds = int(round(total_seconds % 60))

    return "{0:02d}+{1:02d}:{2:02d}:{3:02d}".format(elapsed.days, hours, minutes, seconds)


def start_process():
    pass


def do_something(d):
    result = d.run()
    return result


def progressbar(name, i, total, bar_len=20):
    percent = float(i) / total

    sys.stdout.write("\r")
    progress = ""
    for i in range(bar_len):
        if i < int(bar_len * percent):
            progress += "="
        else:
            progress += " "
    sys.stdout.write("%s: [ %s ] %.2f%%" % (name, progress, percent * 100))
    sys.stdout.flush()


class RunInParallel(object):

    def __init__(self, tasks, n_proc=(cpu_count() - 1)):
        self._pool = Pool(initializer=start_process, processes=n_proc)
        self._tasks = tasks

    def run(self):
        pool_outs = self._pool.map_async(do_something, self._tasks)
        self._pool.close()
        self._pool.join()

        try:
            work_done = [out for out in pool_outs.get() if out]
            assert (len(work_done)) == len(self._tasks)

        except AssertionError:
            sys.stderr.write("ERROR: some objects could not be processed!\n")
            sys.exit(1)


def acc_threshold(neg_scores, pos_scores, threshold):

    neg_scores = np.array(neg_scores)
    pos_scores = np.array(pos_scores)

    n_scores = float(len(neg_scores) + len(pos_scores))

    n_fa = (neg_scores >= threshold).sum()
    n_fr = (pos_scores < threshold).sum()

    return (1. - ((n_fa + n_fr) / n_scores)) * 100.


def eer_threshold(negatives, positives):

    n_points = 100
    threshold = 0.0
    # eer = 0.0
    delta_min = 999.0

    if negatives.size == 0:
        return 0.

    if positives.size == 0:
        return 0.

    lower_bound = min(np.min(negatives), np.min(negatives))
    upper_bound = max(np.max(negatives), np.max(negatives))

    steps = float((upper_bound - lower_bound)/(n_points-1))

    thr = lower_bound

    for pt in xrange(n_points):

        far, frr = farfrr(negatives, positives, thr)

        if abs(far - frr) < delta_min:
            delta_min = abs(far - frr)
            # eer = (far + frr) / 2.0
            threshold = thr

        thr += steps

    return threshold


def farfrr(negatives, positives, threshold):

    if negatives.size != 0:
        far = (np.array(negatives) >= threshold).mean()
    else:
        far = 1.

    if positives.size != 0:
        frr = (np.array(positives) < threshold).mean()
    else:
        frr = 1.

    return far, frr


def calc_hter(neg_devel, pos_devel, neg_test, pos_test):

    # calculate threshold upon eer point
    # threshold = bob.measure.eer_threshold(neg_devel, pos_devel)
    threshold = eer_threshold(neg_devel, pos_devel)

    # calculate far and frr
    # far, frr = bob.measure.farfrr(neg_test, pos_test, threshold)
    far, frr = farfrr(neg_test, pos_test, threshold)

    far *= 100.
    frr *= 100.

    hter = ((far + frr) / 2.)

    return threshold, far, frr, hter


def ppndf_over_array(cum_prob):
    split = 0.42
    a_0 = 2.5066282388
    a_1 = -18.6150006252
    a_2 = 41.3911977353
    a_3 = -25.4410604963
    b_1 = -8.4735109309
    b_2 = 23.0833674374
    b_3 = -21.0622410182
    b_4 = 3.1308290983
    c_0 = -2.7871893113
    c_1 = -2.2979647913
    c_2 = 4.8501412713
    c_3 = 2.3212127685
    d_1 = 3.5438892476
    d_2 = 1.6370678189
    eps = 2.2204e-16

    n_rows, n_cols = cum_prob.shape

    norm_dev = np.zeros((n_rows, n_cols))
    for irow in xrange(n_rows):
        for icol in xrange(n_cols):

            prob = cum_prob[irow, icol]
            if prob >= 1.0:
                prob = 1-eps
            elif prob <= 0.0:
                prob = eps

            q = prob - 0.5
            if abs(prob-0.5) <= split:
                r = q * q
                pf = q * (((a_3 * r + a_2) * r + a_1) * r + a_0)
                pf /= (((b_4 * r + b_3) * r + b_2) * r + b_1) * r + 1.0

            else:
                if q > 0.0:
                    r = 1.0-prob
                else:
                    r = prob

                r = np.sqrt((-1.0) * np.log(r))
                pf = (((c_3 * r + c_2) * r + c_1) * r + c_0)
                pf /= ((d_2 * r + d_1) * r + 1.0)

                if q < 0:
                    pf *= -1.0

            norm_dev[irow, icol] = pf

    return norm_dev


def ppndf(prob):
    split = 0.42
    a_0 = 2.5066282388
    a_1 = -18.6150006252
    a_2 = 41.3911977353
    a_3 = -25.4410604963
    b_1 = -8.4735109309
    b_2 = 23.0833674374
    b_3 = -21.0622410182
    b_4 = 3.1308290983
    c_0 = -2.7871893113
    c_1 = -2.2979647913
    c_2 = 4.8501412713
    c_3 = 2.3212127685
    d_1 = 3.5438892476
    d_2 = 1.6370678189
    eps = 2.2204e-16

    if prob >= 1.0:
        prob = 1-eps
    elif prob <= 0.0:
        prob = eps

    q = prob - 0.5
    if abs(prob-0.5) <= split:
        r = q * q
        pf = q * (((a_3 * r + a_2) * r + a_1) * r + a_0)
        pf /= (((b_4 * r + b_3) * r + b_2) * r + b_1) * r + 1.0

    else:
        if q > 0.0:
            r = 1.0-prob
        else:
            r = prob

        r = np.sqrt((-1.0) * np.log(r))
        pf = (((c_3 * r + c_2) * r + c_1) * r + c_0)
        pf /= ((d_2 * r + d_1) * r + 1.0)

        if q < 0:
            pf *= -1.0

    return pf


def compute_det(negatives, positives, npoints):

    # delta_min = 999.0

    lower_bound = min(np.min(negatives), np.min(negatives))
    upper_bound = max(np.max(negatives), np.max(negatives))

    steps = float((upper_bound - lower_bound)/(npoints-1))

    threshold = lower_bound
    curve = []
    for pt in xrange(npoints):

        far, frr = farfrr(negatives, positives, threshold)

        curve.append([far, frr])
        threshold += steps

    curve = np.array(curve)

    return ppndf_over_array(curve.T)


def det(negatives, positives, n_points, axis_font_size='x-small', **kwargs):

    # these are some constants required in this method
    desired_ticks = [
        '0.00001', '0.00002', '0.00005',
        '0.0001', '0.0002', '0.0005',
        '0.001', '0.002', '0.005',
        '0.01', '0.02', '0.05',
        '0.1', '0.2', '0.4', '0.6', '0.8', '0.9',
        '0.95', '0.98', '0.99',
        '0.995', '0.998', '0.999',
        '0.9995', '0.9998', '0.9999',
        '0.99995', '0.99998', '0.99999',
    ]

    desired_labels = [
        '0.001', '0.002', '0.005',
        '0.01', '0.02', '0.05',
        '0.1', '0.2', '0.5',
        '1', '2', '5',
        '10', '20', '40', '60', '80', '90',
        '95', '98', '99',
        '99.5', '99.8', '99.9',
        '99.95', '99.98', '99.99',
        '99.995', '99.998', '99.999',
    ]

    curve = compute_det(negatives, positives, n_points)

    output_plot = plt.plot(curve[0, :], curve[1, :], **kwargs)

    # -- now the trick: we must plot the tick marks by hand using the PPNDF method
    p_ticks = [ppndf(float(v)) for v in desired_ticks]

    # -- and finally we set our own tick marks
    ax = plt.gca()
    ax.set_xticks(p_ticks)
    ax.set_xticklabels(desired_labels, size=axis_font_size)
    ax.set_yticks(p_ticks)
    ax.set_yticklabels(desired_labels, size=axis_font_size)

    return output_plot


def det_axis(v, **kwargs):

    tv = list(v)
    tv = [ppndf(float(k)/100) for k in tv]
    ret = plt.axis(tv, **kwargs)

    return ret


def plot_score_distributions(threshold, neg_devel, pos_devel, neg_test, pos_test, filename='score_dist.png'):

    plt.clf()
    plt.figure(1)
    plt.subplot(211)
    plt.title("Score distributions (Deve set)")
    n, bins, patches = plt.hist(neg_devel, bins=25, normed=1, histtype='bar', label='Negative class')
    na, bins_a, patches_a = plt.hist(pos_devel, bins=25, normed=1, histtype='bar', label='Positive class')

    # add a line showing the expected distribution
    y = mlab.normpdf(bins, np.mean(neg_devel), np.std(neg_devel))
    plt.plot(bins, y, 'k--', linewidth=1.5)
    y = mlab.normpdf(bins_a, np.mean(pos_devel), np.std(pos_devel))
    plt.plot(bins_a, y, 'k--', linewidth=1.5)
    plt.axvline(x=threshold, linewidth=2, color='blue')
    plt.legend()

    plt.subplot(212)
    plt.title("Score distributions (Test set)")
    n, bins, patches = plt.hist(neg_test, bins=25, normed=1, facecolor='green', alpha=0.5, histtype='bar',
                                label='Negative class')
    na, bins_a, patches_a = plt.hist(pos_test, bins=25, normed=1, facecolor='red', alpha=0.5, histtype='bar',
                                     label='Positive class')

    # add a line showing the expected distribution
    y = mlab.normpdf(bins, np.mean(neg_test), np.std(neg_test))
    plt.plot(bins, y, 'k--', linewidth=1.5)
    y = mlab.normpdf(bins_a, np.mean(pos_test), np.std(pos_test))
    plt.plot(bins_a, y, 'k--', linewidth=1.5)
    plt.axvline(x=threshold, linewidth=2, color='blue')
    plt.legend()

    current_dir = os.getcwd()
    output = '{0}/{1}.png'.format(current_dir, filename)
    plt.savefig(output)


def retrieve_samples(input_path, file_type):
    dir_names = []
    for root, subFolders, files in os.walk(input_path):
        for f in files:
            if f[-len(file_type):] == file_type:
                dir_names += [root]
                break

    dir_names = sorted(dir_names)

    fnames = []
    for dir_name in dir_names:
        dir_fnames = sorted(glob(os.path.join(input_path, dir_name, '*.' + file_type)))
        fnames += dir_fnames

    return fnames


def grouper(n, iterable, fill_value=None):
    args = [iter(iterable)] * n
    return it.izip_longest(fillvalue=fill_value, *args)


def mosaic(w, imgs):

    imgs = iter(imgs)
    img0 = imgs.next()
    pad = np.zeros_like(img0)
    imgs = it.chain([img0], imgs)
    rows = grouper(w, imgs, pad)

    return np.vstack(map(np.hstack, rows))


def replace_from_list(string_list, old_str, new_str):
    return map(lambda x: str.replace(x, old_str, new_str), string_list)


def creating_csv(f_results, output_path, test_set, measure):
    f_measure = [f for f in f_results if ((test_set in f) and (measure in f))]

    configs, values = [], []
    for f_m in f_measure:
        configs += [os.path.dirname(os.path.relpath(f_m, output_path))]
        values += [float(open(f_m, 'r').readline())]

    configs_orin = configs
    configs = replace_from_list(configs, '/', ',')

    configs = replace_from_list(configs, test_set, '')
    configs = replace_from_list(configs, 'classifiers,', '')

    configs = replace_from_list(configs, '300,', '')

    configs = replace_from_list(configs, 'realization_1', 'R1')
    configs = replace_from_list(configs, 'realization_2', 'R2')
    configs = replace_from_list(configs, 'realization_3', 'R3')

    configs = replace_from_list(configs, 'centerframe', 'C')
    configs = replace_from_list(configs, 'wholeframe', 'W')

    configs = replace_from_list(configs, 'dftenergymag', 'ME')
    configs = replace_from_list(configs, 'dftentropymag', 'MS')
    configs = replace_from_list(configs, 'dftenergyphase', 'PE')
    configs = replace_from_list(configs, 'dftentropyphase', 'PS')

    configs = replace_from_list(configs, 'kmeans', 'K')
    configs = replace_from_list(configs, 'random', 'R')

    configs = replace_from_list(configs, 'class_based', 'D')
    configs = replace_from_list(configs, 'unified', 'S')

    configs = replace_from_list(configs, 'svm', 'SVM')
    configs = replace_from_list(configs, 'pls', 'PLS')

    configs = replace_from_list(configs, 'energy_phase', 'PE')
    configs = replace_from_list(configs, 'entropy_phase', 'PH')
    configs = replace_from_list(configs, 'energy_mag', 'ME')
    configs = replace_from_list(configs, 'entropy_mag', 'MH')
    configs = replace_from_list(configs, 'mutualinfo_phase', 'PMI')
    configs = replace_from_list(configs, 'mutualinfo_mag', 'MMI')
    configs = replace_from_list(configs, 'correlation_phase', 'PC')
    configs = replace_from_list(configs, 'correlation_mag', 'MC')

    reverse = False if 'hter' in measure else True

    results = sorted(zip(configs, values), key=operator.itemgetter(1), reverse=reverse)

    fname = "{0}/{1}.{2}.csv".format(output_path, test_set, measure)
    f_csv = open(fname, 'w')
    f_csv.write("N,LGF,M,CS,SDD,DS,CP,C,%s\n" % str(measure).upper())
    for r in results:
        f_csv.write("%s%s\n" % (r[0], r[1]))
    f_csv.close()

    print fname, results[:4]

    return sorted(zip(configs_orin, values), key=operator.itemgetter(1), reverse=reverse)

# def padwithtens(vector, pad_width):
#     vector[:pad_width[0]] = 10
#     vector[-pad_width[1]:] = 10
#     return vector
#
#
