# -*- coding: utf-8 -*-

import os
import sys
import argparse
from antispoofing.spectralcubes.utils import *
from antispoofing.spectralcubes.protocols import *
from antispoofing.spectralcubes.datasets import *


def running_anova(opts):
    dataset = registered_datasets[opts.dataset]
    data = dataset(opts.dataset_path)

    if opts.output_path is not None:
        data.output_path = opts.output_path

    data.output_path = os.path.join(data.output_path,
                                    str(data.__class__.__name__).lower(),
                                    "anova")

    if data.__class__.__name__.lower() == 'maskattack':
        print "Running KFoldANOVA Protocol"

        anova = KFoldANOVAProtocol(data, k_fold=opts.k)
        anova.execute_protocol()
    else:
        print "Running ANOVA Protocol"

        anova = ANOVAProtocol(data)
        anova.execute_protocol()

    # print "gathering *.result files"
    # results = retrieve_samples(data.output_path, "result")

    # print "saving *.csv files"
    # test_0_acc = creating_csv(results, data.output_path, "test_0", "acc")
    # test_0_auc = creating_csv(results, data.output_path, "test_0", "auc")
    # test_0_hter = creating_csv(results, data.output_path, "test_0", "hter")

    # test_1_acc = creating_csv(results, data.output_path, "test_1", "acc")
    # test_1_auc = creating_csv(results, data.output_path, "test_1", "auc")
    # test_1_hter = creating_csv(results, data.output_path, "test_1", "hter")

    # test_2_acc = creating_csv(results, data.output_path, "test_2", "acc")
    # test_2_auc = creating_csv(results, data.output_path, "test_2", "auc")
    # test_2_hter = creating_csv(results, data.output_path, "test_2", "hter")

    print "done!"


def running_cross_dataset(opts):
    print "Running CrossDataset Protocol"

    dataset = registered_datasets[opts.dataset_a]
    data_a = dataset(opts.dataset_path_a)

    data_a.output_path = os.path.join(opts.output_path,
                                      "cross_dataset/trainsets",
                                      str(data_a.__class__.__name__).lower())

    dataset = registered_datasets[opts.dataset_b]
    data_b = dataset(opts.dataset_path_b)

    if data_b.__class__.__name__.lower() == 'maskattack':

        data_b.output_path = os.path.join(opts.output_path,
                                          "cross_dataset/testsets",
                                          str(data_a.__class__.__name__).lower(),
                                          str(data_b.__class__.__name__).lower())
        anova = CrossKFoldProtocol(data_a, data_b, k_fold=opts.k)
        anova.execute_protocol()

    else:

        data_b.output_path = os.path.join(opts.output_path,
                                          "cross_dataset/testsets",
                                          str(data_a.__class__.__name__).lower(),
                                          str(data_b.__class__.__name__).lower())
        anova = CrossDatasetProtocol(data_a, data_b)
        anova.execute_protocol()


def running_intra_dataset_protocol(opts):
    dataset = registered_datasets[opts.dataset]
    data = dataset(opts.dataset_path)

    if data.__class__.__name__.lower() == 'replayattack':

        print "Running TDTProtocol"

        data.get_faceloc = opts.get_faceloc
        data.facelocations_path = opts.facelocations_path

        data.output_path = os.path.join(opts.output_path,
                                        str(data.__class__.__name__).lower(),
                                        'original_protocol')

        if opts.frame_numbers:
            str_nframes = '{0}_frames'.format(opts.frame_numbers)
            data.output_path = os.path.join(opts.output_path,
                                            str(data.__class__.__name__).lower(),
                                            'analize_frame_numbers',
                                            str_nframes)

        protocol = TDTProtocol(data, frame_numbers=opts.frame_numbers, only_face=opts.only_face)
        protocol.execute_protocol()

    elif data.__class__.__name__.lower() == 'casia':

        print "Running TDProtocol"

        data.get_faceloc = opts.get_faceloc
        data.facelocations_path = opts.facelocations_path

        data.output_path = os.path.join(opts.output_path,
                                        str(data.__class__.__name__).lower(),
                                        'original_protocol')

        if opts.frame_numbers:
            str_nframes = '{0}_frames'.format(opts.frame_numbers)
            data.output_path = os.path.join(opts.output_path,
                                            str(data.__class__.__name__).lower(),
                                            'analize_frame_numbers',
                                            str_nframes)

        protocol = TDProtocol(data, frame_numbers=opts.frame_numbers, only_face=opts.only_face)
        protocol.execute_protocol()

    elif data.__class__.__name__.lower() == 'maskattack':

        print "Running KFoldProtocol"
        data.get_faceloc = opts.get_faceloc
        data.facelocations_path = opts.facelocations_path

        data.output_path = os.path.join(opts.output_path,
                                        str(data.__class__.__name__).lower(),
                                        "original_protocol")

        if opts.frame_numbers:
            str_nframes = '{0}_frames'.format(opts.frame_numbers)
            data.output_path = os.path.join(opts.output_path,
                                            str(data.__class__.__name__).lower(),
                                            'analize_frame_numbers',
                                            str_nframes)

        protocol = KFoldProtocol(data, k_fold=opts.k, frame_numbers=opts.frame_numbers)
        protocol.execute_protocol()

    elif data.__class__.__name__.lower() == 'uvad':

        print "Running TDTProtocol"
        data.output_path = os.path.join(opts.output_path,
                                        str(data.__class__.__name__).lower(),
                                        'original_protocol')

        data.get_faceloc = opts.get_faceloc
        data.facelocations_path = opts.facelocations_path

        protocol = TDTProtocol(data, frame_numbers=opts.frame_numbers)
        protocol.execute_protocol()

    elif data.__class__.__name__.lower() == 'reduceduvad':

        print "Running TDTProtocol"
        data.output_path = os.path.join(opts.output_path,
                                        str(data.__class__.__name__).lower(),
                                        'original_protocol')

        data.get_faceloc = opts.get_faceloc
        data.facelocations_path = opts.facelocations_path

        protocol = TDTProtocol(data, frame_numbers=opts.frame_numbers)
        protocol.execute_protocol()

    else:
        print 'This Protocol is not defined for this dataset!'
        sys.exit(0)

        # results = retrieve_samples(data.output_path, 'result')
        # devel_auc = creating_csv(results, opts.output_path, 'devel', 'auc')


def running_kfold_protocol(opts):
    print "Running KFoldProtocol"

    dataset = registered_datasets[opts.dataset]
    data = dataset(opts.dataset_path)

    data.get_faceloc = opts.get_faceloc
    data.facelocations_path = opts.facelocations_path

    data.output_path = os.path.join(opts.output_path,
                                    str(data.__class__.__name__).lower(),
                                    "k_fold")

    if opts.frame_numbers:
        str_nframes = '{0}_frames'.format(opts.frame_numbers)
        data.output_path = os.path.join(opts.output_path,
                                        str(data.__class__.__name__).lower(),
                                        'analize_frame_numbers',
                                        str_nframes)

    protocol = KFoldProtocol(data, k_fold=opts.k, frame_numbers=opts.frame_numbers)
    protocol.execute_protocol()


def main():
    dataset_options = "Available datasets: "
    for k in sorted(registered_datasets.keys()):
        dataset_options += ("%s-%s  " % (k, registered_datasets[k].__name__))

    available_protocols = ["intra_dataset", "cross_dataset", "kfold"]

    parser = argparse.ArgumentParser(version='1.0', formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument("--protocol", type=str, default="intra_dataset", metavar="",
                        choices=available_protocols,
                        help="Protocol evaluation. " + "Allowed values are: " + ", ".join(available_protocols) +
                             " (default=%(default)s)\n\n")

    group_a = parser.add_argument_group('Arguments used in the intra-dataset protocol')
    group_a.add_argument("--dataset", type=int, default=0, metavar="", choices=range(len(registered_datasets)),
                        help=dataset_options + "(default=%(default)s)")

    group_a.add_argument("--dataset_path", type=str, metavar="", default='./datasets/replayattack',
                        help="Path to dataset (default=%(default)s)")

    group_a.add_argument("--output_path", type=str, metavar="", default='./working',
                        help="Path to output directory (default=%(default)s)")

    group_a.add_argument('--facelocations_path', type=str, metavar='', default='./datasets/replayattack/face-locations',
                        help='Path to directory containing the annotation to face locations (default=%(default)s)')

    group_b = parser.add_argument_group('Arguments used in the cross-dataset protocol')
    group_b.add_argument("--dataset_a", type=int, default=0, metavar="", choices=range(len(registered_datasets)),
                        help=dataset_options + "(default=%(default)s)")

    group_b.add_argument("--dataset_path_a", type=str, metavar="", default='./datasets/replayattack',
                        help="Path to dataset (default=%(default)s)")

    group_b.add_argument("--dataset_b", type=int, default=1, metavar="", choices=range(len(registered_datasets)),
                        help=dataset_options + "(default=%(default)s)")

    group_b.add_argument("--dataset_path_b", type=str, metavar="", default='./datasets/replayattack',
                        help="Path to dataset (default=%(default)s)")

    group_c = parser.add_argument_group('Other options')
    group_c.add_argument("--k", type=int, metavar="", default=10,
                        help="Number of fold considered in the k-fold cross validation (default=%(default)s)")

    group_c.add_argument("--anova", action='store_true',
                         help="Argument used to fit the best configuration of the method (default=%(default)s)")

    group_c.add_argument("--only_face", type=bool, default=False, metavar="",
                        help="Spoofing detection using only the face region (default=%(default)s)")

    group_c.add_argument("--get_faceloc", type=bool, default=False, metavar="",
                        help="Argument used in cases in that the face locations annotation do not available " +
                             "(default=%(default)s)")

    group_c.add_argument("--video_type", type=str, metavar="", default=VIDEO_TYPE,
                        help="Type of the videos to be loaded (default=%(default)s)")

    group_c.add_argument("--frame_numbers", type=int, metavar="", default=0,
                        help="Number of frames considered during execution of the method (default=%(default)s)\n\n")

    opts = parser.parse_args()

    if opts.anova:
        running_anova(opts)

    elif 'intra_dataset' in opts.protocol:
        running_intra_dataset_protocol(opts)

    elif 'cross_dataset' in opts.protocol:
        running_cross_dataset(opts)

    elif 'kfold' in opts.protocol:
        running_kfold_protocol(opts)

    else:
        print 'Protocol not found!'


if __name__ == "__main__":
    start = get_time()
    main()
    total_time_elapsed(start, get_time())
