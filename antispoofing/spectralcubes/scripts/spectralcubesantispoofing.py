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


def running_original_protocol(opts):
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


def running_misc_protocol(opts):
    dataset = registered_datasets[opts.dataset]
    data = dataset(opts.dataset_path)

    if opts.output_path is not None:
        data.output_path = opts.output_path

    data.output_path = os.path.join(data.output_path,
                                    str(data.__class__.__name__).lower(),
                                    "misc_protocol")

    protocol = MISCProtocol(data, opts.sample)
    protocol.execute_protocol()

    # results = retrieve_samples(data.output_path, 'result')
    # devel_auc = creating_csv(results, opts.output_path, 'devel', 'auc')


def running_for_one_sample(opts):
    print "Running AnalizeProtocol"

    dataset = registered_datasets[opts.dataset]
    data = dataset(opts.dataset_path)

    data.facelocations_path = opts.facelocations_path

    data.output_path = os.path.join(opts.output_path,
                                    str(data.__class__.__name__).lower(),
                                    "analize")

    anova = AnalizeProtocol(data, opts.sample)
    anova.execute_protocol()


def main():

    dataset_options = "Available datasets: "
    for k in sorted(registered_datasets.keys()):
        dataset_options += ("%s-%s  " % (k, registered_datasets[k].__name__))

    parser = argparse.ArgumentParser(version='1.0')
    parser.add_argument("--dataset", type=int, default=0, choices=range(len(registered_datasets)), help=dataset_options)
    parser.add_argument("--dataset_a", type=int, default=0, choices=range(len(registered_datasets)), help="")
    parser.add_argument("--dataset_b", type=int, default=0, choices=range(len(registered_datasets)), help="")

    parser.add_argument("--dataset_path", type=str, metavar="str", default='', help="<dataset_path>")
    parser.add_argument("--dataset_path_a", type=str, metavar="str", default='', help="")
    parser.add_argument("--dataset_path_b", type=str, metavar="str", default='', help="")

    parser.add_argument("--output_path", type=str, metavar="str", default='./working', help="<output_path>")

    parser.add_argument("--anova_protocol", action='store_true')
    parser.add_argument("--original_protocol", action='store_true')
    parser.add_argument("--cross_dataset", action='store_true')
    parser.add_argument("--misc_protocol", action='store_true')
    parser.add_argument("--kfold_protocol", action='store_true')

    parser.add_argument("--k", type=int, default=10, help="")
    parser.add_argument("--sample", type=str, metavar="str", default='client020', help="(default=%(default)s)")
    parser.add_argument("--analize", action='store_true')
    parser.add_argument("--only_face", action='store_true')
    parser.add_argument("--video_type", type=str, metavar="str", default=VIDEO_TYPE,
                        help="Type of the videos to be loaded (default=%(default)s)")
    parser.add_argument("--frame_numbers", type=int, metavar="int", default=0, help="(default=%(default)s)")
    parser.add_argument('--facelocations_path', type=str, metavar='str', default='', help='<facelocations_path>')

    parser.add_argument("--get_faceloc", action='store_true')

    opts = parser.parse_args()

    if opts.analize:
        if opts.sample is None:
            print 'ERROR: --analize should be used with --sample'
            sys.exit(0)
        else:
            running_for_one_sample(opts)

    elif opts.original_protocol:
        running_original_protocol(opts)

    elif opts.kfold_protocol:
        running_kfold_protocol(opts)

    elif opts.misc_protocol:
        running_misc_protocol(opts)

    elif opts.cross_dataset:
        running_cross_dataset(opts)

    elif opts.anova_protocol:
        running_anova(opts)

    else:
        print 'Protocol not found!'


if __name__ == "__main__":
    start = get_time()
    main()
    total_time_elapsed(start, get_time())
