# -*- coding: utf-8 -*-

import os
from abc import ABCMeta
from abc import abstractmethod


class Dataset(object):

    """docstring for Dataset"""

    __metaclass__ = ABCMeta

    def __init__(self, dataset_path, output_path):

        self.__dataset_path = ""
        self.__output_path = ""

        self.dataset_path = dataset_path
        self.output_path = output_path
        # self.file_types = file_types

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

    # @property
    # def file_types(self):
    #     return self.__file_types
    #
    # @file_types.setter
    # def file_types(self, filetypes):
    #     self.__file_types = filetypes

    # @file_types.setter
    # def file_types(self, filetypes):
    #     extensions = []
    #     for ext in filetypes:
    #         if not "*." in ext:
    #             ext = "*.%s" % ext
    #         extensions += [ext]
    #     self.__file_types = extensions

    @abstractmethod
    def _build_meta(self, inpath, filetypes):
        """ docstring """

    @staticmethod
    def _list_dirs(rootpath, filetype):
        folders = []

        for root, dirs, files in os.walk(rootpath):
            for f in files:
                if filetype in os.path.splitext(f)[1]:
                    folders += [os.path.relpath(root, rootpath)]
                    break

        return folders

    # def _list_dirs(self, rootpath, filetypes):
    #     folders = []

    #     for root, dirs, files in os.walk(rootpath):
    #         for extension in filetypes:
    #             for filename in fnmatch.filter(files, extension):
    #                 folders += [os.path.join(root, filename)]

    #         # for f in files:
    #         #     if filetype in os.path.splitext(f)[1]:
    #         #         folders += [os.path.relpath(root, inpath)]
    #         #         break

    #     return folders

    # @property
    # def metainfo(self):
    #     try:
    #         return self.__metainfo
    #     except AttributeError:
    #         self.__metainfo = self._build_meta(self.dataset_path, self.file_types)
    #         return self.__metainfo

    # def metainfo_feats(self, output_path, file_types):
    #     return self._build_meta(output_path, file_types)
