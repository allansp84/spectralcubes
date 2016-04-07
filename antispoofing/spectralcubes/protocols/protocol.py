# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod


class Protocol(object):

    """docstring for Protocol"""

    __metaclass__ = ABCMeta

    @abstractmethod
    def build_search_space(self):
        """ docstring """
        pass

    @abstractmethod
    def extract_low_level_features(self):
        """ docstring """
        pass

    @abstractmethod
    def extract_mid_level_features(self):
        """ docstring """
        pass

    @abstractmethod
    def classification(self):
        """ docstring """
        pass

    def execute_protocol(self):
        """ docstring """

        print "building search space ..."
        self.build_search_space()

        print "computing low level features ..."
        self.extract_low_level_features()

        print "computing mid level features ..."
        self.extract_mid_level_features()

        print "building classifiers ..."
        self.classification()
