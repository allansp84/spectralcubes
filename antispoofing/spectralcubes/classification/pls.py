# -*- coding: utf-8 -*-

import os
import sys
import cPickle
import numpy as np

from joblib import Parallel, delayed

from sklearn.svm import SVC
from sklearn.cross_decomposition import PLSRegression

from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score

from multiprocessing import cpu_count

from copy import deepcopy

from scipy.sparse import *
from scipy import *


class PLSClassifier(object):

    """docstring for PLSClassifier"""

    def __init__(self, output_path = './', n_components = 2, scale_features=True, persist_model = True,
                    n_jobs = int(3 * cpu_count() / 4), seed = 42, debug=True):

        super(PLSClassifier, self).__init__()

        # private attributes
        self.__persist_model = persist_model
        self.__seed = seed
        self.__debug = debug
        self.__fname_model = os.path.join(output_path, 'pls.model')

        # public attributes
        self.output_path = output_path
        self.n_components = n_components
        self.scale_features = scale_features
        self.n_jobs = n_jobs
        self.train_set = {}
        self.test_set = {}
        self.model = []
        self.scaling_params = {}


    @property
    def train_set(self):
        return self.__train_set

    @train_set.setter
    def train_set(self, train_dict):
        try:
            assert isinstance(train_dict, dict)

            if train_dict:
                self.__train_set = {'data':[], 'labels': [], 'is_scaled': False}

                self.__train_set['data'] = train_dict['data']
                self.__train_set['labels'] = train_dict['labels']

                self.scaling_params = {}

        except Exception, e:
            raise e

    @property
    def test_set(self):
        return self.__test_set

    @test_set.setter
    def test_set(self, test_dict):
        try:
            assert isinstance(test_dict, dict)

            if test_dict:
                self.__test_set = {'data':[], 'labels': [], 'is_scaled': False}

                self.__test_set['data'] = test_dict['data']
                self.__test_set['labels'] = test_dict['labels']

        except Exception, e:
            raise e

    @property
    def model(self):
        return self.__model

    @model.setter
    def model(self, model_list):
        try:
            assert isinstance(model_list, list)
            self.__model = model_list
        except Exception, e:
            raise e

    @property
    def scaling_params(self):
        if self.__scaling_params:
            return self.__scaling_params
        else:
            self.__scaling_params = self.__compute_stats()
            return self.__scaling_params

    @scaling_params.setter
    def scaling_params(self, params):
        try:
            assert isinstance(params, dict)
            self.__scaling_params = params
        except Exception, e:
            raise e

    @property
    def scale_features(self):
        return self.__scale_features

    @scale_features.setter
    def scale_features(self, is_scale_features):
        try:
            assert isinstance(is_scale_features, bool)
            self.__scale_features = is_scale_features
        except Exception, e:
            raise e

    @property
    def output_path(self):
        return self.__output_path

    @output_path.setter
    def output_path(self, path):
        self.__output_path = os.path.abspath(path)
        self.__fname_model = os.path.join(self.__output_path, 'pls.model')


    def __save_model(self, model, fname):

        if self.__debug:
            print '-- saving model'

        try:
            os.makedirs(os.path.dirname(fname))
        except OSError:
            pass

        if not os.path.isfile(fname):
            fo = open(fname, 'wb')
            cPickle.dump(model,fo)
            fo.close()

    def __one_pls(self, cat):

        np.seterr(all='raise')

        lcat = np.zeros(self.train_set['labels'].size)

        lcat[self.train_set['labels'] != cat] = -1
        lcat[self.train_set['labels'] == cat] = +1

        pls = PLSRegression(n_components=2, scale=False)

        pls.fit(self.train_set['data'], lcat)

        return pls

    def __compute_stats(self):

        params = {}
        if self.train_set:
            mean = self.train_set['data'].mean(axis=0)
            std  = self.train_set['data'].std(axis=0, ddof=1)
            std[std==0.] = 1.
            params = {'mean':mean, 'std':std}

        else:
            sys.exit('Train set not found!')

        return params

    def __scaling_data(self, scaled_data):

        # scaled_data = deepcopy(data)

        params = self.scaling_params

        scaled_data -= params['mean']
        scaled_data /= params['std']

        return scaled_data

    def __load_persisted_model(self, fname):
        ''''''
        fo = open(fname, 'rb')
        model = cPickle.load(fo)
        fo.close()

        return model

    def __load_model(self):

        model = []

        if self.__debug:
            print '-- loading model'

        # -- load model persisted on disk
        if os.path.isfile(self.__fname_model):
            model = self.__load_persisted_model(self.__fname_model)
            if self.__debug:
                print '-- found in {0}'.format(self.__fname_model)

        # -- load model found in memory
        elif self.model:
            model = self.model
            if self.__debug:
                print '-- found in memory'

        # -- model does not generated yet
        else:
            if self.__debug:
                print '-- not found'

        self.model = model

    def training(self):

        print 'Training ...'

        # -- try loading model
        self.__load_model()

        # -- True if model does not generated yest
        if not self.model:

            if self.__debug:
                print '-- building model'

            # -- True if train set doesn't scaled yest and if is to scale it
            if((self.scale_features) and (not self.train_set['is_scaled'])):
                self.train_set['data'] = self.__scaling_data(self.train_set['data'])
                self.train_set['is_scaled'] = True

            model = []
            # -- compute model for eat category
            categories = np.unique(self.train_set['labels'])
            for cat in categories:
                model += [self.__one_pls(cat)]

            # -- True if is to persist the model
            if self.__persist_model:
                self.__save_model(model, self.__fname_model)

            self.model = model

        print '-- model parameters: ', self.model[0].get_params()

        if self.__debug:
            print '-- model is equal for two categories?', self.model[0].get_params() == self.model[1].get_params()
            print '-- finished'

    def testing(self):

        print 'Testing ...'

        outputs = {}

        self.__load_model()

        if self.model:

            if((self.scale_features)and(not self.test_set['is_scaled'])):
                self.test_set['data'] = self.__scaling_data(self.test_set['data'])
                self.test_set['is_scaled'] = True

            n_test = self.test_set['data'].shape[0]
            categories = np.unique(self.test_set['labels'])
            n_categories = len(categories)

            cat_index = {}
            predictions = np.empty((n_test, n_categories))

            model = self.model
            for icat, cat in enumerate(categories):
                cat_index[cat] = icat
                resps = model[icat].predict(self.test_set['data'])[:, 0]
                predictions[:, icat] = resps

            outputs = {'predictions': predictions, 'ground_truth': self.test_set['labels']}

        else:
            sys.exit('-- model not found! Please, execute training again!')

        return outputs

    def run(self):

        self.training()
        outputs = self.testing()

        return outputs
