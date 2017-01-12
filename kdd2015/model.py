# -*- coding: utf-8 -*-
import math

from keras.models import Sequential, Graph
from keras.layers.core import Dense, MaxoutDense, Dropout, Activation, TimeDistributedDense, Flatten, Merge
from keras.layers.noise import GaussianNoise
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU, SimpleRNN, SimpleDeepRNN, JZS1, JZS2, JZS3
from keras.layers.advanced_activations import PReLU
from keras.layers.normalization import BatchNormalization, LRN2D
from keras.optimizers import SGD
from keras.regularizers import l2, activity_l2, l1l2
from keras.constraints import nonneg
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import SGDClassifier, LogisticRegression, RidgeClassifier, ElasticNetCV, BayesianRidge, LogisticRegression, LogisticRegressionCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
import xgboost as xgb
import numpy as np
from scipy.stats import itemfreq


def build_gmodel():
    graph = Graph()
    graph.add_input(name='time_series', ndim=3)
    graph.add_node(JZS3(63, 40), name='rnn1', input='time_series')
    # # graph.add_node(JZS3(63, 40), name='rnn2', input='time_series')
    # # graph.add_node(JZS3(63, 40), name='rnn3', input='time_series')
    # # graph.add_node(JZS3(63, 40), name='rnn4', input='time_series')
    graph.add_node(Dense(40, 40), name='dense1', input='rnn1')
    # # graph.add_node(Dense(40, 40), name='dense2', input='rnn2')
    # # graph.add_node(Dense(40, 40), name='dense3', input='rnn3')
    # # graph.add_node(Dense(40, 40), name='dense4', input='rnn4')
    graph.add_node(MaxoutDense(40, 20, nb_feature=4), name='maxout1', input='dense1')
    # # graph.add_node(MaxoutDense(40, 80, nb_feature=4), name='maxout2', input='dense2')
    # # graph.add_node(MaxoutDense(40, 80, nb_feature=4), name='maxout3', input='dense3')
    # # graph.add_node(MaxoutDense(40, 80, nb_feature=4), name='maxout4', input='dense4')
    graph.add_node(Dropout(0.5), name='dropout1', input='maxout1')
    # # graph.add_node(Dropout(0.5), name='dropout2', input='maxout2')
    # # graph.add_node(Dropout(0.5), name='dropout3', input='maxout3')
    # # graph.add_node(Dropout(0.5), name='dropout4', input='maxout4')
    # graph.add_node(Dense(320, 160, activation='softmax'), name='merge', inputs=['dropout1', 'dropout2', 'dropout3', 'dropout4'], merge_mode='concat')
    # graph.add_node(MaxoutDense(160, 160, nb_feature=4), name='merge_maxout', input='merge')
    # graph.add_node(Dropout(0.5), name='merge_dropout', input='merge_maxout')
    graph.add_node(Dense(20, 1, activation='sigmoid'), name='out_dense', input='dropout1')

    graph.add_input(name='enrollment', ndim=2)
    graph.add_node(GaussianNoise(0.05), name='noise', input='enrollment')
    # graph.add_node(Dense(54, 64), name='mlp_dense', inputs=['enrollment', 'out_dense'])
    graph.add_node(Dense(53, 64), name='mlp_dense', input='noise')
    graph.add_node(MaxoutDense(64, 64, nb_feature=4), name='mlp_maxout', input='mlp_dense')
    graph.add_node(Dropout(0.5), name='mlp_dropout', input='mlp_maxout')
    # graph.add_node(Dense(32, 16), name='mlp_dense2', input='mlp_dropout')
    graph.add_node(Dense(64, 1, activation='sigmoid'), name='mlp_dense3', input='mlp_dropout')
    graph.add_node(Dense(4, 2, activation='softmax'), name='mlp_dense4', inputs=['mlp_dense3', 'out_dense', 'mlp_dense3', 'out_dense'], merge_mode='concat')
    graph.add_node(Dense(2, 1, activation='sigmoid'), name='mlp_dense5', input='mlp_dense4')

    # graph.add_node(Dense(2, 1), name='my_output', inputs=['mlp_dense2', 'out_dense'], merge_mode='concat')

    graph.add_output(name='output', input='mlp_dense5')

    graph.compile('adam', {'output': 'binary_crossentropy'})

    return graph


def build_model0():
    # max_features = 8

    # model_left = Sequential()
    # model_left.add(GRU(
    #     33, 512,
    #     # activation='sigmoid',
    #     # inner_activation='hard_sigmoid',
    #     # return_sequences=True
    # ))
    # model_left.add(Activation('sigmoid'))
    # # model_left.add(Dense(512, 256, activation='sigmoid'))
    # model_right = Sequential()
    # model_right.add(GRU(
    #     33, 512,
    #     # activation='sigmoid',
    #     # inner_activation='hard_sigmoid',
    #     # return_sequences=True
    # ))
    # model_right.add(Activation('sigmoid'))
    # model_right.add(Dense(512, 256, activation='sigmoid'))

    # graph = Graph()
    # graph.add_input(name='input', ndim=3)
    # graph.add_node(Dropout(0.5), name='dropout1', input='input')
    # graph.add_node(J2S3(63, 64), name='rnn1', input='dropout1')
    # graph.add_node(Dropout(0.5), name='dropout1_1', input='rnn1')
    # graph.add_node(Dense(64, 64), name='dense1', input='dropout1_1')
    # graph.add_node(Dense(64, 1), name='dense1_1', input='dense1')

    # graph.add_node(Dropout(0.5), name='dropout2', input='input')
    # graph.add_node(J2S3(63, 64), name='rnn2', input='dropout2')
    # graph.add_node(Dropout(0.5), name='dropout2_1', input='rnn2')
    # graph.add_node(Dense(64, 64), name='dense2', input='dropout2_1')
    # graph.add_node(Dense(64, 1), name='dense2_1', input='dense2')

    # graph.add_output(name='output', inputs=['dense1_1', 'dense2_1'], merge_mode='sum')
    # graph.compile('adam', {'output': 'binary_crossentropy'})

    rnn1 = Sequential()
    rnn1.add(JZS3(63, 40))
    rnn1.add(Dense(40, 40))
    # rnn1.add(PReLU((40,)))
    rnn1.add(MaxoutDense(40, 80, nb_feature=4))
    rnn1.add(Dropout(0.5))

    rnn2 = Sequential()
    rnn2.add(JZS3(63, 40))
    rnn2.add(Dense(40, 40))
    # rnn2.add(PReLU((40,)))
    rnn2.add(MaxoutDense(40, 80, nb_feature=4))
    rnn2.add(Dropout(0.5))

    rnn3 = Sequential()
    rnn3.add(JZS3(63, 40))
    rnn3.add(Dense(40, 40))
    # rnn3.add(PReLU((40,)))
    rnn3.add(MaxoutDense(40, 80, nb_feature=4))
    rnn3.add(Dropout(0.5))

    rnn = Sequential()
    rnn.add(Merge([rnn1, rnn2, rnn3], mode='concat'))
    rnn.add(Dense(240, 120, activation='softmax'))
    rnn.add(MaxoutDense(120, 120, nb_feature=4))
    rnn.add(Dropout(0.5))
    rnn.add(Dense(120, 1, activation='sigmoid'))

    rnn.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        class_mode="binary"
    )

    return rnn


def build_model():
    # max_features = 8

    # model_left = Sequential()
    # model_left.add(GRU(
    #     33, 512,
    #     # activation='sigmoid',
    #     # inner_activation='hard_sigmoid',
    #     # return_sequences=True
    # ))
    # model_left.add(Activation('sigmoid'))
    # # model_left.add(Dense(512, 256, activation='sigmoid'))
    # model_right = Sequential()
    # model_right.add(GRU(
    #     33, 512,
    #     # activation='sigmoid',
    #     # inner_activation='hard_sigmoid',
    #     # return_sequences=True
    # ))
    # model_right.add(Activation('sigmoid'))
    # model_right.add(Dense(512, 256, activation='sigmoid'))

    rnn = Sequential()
    rnn.add(Dropout(0.5))
    rnn.add(JZS3(63, 64))
    # rnn.add(BatchNormalization((32,)))
    # rnn.add(MaxoutDense(32, 1, nb_feature=64))
    # rnn.add(BatchNormalization((1,)))
    rnn.add(Dropout(0.5))
    rnn.add(Dense(64, 1, activation='sigmoid'))

    mlp1 = Sequential()
    # mlp.add(Dropout(0.5))
    mlp1.add(Dense(53, 64))
    mlp1.add(PReLU((64,)))
    mlp1.add(BatchNormalization((64,)))
    mlp1.add(MaxoutDense(64, 1, nb_feature=256))
    # mlp1.add(Dropout(0.5))

    mlp2 = Sequential()
    # mlp.add(Dropout(0.5))
    mlp2.add(Dense(53, 64))
    mlp2.add(PReLU((64,)))
    mlp2.add(BatchNormalization((64,)))
    mlp2.add(MaxoutDense(64, 1, nb_feature=256))
    # mlp2.add(Dropout(0.5))

    mlp3 = Sequential()
    # mlp.add(Dropout(0.5))
    mlp3.add(Dense(53, 64))
    mlp3.add(PReLU((64,)))
    mlp3.add(BatchNormalization((64,)))
    mlp3.add(MaxoutDense(64, 1, nb_feature=256))

    mlp4 = Sequential()
    # mlp.add(Dropout(0.5))
    mlp4.add(Dense(53, 64))
    mlp4.add(PReLU((64,)))
    mlp4.add(BatchNormalization((64,)))
    mlp4.add(MaxoutDense(64, 1, nb_feature=256))

    mlp = Sequential()
    mlp.add(Merge([mlp1, mlp2, mlp3, mlp4], mode='concat'))

    mlp.add(Dense(4, 1, activation='sigmoid'))
    # mlp.add(PReLU((32,)))
    # mlp.add(BatchNormalization((32,)))
    # mlp.add(MaxoutDense(32, 16, nb_feature=128))
    # mlp.add(BatchNormalization((16,)))
    # mlp.add(Dropout(0.125))

    model = Sequential()
    model.add(Merge([rnn, mlp], mode='concat'))

    model.add(Dense(2, 1, activation='sigmoid'))
    # model.add(Dropout(0.25))
    # model.add(Dense(2, 1, activation='sigmoid'))
    # model.add(PReLU((1,)))

    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        class_mode="binary"
    )

    return model


def build_model2():
    model = Sequential()

    # model.add(Dropout(0.5))
    model.add(Dense(289, 512, W_regularizer=l1l2()))
    # model.add(PReLU((16,)))
    model.add(MaxoutDense(512, 512, nb_feature=4))
    model.add(Dropout(0.5))
    model.add(BatchNormalization((512,)))

    model.add(Dense(512, 256))
    # model.add(PReLU((256,)))
    model.add(MaxoutDense(256, 256, nb_feature=4))
    model.add(Dropout(0.25))
    model.add(BatchNormalization((256,)))

    model.add(Dense(256, 128))
    # model.add(PReLU((128,)))
    model.add(MaxoutDense(128, 128, nb_feature=4))
    model.add(Dropout(0.125))
    model.add(BatchNormalization((128,)))

    # model = Sequential()
    # model.add(Merge([rnn, mlp], mode='concat'))

    model.add(Dense(128, 2, activation='softmax'))
    # model.add(PReLU((1,)))

    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        class_mode="binary"
    )

    return model


class XGBoostClassifier():
    def __init__(self, **params):
        self.clf = None
        self.num_boost_round = params['num_boost_round']
        if '_lambda' in params:
            params['lambada'] = params['_lambda']
            del params['_lambda']
        self.params = params
        self.params.update({'objective': 'binary:logistic', 'eval_metric': 'auc', 'silent': 1, 'nthread': 36})
        self.classes_ = [1]

    def fit(self, X, y, num_boost_round=None):
        # freqs = itemfreq(y)
        # pos_count = freqs[1][1]
        # neg_count = freqs[0][1]
        # self.params['scale_pos_weight'] = neg_count / pos_count
        num_boost_round = num_boost_round or self.num_boost_round
        self.label2num = dict((label, i) for i, label in enumerate(sorted(set(y))))
        dtrain = xgb.DMatrix(X, label=[self.label2num[label] for label in y])
        num_boost_round = int(num_boost_round)
        # self.params['max_depth'] = int(self.params['max_depth'])
        self.clf = xgb.train(params=self.params, dtrain=dtrain, num_boost_round=num_boost_round)

    def predict(self, X):
        num2label = dict((i, label)for label, i in self.label2num.items())
        Y = self.predict_proba(X)
        y = np.argmax(Y, axis=1)
        return np.array([num2label[i] for i in y])

    def predict_proba(self, X):
        dtest = xgb.DMatrix(X)
        predicts = self.clf.predict(dtest)
        return predicts.reshape(predicts.shape[0], 1)

    def score(self, X, y):
        Y = self.predict_proba(X)
        return 1 / logloss(y, Y)

    def get_params(self, deep=True):
        return self.params

    def set_params(self, **params):
        if 'num_boost_round' in params:
            self.num_boost_round = params.pop('num_boost_round')
        if 'objective' in params:
            del params['objective']
        self.params.update(params)
        return self


class XGBoostRegressor():
    def __init__(self, **params):
        self.clf = None
        self.num_boost_round = params['num_boost_round']
        if '_lambda' in params:
            params['lambada'] = params['_lambda']
            del params['_lambda']
        self.params = params
        self.params.update({'objective': 'reg:logistic', 'eval_metric': 'auc', 'silent': 1, 'nthread': 36})
        self.classes_ = [1]

    def fit(self, X, y, num_boost_round=None):
        # freqs = itemfreq(y)
        # pos_count = freqs[1][1]
        # neg_count = freqs[0][1]
        # self.params['scale_pos_weight'] = neg_count / pos_count
        num_boost_round = num_boost_round or self.num_boost_round
        self.label2num = dict((label, i) for i, label in enumerate(sorted(set(y))))
        dtrain = xgb.DMatrix(X, label=[self.label2num[label] for label in y])
        num_boost_round = int(num_boost_round)
        self.params['max_depth'] = int(self.params['max_depth'])
        self.clf = xgb.train(params=self.params, dtrain=dtrain, num_boost_round=num_boost_round)

    def predict(self, X):
        num2label = dict((i, label)for label, i in self.label2num.items())
        Y = self.predict_proba(X)
        y = np.argmax(Y, axis=1)
        return np.array([num2label[i] for i in y])

    def predict_proba(self, X):
        dtest = xgb.DMatrix(X)
        predicts = self.clf.predict(dtest)
        return predicts.reshape(predicts.shape[0], 1)

    def score(self, X, y):
        Y = self.predict_proba(X)
        return 1 / logloss(y, Y)

    def get_params(self, deep=True):
        return self.params

    def set_params(self, **params):
        if 'num_boost_round' in params:
            self.num_boost_round = params.pop('num_boost_round')
        if 'objective' in params:
            del params['objective']
        self.params.update(params)
        return self


def logloss(y_true, Y_pred):
    label2num = dict((name, i) for i, name in enumerate(sorted(set(y_true))))
    return -1 * sum(math.log(y[label2num[label]]) if y[label2num[label]] > 0 else -np.inf for y, label in zip(Y_pred, y_true)) / len(Y_pred)


def build_xgbm(gamma, seed):

    # model = XGBoostClassifier(params)
    # model = XGBoostClassifier(num_boost_round=300, eta=0.06143935837511715, gamma=0.2769411806088273, subsample=0.7212129811909465, colsample_bytree=0.700388722822422, min_child_weight=1.0289254146525995, max_depth=6)
    # model = XGBoostClassifier(num_boost_round=120, eta=0.07143935837511715, gamma=0.2769411806088273, subsample=0.6212129811909465, colsample_bytree=0.700388722822422, min_child_weight=1.0289254146525995, max_depth=6)
    # model = XGBoostClassifier(num_boost_round=2700, eta=0.005, gamma=0.1, subsample=0.8, colsample_bytree=0.5, min_child_weight=1, max_depth=6)  # 0.901929
    # xgb = XGBoostClassifier(num_boost_round=3000, eta=0.004, gamma=0.1, subsample=0.8, colsample_bytree=0.45, min_child_weight=1, max_depth=6)  # 0.901935
    xgb = XGBoostClassifier(num_boost_round=3050, eta=0.004, gamma=gamma, subsample=1, colsample_bytree=0.2, min_child_weight=1, max_depth=5, seed=seed)  # 0.901935
    # bagging = BaggingClassifier(xgb, n_estimators=5, max_samples=0.9)
    # gbr = GradientBoostingRegressor()
    return xgb


def build_xgblm():

    # model = XGBoostClassifier(params)
    # model = XGBoostClassifier(num_boost_round=300, eta=0.06143935837511715, gamma=0.2769411806088273, subsample=0.7212129811909465, colsample_bytree=0.700388722822422, min_child_weight=1.0289254146525995, max_depth=6)
    # model = XGBoostClassifier(num_boost_round=120, eta=0.07143935837511715, gamma=0.2769411806088273, subsample=0.6212129811909465, colsample_bytree=0.700388722822422, min_child_weight=1.0289254146525995, max_depth=6)
    # model = XGBoostClassifier(num_boost_round=2700, eta=0.005, gamma=0.1, subsample=0.8, colsample_bytree=0.5, min_child_weight=1, max_depth=6)  # 0.901929
    # xgb = XGBoostClassifier(num_boost_round=3000, eta=0.004, gamma=0.1, subsample=0.8, colsample_bytree=0.45, min_child_weight=1, max_depth=6)  # 0.901935
    xgb = XGBoostClassifier(booster='gblinear', num_boost_round=200, _lambda=1)  # 0.901935
    # bagging = BaggingClassifier(xgb, n_estimators=5, max_samples=0.9)
    # gbr = GradientBoostingRegressor()
    return xgb


def build_xgbmr(colsample_bytree, subsample):

    # model = XGBoostClassifier(params)
    # model = XGBoostClassifier(num_boost_round=300, eta=0.06143935837511715, gamma=0.2769411806088273, subsample=0.7212129811909465, colsample_bytree=0.700388722822422, min_child_weight=1.0289254146525995, max_depth=6)
    # model = XGBoostClassifier(num_boost_round=120, eta=0.07143935837511715, gamma=0.2769411806088273, subsample=0.6212129811909465, colsample_bytree=0.700388722822422, min_child_weight=1.0289254146525995, max_depth=6)
    # model = XGBoostClassifier(num_boost_round=2700, eta=0.005, gamma=0.1, subsample=0.8, colsample_bytree=0.5, min_child_weight=1, max_depth=6)  # 0.901929
    # xgb = XGBoostClassifier(num_boost_round=3000, eta=0.004, gamma=0.1, subsample=0.8, colsample_bytree=0.45, min_child_weight=1, max_depth=6)  # 0.901935
    xgb = XGBoostRegressor(num_boost_round=3050, eta=0.004, gamma=0.1, subsample=subsample, colsample_bytree=colsample_bytree, min_child_weight=1, max_depth=6)  # 0.901935
    # bagging = BaggingClassifier(xgb, n_estimators=5, max_samples=0.9)
    # gbr = GradientBoostingRegressor()
    return xgb


def build_gbm():
    gbm = GradientBoostingClassifier(n_estimators=500)
    return gbm


def build_rf():
    model = GradientBoostingClassifier(
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=6,
        # max_features=None,
        # min_samples_leaf=20,
        # n_jobs=-1,
        random_state=23)
    return model


def build_linear():
    model = LogisticRegressionCV(n_jobs=-1, cv=5)
    return model


def build_knn():
    model = KNeighborsClassifier(20)
    return model


def build_semi():
    model = LabelSpreading()
    return model
