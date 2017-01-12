# -*- coding: utf-8 -*-
from __future__ import print_function, division

from multiprocessing import Pool
from datetime import datetime

from sklearn.cross_validation import train_test_split, StratifiedKFold, KFold, ShuffleSplit
from sklearn.metrics import roc_auc_score, accuracy_score

import numpy as np

# from hyperopt import hp
# from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from scipy.stats.mstats import zscore

from kdd2015.data import load_data, to_submission
from kdd2015.model import build_model, build_xgbm, build_rf, build_model2, build_model0, build_gmodel, build_linear, build_knn, build_gbm, build_xgbmr, build_xgblm, build_semi
from kdd2015.analyze import errors

import os
import pickle


def exp0():
    x_time_series_train, x_enrollment_train, y_train, enrollment_id_df, sample_weight_df = load_data()
    # y_train = y_train[enrollment_ids]

    if os.path.isfile('cv.pickle'):
        print('laod cv')
        skf = pickle.load(open('cv.pickle'))
    else:
        print('gen cv')
        skf = ShuffleSplit(y_train.shape[0], 1, 0.4, random_state=23)
        pickle.dump(skf, open('cv.pickle', 'wb'))

    # enrollment_ids = np.arange(y_train.shape[0])
    # np.random.shuffle(enrollment_ids)


    # selected_features = list(set(range(44)) - set([]))
    # import pdb; pdb.set_trace()

    for train_index, test_index in skf:

        model = build_gmodel()

        train_weight_df = sample_weight_df[sample_weight_df['train'] == True]

        # import pdb; pdb.set_trace()

        model.fit(
            # [
            {
                'time_series': x_time_series_train[train_index],
                'enrollment': x_enrollment_train[train_index],
                'output': y_train[train_index].reshape(train_index.shape[0], 1)
            },
                # x_time_series_train[train_index],
                # x_time_series_train[train_index],
            # ],
            # y_train[train_index],
            # x_train[:, :, rnn_features], y_train,
            batch_size=32,
            nb_epoch=10,
            # show_accuracy=True,
            # class_weight={0: 0.7929269466244131, 1: 0.2070730533755869}
            # class_weight={0: 1, 1: (3.829213573174152 / 3)},
            # sample_weight=train_weight_df.iloc[train_index]['weight'].values
            # class_weight={0: 2, 1: 1}
        )

        # import pdb; pdb.set_trace()

        predicts_cv = model.predict(
            # [
            {
                'time_series': x_time_series_train[train_index],
                'enrollment': x_enrollment_train[train_index]
            },
                # x_time_series_train[train_index],
                # x_time_series_train[train_index],
            # ],
            batch_size=4096)['output'].flatten()
        print('roc_auc_score of cv on train %f' % roc_auc_score(y_train[train_index], predicts_cv))
        print('accuracy_score of cv on train %f' % accuracy_score(y_train[train_index], predicts_cv.round()))

        # errors(x_time_series_train, y_train[train_index], predicts_cv, train_index, 'train')

        predicts_cv = model.predict(
            # [
            {
                'time_series': x_time_series_train[test_index],
                'enrollment': x_enrollment_train[test_index]
            },
                # x_time_series_train[test_index],
                # x_time_series_train[test_index],
            # ],
            batch_size=4096)['output'].flatten()
        print('roc_auc_score of cv on test %f' % roc_auc_score(y_train[test_index], predicts_cv))
        print('accuracy_score of cv on test %f' % accuracy_score(y_train[test_index], predicts_cv.round()))

        # errors(x_time_series_train, y_train[test_index], predicts_cv, test_index, 'test')


def exp():
    x_time_series_train, x_enrollment_train, y_train, enrollment_id_df, sample_weight_df = load_data()
    # y_train = y_train[enrollment_ids]
    # x_time_series_train.astype('float32')
    # x_enrollment_train.astype('float32')
    # y_train.astype('int32')

    # import pdb; pdb.set_trace()
    if os.path.isfile('cv.pickle'):
        print('laod cv')
        skf = pickle.load(open('cv.pickle'))
    else:
        print('gen cv')
        skf = ShuffleSplit(y_train.shape[0], 1, 0.4, random_state=23)
        pickle.dump(skf, open('cv.pickle', 'wb'))

    # enrollment_ids = np.arange(y_train.shape[0])
    # np.random.shuffle(enrollment_ids)


    selected_features = list(set(range(53)) - set([]))

    for train_index, test_index in skf:

        model = build_model()
        # import pdb; pdb.set_trace()

        model.fit(
            [
                x_time_series_train[train_index],
                x_enrollment_train[:, selected_features][train_index],
                x_enrollment_train[:, selected_features][train_index],
                x_enrollment_train[:, selected_features][train_index],
                x_enrollment_train[:, selected_features][train_index]
            ],
            y_train[train_index],
            # x_train[:, :, rnn_features], y_train,
            batch_size=64,
            nb_epoch=10,
            show_accuracy=True,
            # sample_weight=sample_weight_df['weight'].values
            # class_weight={0: 0.7929269466244131, 1: 0.2070730533755869}
            # class_weight={0: 1, 1: (3.829213573174152 / 1)}
            # class_weight={0: 2, 1: 1}
        )

        predicts_cv = model.predict_proba(
            [
                x_time_series_train[train_index],
                x_enrollment_train[:, selected_features][train_index],
                x_enrollment_train[:, selected_features][train_index],
                x_enrollment_train[:, selected_features][train_index],
                x_enrollment_train[:, selected_features][train_index]
            ],
            batch_size=4096)
        print('roc_auc_score of cv on train %f' % roc_auc_score(y_train[train_index], predicts_cv))
        print('accuracy_score of cv on train %f' % accuracy_score(y_train[train_index], predicts_cv.round()))

        predicts_cv = model.predict_proba(
            [
                x_time_series_train[test_index],
                x_enrollment_train[:, selected_features][test_index],
                x_enrollment_train[:, selected_features][test_index],
                x_enrollment_train[:, selected_features][test_index],
                x_enrollment_train[:, selected_features][test_index]
            ],
            batch_size=4096)
        print('roc_auc_score of cv on test %f' % roc_auc_score(y_train[test_index], predicts_cv))
        print('accuracy_score of cv on test %f' % accuracy_score(y_train[test_index], predicts_cv.round()))


def exp1():
    x_enrollment_train, x_normal_enrollment_train, x_enrollment_test, x_normal_enrollment_test, y_train, enrollment_id_df, sample_weight_df = load_data()

    # if os.path.isfile('cv.pickle'):
    #     print('laod cv')
    #     skf = pickle.load(open('cv.pickle'))
    # else:
    #     print('gen cv')
    #     skf = ShuffleSplit(y_train.shape[0], 1, 0.4, random_state=23)
    #     pickle.dump(skf, open('cv.pickle', 'wb'))

    aucs = []

    reject_features = []
    # reject_features = list(range(170, 209)) + list(range(287, 303))
    # reject_features = list(range(170, 287)) + list(range(287, 303)) + list(range(303, 367))
    # reject_features = list(range(170, 248))  #+ list(range(287, 303))
    selected_features = list(set(range(367)) - set(reject_features))

    print('nb_feature:', len(selected_features))

    for i in range(20):

        start = datetime.now()

        # skf = ShuffleSplit(y_train.shape[0], 1, 0.5)
        skf = ShuffleSplit(y_train.shape[0], 1, 0.2)

        for train_index, test_index in skf:

            # skf2 = ShuffleSplit(test_index.shape[0], 1, 0.5)

            # for validate_index, hidden_index in skf2:

            model1 = build_xgbm(0.045, np.random.randint(1000))
            model1.fit(
                x_enrollment_train[:, selected_features][train_index],
                y_train[train_index]
            )

            predicts1 = model1.predict_proba(x_enrollment_train[:, selected_features][test_index])
            roc = roc_auc_score(y_train[test_index], predicts1)
            print('roc1 %f' % roc)

            model2 = build_xgbm(0.1, np.random.randint(1000))
            model2.fit(
                x_enrollment_train[:, selected_features][train_index],
                y_train[train_index]
            )

            predicts2 = model2.predict_proba(x_enrollment_train[:, selected_features][test_index])
            roc = roc_auc_score(y_train[test_index], predicts2)
            print('roc2 %f' % roc)

            model3 = build_xgbm(0.15, np.random.randint(1000))
            model3.fit(
                x_enrollment_train[:, selected_features][train_index],
                y_train[train_index]
            )

            predicts3 = model3.predict_proba(x_enrollment_train[:, selected_features][test_index])
            roc = roc_auc_score(y_train[test_index], predicts3)
            print('roc3 %f' % roc)

            predicts = (predicts1 + predicts2 + predicts3) / 3
            roc = roc_auc_score(y_train[test_index], predicts)
            print('roc %f' % roc)

            import ipdb; ipdb.set_trace()

                # model5s = []
                # for xgb_id in range(2):
                #     model5 = build_xgbm(0.4, 1, np.random.randint(1000))
                #     model5.fit(
                #         x_enrollment_train[:, selected_features][train_index],
                #         y_train[train_index]
                #     )
                #     model5s.append((model5))

                # predicts_bag1 = []
                # for xgb_id in range(2):
                #     predicts = model1s[xgb_id].predict_proba(x_enrollment_train[:, selected_features][test_index])
                #     roc = roc_auc_score(y_train[test_index], predicts)
                #     print('%d roc_auc_score of cv on test1 %f' % (xgb_id, roc))
                #     predicts_bag1.append(predicts)

                # predicts1 = predicts_bag1[0] + predicts_bag1[1]
                # roc = roc_auc_score(y_train[test_index], predicts1)
                # print('roc_auc_score of cv on test %f' % roc)

                # predicts_bag2 = []
                # for xgb_id in range(2):
                #     predicts = model2s[xgb_id].predict_proba(x_enrollment_train[:, selected_features][test_index])
                #     roc = roc_auc_score(y_train[test_index], predicts)
                #     print('%d roc_auc_score of cv on test2 %f' % (xgb_id, roc))
                #     predicts_bag2.append(predicts)

                # predicts2 = predicts_bag2[0] + predicts_bag2[1]
                # roc = roc_auc_score(y_train[test_index], predicts2)
                # print('roc_auc_score of cv on test %f' % roc)

                # predicts_bag3 = []
                # for xgb_id in range(2):
                #     predicts = model3s[xgb_id].predict_proba(x_enrollment_train[:, selected_features][test_index])
                #     roc = roc_auc_score(y_train[test_index], predicts)
                #     print('%d roc_auc_score of cv on test3 %f' % (xgb_id, roc))
                #     predicts_bag3.append(predicts)

                # predicts3 = predicts_bag3[0] + predicts_bag3[1]
                # roc = roc_auc_score(y_train[test_index], predicts3)
                # print('roc_auc_score of cv on test %f' % roc)

                # predicts_bag4 = []
                # for xgb_id in range(2):
                #     predicts = model4s[xgb_id].predict_proba(x_enrollment_train[:, selected_features][test_index])
                #     roc = roc_auc_score(y_train[test_index], predicts)
                #     print('%d roc_auc_score of cv on test4 %f' % (xgb_id, roc))
                #     predicts_bag4.append(predicts)

                # predicts4 = predicts_bag4[0] + predicts_bag4[1]
                # roc = roc_auc_score(y_train[test_index], predicts4)
                # print('roc_auc_score of cv on test %f' % roc)

                # # predicts_bag5 = []
                # # for xgb_id in range(2):
                # #     predicts = model5s[xgb_id].predict_proba(x_enrollment_train[:, selected_features][test_index])
                # #     roc = roc_auc_score(y_train[test_index], predicts)
                # #     print('%d roc_auc_score of cv on test5 %f' % (xgb_id, roc))
                # #     predicts_bag5.append(predicts)

                # # predicts5 = predicts_bag5[0] + predicts_bag5[1]
                # # roc = roc_auc_score(y_train[test_index], predicts5)
                # # print('roc_auc_score of cv on test %f' % roc)

                # # predicts = (predicts3 + predicts4) / 2
                # # roc = roc_auc_score(y_train[test_index], predicts)
                # # print('roc_auc_score of cv on test %f' % roc)

                # import ipdb; ipdb.set_trace()

                # # roc3_h_f = roc_auc_score(y_train[test_index[hidden_index]], predicts_cv3_h_f)
                # # print('3 roc_auc_score of cv on test_h %f' % roc3_h_f)
                # # roc4 = roc_auc_score(y_train[test_index], predicts_cv4)
                # # print('4 roc_auc_score of cv on test %f' % roc4)
                # # roc5 = roc_auc_score(y_train[test_index], predicts_cv5)
                # # print('5 roc_auc_score of cv on test %f' % roc5)
                # # roc6 = roc_auc_score(y_train[test_index], predicts_cv6)
                # # print('6 roc_auc_score of cv on test %f' % roc6)
                # # roc = roc_auc_score(y_train[test_index], predicts_cv)
                # # print('roc_auc_score of cv on test %f' % roc)

                # # aucs.append(roc)

                # selected = ((predicts_cv3 > 0.9) & (predicts_cv3 < 1)) | ((predicts_cv3 < 0.1) & (predicts_cv3 > 0))
                # selected = selected.reshape(selected.shape[0])

                # filtered = x_enrollment_train[:, selected_features][test_index[validate_index]][:, 0] > 1

                # import ipdb; ipdb.set_trace()

                # reweighted1 = x_enrollment_train[:, selected_features][test_index[validate_index]][:, 0] < 2
                # reweighted2 = x_enrollment_train[:, selected_features][test_index[validate_index]][:, 130] > 28
                # reweighted = reweighted1 & reweighted2

                # predicts_cv3_r = predicts_cv3
                # for i in range(predicts_cv3.shape[0]):
                #     if reweighted[i]:
                #         predicts_cv3_r[i] = 0.8196286472148541

                # roc3_r = roc_auc_score(y_train[test_index[validate_index]], predicts_cv3_r)
                # print('3_r roc_auc_score of cv on test %f' % roc3_r)

                # reweighted1 = x_enrollment_train[:, selected_features][test_index[hidden_index]][:, 0] < 2
                # reweighted2 = x_enrollment_train[:, selected_features][test_index[hidden_index]][:, 14] > 28
                # reweighted = reweighted1 & reweighted2

                # import ipdb; ipdb.set_trace()

                # predicts_cv3_h_r = predicts_cv3_h
                # for i in range(predicts_cv3_h.shape[0]):
                #     if reweighted[i]:
                #         predicts_cv3_h_r[i] = 0.8196286472148541

                # roc3_h_r = roc_auc_score(y_train[test_index[hidden_index]], predicts_cv3_h_r)
                # print('3_h_r roc_auc_score of cv on test %f' % roc3_h_r)

                # # import ipdb; ipdb.set_trace()

                # selected = selected & filtered

                # # import ipdb; ipdb.set_trace()

                # reg = build_xgbm(0.5, 0.5)
                # new_train = np.concatenate(
                #     [
                #         x_enrollment_train[:, selected_features][train_index],
                #         x_enrollment_train[:, selected_features][test_index[validate_index[selected]]]
                #     ]
                # )
                # # new_y1 = model3.predict_proba(x_enrollment_train[:, selected_features][train_index])
                # new_y1 = y_train[train_index]
                # new_y2 = predicts_cv3[selected].round()
                # new_y2 = new_y2.reshape(new_y2.shape[0])
                # new_y = np.concatenate([new_y1, new_y2])
                # new_y = new_y.reshape(new_y.shape[0])
                # reg.fit(new_train, new_y)
                # new_predicts = reg.predict_proba(x_enrollment_train[:, selected_features][test_index[validate_index]])
                # new_roc = roc_auc_score(y_train[test_index[validate_index]], new_predicts)
                # print('new roc_auc_score of cv on test %f' % new_roc)

                # avg_predicts = (new_predicts + predicts_cv3) / 2

                # avg_roc = roc_auc_score(y_train[test_index[validate_index]], avg_predicts)
                # print('avg roc_auc_score of cv on test %f' % avg_roc)

                # new_predicts_h = reg.predict_proba(x_enrollment_train[:, selected_features][test_index[hidden_index]])
                # new_roc_h = roc_auc_score(y_train[test_index[hidden_index]], new_predicts_h)
                # print('new roc_auc_score of cv on test_h %f' % new_roc_h)

                # avg_predicts_h = (new_predicts_h + predicts_cv3_h) / 2

                # avg_roc_h = roc_auc_score(y_train[test_index[hidden_index]], avg_predicts_h)
                # print('avg roc_auc_score of cv on test %f' % avg_roc_h)

                # import ipdb; ipdb.set_trace()

        end = datetime.now()

        print('second: %d' % (end - start).seconds)

    # print('mean auc of 3 cv %f' % np.mean(aucs))
    # print('max auc of 3 cv %f' % np.max(aucs))
    # print('min auc of 3 cv %f' % np.min(aucs))


def exp11():
    x_enrollment_train, x_normal_enrollment_train, x_enrollment_test, x_normal_enrollment_test, y_train, enrollment_id_df, sample_weight_df = load_data()

    reject_features = list(range(170, 248))
    selected_features = list(set(range(367)) - set(reject_features))

    print('nb_feature:', len(selected_features))

    start = datetime.now()

    model = build_xgbm()

    model.fit(
        x_enrollment_train[:, selected_features],
        y_train
    )

    if hasattr(model, 'predict_proba'):
        predicts = model.predict_proba(x_enrollment_test[:, selected_features])
    else:
        predicts = model.decision_function(x_enrollment_test[:, selected_features])

    if len(predicts.shape) == 2:
        if predicts.shape[1] == 2:
            to_submission(predicts[:, 1])
        else:
            to_submission(predicts[:, 0])
    else:
        to_submission(predicts)

    end = datetime.now()

    print('second: %d' % (end - start).seconds)


def exp2proc(i):
    x_enrollment_train, x_normal_enrollment_train, x_enrollment_test, x_normal_enrollment_test, y_train, enrollment_id_df, sample_weight_df = load_data()

    random_state = i + np.random.randint(1000)
    skf = ShuffleSplit(y_train.shape[0], 1, 0.4, random_state=random_state)

    for train_index, test_index in skf:

        model = build_model2()

        reject_features = []
        # reject_features = list(range(170, 209)) + list(range(287, 303))
        # reject_features = list(range(170, 287)) + list(range(287, 303)) + list(range(303, 367))
        reject_features = list(range(170, 248))
        selected_features = list(set(range(367)) - set(reject_features))

        # import ipdb; ipdb.set_trace()
        y_train2 = np.vstack([1 - y_train[train_index], y_train[train_index]]).T

        model.fit(
            x_normal_enrollment_train[:, selected_features][train_index],
            y_train2,
            nb_epoch=7
        )

        if hasattr(model, 'predict_proba'):
            predicts_cv = model.predict_proba(x_normal_enrollment_train[:, selected_features][test_index])
        else:
            predicts_cv = model.decision_function(x_normal_enrollment_train[:, selected_features][test_index])

        if len(predicts_cv.shape) == 2:
            if predicts_cv.shape[1] == 2:
                roc = roc_auc_score(y_train[test_index], predicts_cv[:, 1])
                print('roc_auc_score of cv on test %f' % roc)
            else:
                roc = roc_auc_score(y_train[test_index], predicts_cv[:, 0])
                print('roc_auc_score of cv on test %f' % roc)
        else:
            roc = roc_auc_score(y_train[test_index], predicts_cv)
            print('roc_auc_score of cv on test %f' % roc)

    return roc


def exp2():

    pool = Pool(5)
    aucs = pool.map(exp2proc, range(10))
    pool.close()
    pool.join()

    print('mean auc of 10 cv %f' % np.mean(aucs))
    print('max auc of 10 cv %f' % np.max(aucs))
    print('min auc of 10 cv %f' % np.min(aucs))


def exp22():
    x_enrollment_train, x_normal_enrollment_train, x_enrollment_test, x_normal_enrollment_test, y_train, enrollment_id_df, sample_weight_df = load_data()

    skf = ShuffleSplit(y_train.shape[0], 1, 0.4)

    for train_index, test_index in skf:

        reject_features = []
        # reject_features = list(range(170, 209)) + list(range(287, 303))
        # reject_features = list(range(170, 287)) + list(range(287, 303)) + list(range(303, 367))
        reject_features = list(range(170, 248)) + list(range(287, 303))
        selected_features = list(set(range(367)) - set(reject_features))

        print('nb_feature:', len(selected_features))

        # import ipdb; ipdb.set_trace()
        # y_train2 = np.vstack([1 - y_train[train_index], y_train[train_index]]).T

        model = build_linear()

        # model.fit(
        #     x_normal_enrollment_train[:, selected_features][train_index],
        #     y_train2,
        #     nb_epoch=7
        # )

        model.fit(
            x_normal_enrollment_train[:, selected_features][train_index],
            y_train[train_index]
        )

        if hasattr(model, 'predict_proba'):
            predicts_cv = model.predict_proba(x_normal_enrollment_train[:, selected_features][test_index])
        else:
            predicts_cv = model.decision_function(x_normal_enrollment_train[:, selected_features][test_index])

        if len(predicts_cv.shape) == 2:
            if predicts_cv.shape[1] == 2:
                roc = roc_auc_score(y_train[test_index], predicts_cv[:, 1])
                print('roc_auc_score of cv on test %f' % roc)
            else:
                roc = roc_auc_score(y_train[test_index], predicts_cv[:, 0])
                print('roc_auc_score of cv on test %f' % roc)
        else:
            roc = roc_auc_score(y_train[test_index], predicts_cv)
            print('roc_auc_score of cv on test %f' % roc)


def exp23():
    x_enrollment_train, x_normal_enrollment_train, x_enrollment_test, x_normal_enrollment_test, y_train, enrollment_id_df, sample_weight_df = load_data()

    skf = ShuffleSplit(y_train.shape[0], 1, 0.4)

    for train_index, test_index in skf:

        reject_features = []
        # reject_features = list(range(170, 209)) + list(range(287, 303))
        # reject_features = list(range(170, 287)) + list(range(287, 303)) + list(range(303, 367))
        reject_features = list(range(170, 248))
        selected_features = list(set(range(367)) - set(reject_features))

        print('nb_feature:', len(selected_features))

        # import ipdb; ipdb.set_trace()
        y_train2 = np.vstack([1 - y_train[train_index], y_train[train_index]]).T

        model = build_model2()

        model.fit(
            x_normal_enrollment_train[:, selected_features][train_index],
            y_train2,
            nb_epoch=7
        )

        # model.fit(
        #     x_normal_enrollment_train[:, selected_features][train_index],
        #     y_train[train_index]
        # )

        if hasattr(model, 'predict_proba'):
            predicts_cv = model.predict_proba(x_normal_enrollment_train[:, selected_features][test_index])
        else:
            predicts_cv = model.decision_function(x_normal_enrollment_train[:, selected_features][test_index])

        if len(predicts_cv.shape) == 2:
            if predicts_cv.shape[1] == 2:
                roc = roc_auc_score(y_train[test_index], predicts_cv[:, 1])
                print('roc_auc_score of cv on test %f' % roc)
            else:
                roc = roc_auc_score(y_train[test_index], predicts_cv[:, 0])
                print('roc_auc_score of cv on test %f' % roc)
        else:
            roc = roc_auc_score(y_train[test_index], predicts_cv)
            print('roc_auc_score of cv on test %f' % roc)


def exp3():
    x_time_series_train, x_enrollment_train, y_train, enrollment_id_df, sample_weight_df, x_enrollment_test = load_data()
    # import pdb; pdb.set_trace()
    if os.path.isfile('cv.pickle'):
        print('laod cv')
        skf = pickle.load(open('cv.pickle'))
    else:
        print('gen cv')
        skf = ShuffleSplit(y_train.shape[0], 1, 0.4, random_state=23)
        pickle.dump(skf, open('cv.pickle', 'wb'))

    # y_train = y_train[enrollment_ids]

    # enrollment_ids = np.arange(y_train.shape[0])
    # np.random.shuffle(enrollment_ids)

    selected_features = list(set(range(170)) - set([]))
    # selected_features = list(range(12, 21))

    # import pdb; pdb.set_trace()
    for train_index, test_index in skf:

        model = build_rf()

        model.fit(
            x_enrollment_train[:, selected_features][train_index],
            y_train[train_index]
        )

        predicts_cv = model.predict_proba(x_enrollment_train[:, selected_features][train_index])
        print('roc_auc_score of cv on train %f' % roc_auc_score(y_train[train_index], predicts_cv[:, 1]))
        print('accuracy_score of cv on train %f' % accuracy_score(y_train[train_index], predicts_cv[:, 1].round()))

        predicts_cv = model.predict_proba(x_enrollment_train[:, selected_features][test_index])
        print('roc_auc_score of cv on test %f' % roc_auc_score(y_train[test_index], predicts_cv[:, 1]))
        print('accuracy_score of cv on test %f' % accuracy_score(y_train[test_index], predicts_cv[:, 1].round()))

    import pdb; pdb.set_trace()


def exp4():
    x_time_series_train, x_enrollment_train, y_train, enrollment_id_df = load_data()

    x_time_series_train.astype('float32')
    x_enrollment_train.astype('float32')
    y_train.astype('int32')

    # y_train = y_train[enrollment_ids]
    skf = KFold(y_train.shape[0], shuffle=True)

    model = build_model2()

    selected_features = list(set(range(53)) - set([]))
    # selected_features = list(range(12, 21))
    for train_index, test_index in skf:

        model.fit(
            x_enrollment_train[:, selected_features][train_index],
            y_train[train_index],
            batch_size=128,
            nb_epoch=10,
            show_accuracy=True,
        )

        predicts_cv = model.predict_proba(x_enrollment_train[:, selected_features][train_index], batch_size=128)
        print('roc_auc_score of cv on train %f' % roc_auc_score(y_train[train_index], predicts_cv))
        # import pdb; pdb.set_trace()
        print('accuracy_score of cv on train %f' % accuracy_score(y_train[train_index], predicts_cv.round()))

        predicts_cv = model.predict_proba(x_enrollment_train[:, selected_features][test_index], batch_size=128)
        print('roc_auc_score of cv on test %f' % roc_auc_score(y_train[test_index], predicts_cv))
        print('accuracy_score of cv on test %f' % accuracy_score(y_train[test_index], predicts_cv.round()))

    # import pdb; pdb.set_trace()


if __name__ == '__main__':
    # x_time_series_train, x_enrollment_train, y_train, enrollment_id_df, x_time_series_test, x_enrollment_test = load_data()
    # x_time_series_train, x_enrollment_train, y_train, enrollment_id_df = load_data()
    # x_train_cv, x_test_cv, y_train_cv, y_test_cv = train_test_split(
    #     x_train, y_train, test_size=0.375, random_state=24)

    # x_enrollment_train = np.column_stack([x_enrollment_train, np.zeros(x_enrollment_train.shape[0])])

    # enrollment_ids = np.arange(y_train.shape[0])
    # np.random.shuffle(enrollment_ids)

    # model = build_model()

    # # print('shape:', x_train_cv[0].shape)

    # rnn_features = list(range(35)) + list(range(37, 65))
    # mlp_features = [35, 36, 65, 66, 67, 68]  # [65, 66, 67, 68]

    # model.fit(
    #     [x_time_series_train[enrollment_ids[:7200]], x_enrollment_train[enrollment_ids[:7200]]],
    #     y_train[enrollment_ids[:7200]],
    #     # x_train[:, :, rnn_features], y_train,
    #     batch_size=128,
    #     nb_epoch=10,
    #     show_accuracy=True,
    #     # class_weight={0: 0.7929269466244131, 1: 0.2070730533755869}
    #     # class_weight={0: 1, 1: (3.829213573174152 / 1)}
    #     # class_weight={0: 2, 1: 1}
    # )

    # predicts_cv = model.predict_proba(
    #     [x_time_series_train[enrollment_ids[:7200]], x_enrollment_train[enrollment_ids[:7200]]],
    #     batch_size=128)
    # print('roc_auc_score of cv %f' % roc_auc_score(y_train[enrollment_ids[:7200]], predicts_cv))
    # print('accuracy_score of cv %f' % accuracy_score(y_train[enrollment_ids[:7200]], predicts_cv.round()))

    # predicts_cv = model.predict_proba(
    #     [x_time_series_train[enrollment_ids[7200:]], x_enrollment_train[enrollment_ids[7200:]]],
    #     batch_size=128)
    # print('roc_auc_score of cv %f' % roc_auc_score(y_train[enrollment_ids[7200:]], predicts_cv))
    # print('accuracy_score of cv %f' % accuracy_score(y_train[enrollment_ids[7200:]], predicts_cv.round()))

    # predicts = model.predict_proba(
    #     [x_time_series_test, x_enrollment_test],
    #     # x_test[:, :, rnn_features],
    #     batch_size=16)
    # to_submission(predicts)

    # errors(x_train, y_train, predicts)
    exp2()
