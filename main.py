# -*- coding: utf-8 -*-
from __future__ import print_function

from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_auc_score

from kdd2015.data import load_data, to_submission
from kdd2015.model import build_model
from kdd2015.analyze import errors


if __name__ == '__main__':
    x_time_series_train, x_enrollment_train, y_train, x_time_series_test, x_enrollment_test = load_data()
    # x_train_cv, x_test_cv, y_train_cv, y_test_cv = train_test_split(
    #     x_train, y_train, test_size=0.375, random_state=24)

    model = build_model()

    # print('shape:', x_train_cv[0].shape)

    rnn_features = list(range(35)) + list(range(37, 65))
    mlp_features = [35, 36, 65, 66, 67, 68]  # [65, 66, 67, 68]

    model.fit(
        [x_time_series_train, x_enrollment_train], y_train,
        # x_train[:, :, rnn_features], y_train,
        batch_size=8,
        nb_epoch=10,
        show_accuracy=True,
        # class_weight={0: 0.7929269466244131, 1: 0.2070730533755869}
        # class_weight={1: 1, 0: 3.829213573174152}
    )

    predicts_cv = model.predict_proba(
        [x_time_series_train, x_enrollment_train],
        batch_size=8)
    print('roc_auc_score of cv %f' % roc_auc_score(y_train, predicts_cv))
    predicts = model.predict_proba(
        [x_time_series_test, x_enrollment_test],
        # x_test[:, :, rnn_features],
        batch_size=8)
    to_submission(predicts)

    # errors(x_train, y_train, predicts)
