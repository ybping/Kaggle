#!/usr/bin/env python
# -*- coding: utf-8 -*-
###############################################################################
# The MIT License (MIT)
#
# Copyright (c)  Yabin Ping
# Mail(Forever) yabping@gmail.com
#
# Created Time: 2016-09-21 22:44:02
###############################################################################

import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def predict_by_rf(train_feature, train_label, test_feature):
    rf = RandomForestClassifier(n_estimators=200, n_jobs=10)
    rf.fit(train_feature, train_label)
    return rf.predict(test_feature)


def run():
    # The competition datafiles are in the directory ../input
    # Read competition data files:
    train_data = pd.read_csv("./input/train.csv")
    test_data = pd.read_csv("./input/test.csv")

    train_label, train_feature = train_data['label'], train_data.ix[:, 1:]
    predict_label = predict_by_rf(train_feature, train_label, test_data)

    # write to output
    with open("./output/submission.csv", "w") as f:
        f.write("ImageId,Label\n")
        for k, v in enumerate(predict_label):
            f.write("{},{}\n".format(k+1, v))

run()
