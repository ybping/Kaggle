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
import numpy as np


def predict_by_cnn(X_train, Y_train, X_predict):
    from keras.models import Sequential
    from keras.layers import Convolution1D, Activation, Dense

    models = Sequential()
    models.add(Convolution1D(32, 3, input_shape=(42000, 28*28)))
    models.add(Activation('relu'))
    models.add(Dense(10))
    models.add(Activation('softmax'))

    models.compile(
            optimizer='adadelta',
            loss='categorical_crossentropy',
            metrics=['accuracy']
            )
    models.fit(
            X_train.values, np.array(Y_train),
            nb_epoch=10, batch_size=32
            )
    print models.predict(X_predict.values)


def run():
    # The competition datafiles are in the directory ../input
    # Read competition data files:
    train_data = pd.read_csv("./input/train.csv")
    test_data = pd.read_csv("./input/test.csv")

    train_label, train_feature = train_data['label'], train_data.ix[:, 1:]
    ################
    # predict by rf#
    ################
    #  predict_label = predict_by_rf(train_feature, train_label, test_data)
    predict_by_cnn(train_feature, train_label, test_data)

    # write to output
    with open("./output/submission.csv", "w") as f:
        f.write("ImageId,Label\n")
        #  for k, v in enumerate(predict_label):
        #  f.write("{},{}\n".format(k+1, v))

run()
