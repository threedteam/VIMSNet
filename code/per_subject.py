# =========================================================================================================
# #       C Q UC Q UC Q UC Q U          C Q UC Q UC Q U              C Q U          C Q U
# # C Q U               C Q U     C Q U               C Q U          C Q U          C Q U
# # C Q U                         C Q U               C Q U          C Q U          C Q U
# # C Q U                         C Q U               C Q U          C Q U          C Q U
# # C Q U                         C Q U               C Q U          C Q U          C Q U
# # C Q U                         C Q UC Q UC Q U     C Q U          C Q U          C Q U
# # C Q U               C Q U     C Q U          C Q UC Q U          C Q U          C Q U
# #      C Q UC Q UC Q U               C Q UC Q UC Q U                    C Q UC Q U
# #                                              C Q UC Q U
# #
# #     Corresponding author：Ran Liu
# #     Address: College of Computer Science, Chongqing University, 400044, Chongqing, P.R.China
# #     Phone: +86 136 5835 8706
# #     Fax: +86 23 65111874
# #     Email: ran.liu_cqu@qq.com
# #
# #     Filename         : per_subject.py
# #     Description      : For more information, please refer to our paper
# #                        "EEG-Based Detection for Visually Induced Motion Sickness via One-Dimensional
# #                         Convolutional Neural Network"
# #   ----------------------------------------------------------------------------------------------
# #       Revision   |     DATA     |   Authors                                   |   Changes
# #   ----------------------------------------------------------------------------------------------
# #         1.00     |  2021-05-18  |   Shanshan Cui                              |   Initial version
# #   ----------------------------------------------------------------------------------------------
# # =========================================================================================================
#
# # -*- coding: utf-8 -*-

from sklearn.model_selection import train_test_split
from keras.layers import Input, Dense, Dropout, Conv1D, Flatten, concatenate, MaxPooling1D
from keras import losses, utils
from sklearn.preprocessing import StandardScaler
import numpy as np
import datetime
import pandas as pd
import keras
from sklearn.metrics import cohen_kappa_score, roc_auc_score
from keras.optimizers import SGD
from keras import layers
from keras.models import load_model, Model
from keras.layers import Multiply, GlobalAveragePooling1D, GlobalMaxPool1D
from keras.layers.core import *
from keras.models import *
import os


"""Per-subject binary classification"""


os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def per_subject_data(subject_no=1):
    print(subject_no)
    isFive = False  # Determine whether the classification label is binary or multiple (four-level)
    path2 = r'./data/absolute_psd_segment.xlsx'  # absolute power spectrum file
    data2 = pd.read_excel(path2)

    data = data2.loc[data2['subject_no'] == subject_no]

    if isFive:
        label = data.iloc[:, 21]  # multi-label (four-level)
    else:
        label = data.iloc[:, 22]  # binary label

    data = data.iloc[:, 1:21]

    data = np.asarray(data)
    label = np.asarray(label)

    # splitting training set and test set
    x_train, x_test, y_train, y_test = train_test_split(
        data, label,
        test_size=0.1,
        random_state=233,
        shuffle=True, stratify=label)

    # Normalization
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    x_train = x_train.reshape(-1, 20, 1)
    x_test = x_test.reshape(-1, 20, 1)

    x_train, x_eval, y_train, y_eval = train_test_split(
        x_train, y_train,
        test_size=0.1,
        random_state=233,
        shuffle=True, stratify=y_train)

    y_train = utils.to_categorical(y_train, 2)
    y_eval = utils.to_categorical(y_eval, 2)
    y_test = utils.to_categorical(y_test, 2)

    return x_train, x_eval, x_test, y_train, y_eval, y_test


def per_subject_model(filters1, filters2, filters3, units_dense):
    input = Input(shape=(20, 1))

    # Block_1
    x = Conv1D(filters=filters1, kernel_size=4, strides=2, padding='same', dilation_rate=1,
               activation='relu', use_bias=True, kernel_initializer='he_normal',
               bias_initializer='zeros', kernel_regularizer=regularizers.l2(0.001), bias_regularizer=None,
               activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(input)
    x = layers.BatchNormalization()(x)
    x = Dropout(0.3)(x)

    # Block_2
    x_x = Conv1D(filters=filters2, kernel_size=6, strides=2, padding='same', dilation_rate=1,
                 activation='relu', use_bias=True, kernel_initializer='he_normal',
                 bias_initializer='zeros', kernel_regularizer=regularizers.l2(0.001), bias_regularizer=None,
                 activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(input)
    x_x = layers.BatchNormalization()(x_x)
    x_x = Dropout(0.3)(x_x)

    # Block_3
    x_x_x = Conv1D(filters=filters3, kernel_size=2, strides=1, padding='same', dilation_rate=1,
                   activation='relu', use_bias=True, kernel_initializer='he_normal',
                   bias_initializer='zeros', kernel_regularizer=regularizers.l2(0.001), bias_regularizer=None,
                   activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(input)
    x_x_x = layers.BatchNormalization()(x_x_x)
    x_x_x = Dropout(0.3)(x_x_x)
    x_x_x = MaxPooling1D(2)(x_x_x)

    x = concatenate([x, x_x, x_x_x])

    # SE Block
    dim = int(x.shape[-1])
    a = Permute((2, 1))(x)
    a = Reshape((dim, 10))(a)
    squeeze = GlobalMaxPool1D()(a)
    excitation = Dense(units=10 // 2)(squeeze)
    excitation = Activation('relu')(excitation)
    excitation = Dense(units=10)(excitation)
    excitation = Activation('sigmoid')(excitation)
    excitation = Reshape((10, 1))(excitation)
    scale = Multiply()([x, excitation])

    x_x_x = Flatten()(scale)
    x = Dense(units=512, activation='relu')(x_x_x)
    x = Dropout(0.3)(x)
    feature = Flatten()(input)
    x = concatenate([feature, x])
    x = Dense(units_dense, activation='relu')(x)
    x = Dropout(0.3)(x)

    pred = Dense(2, activation='softmax',  bias_regularizer=regularizers.l2(0.001),
                 bias_initializer='he_normal', kernel_initializer='he_normal',
                 kernel_regularizer=regularizers.l2(0.001), activity_regularizer=regularizers.l2(0.001),)(x)
    model = Model(input, pred)
    model.compile(loss=losses.binary_crossentropy,
                  optimizer=SGD(lr=0.0125, decay=1e-6, momentum=0.9, nesterov=True),
                  metrics=['accuracy'])
    return model


def per_subject_fit(model, x_train, x_eval, y_train, y_eval, epochs, subject_no_test, num):
    callbacks_list = [keras.callbacks.ModelCheckpoint(
                filepath='./model/per-subject_' + str(subject_no_test) + '_' + str(num)
                         + '_{epoch:04d}-{val_acc:.4f}.h5',
                monitor='val_acc', verbose=0, save_best_only=True,
                save_weights_only=False, mode='max', period=1)]

    model.fit(x=x_train, y=y_train, batch_size=32, epochs=epochs, verbose=0,
              validation_data=(x_eval, y_eval), callbacks=callbacks_list, shuffle=True)


if __name__ == '__main__':
    nowTime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('Program start time：', nowTime)
    print("The program is running, please wait：")

    list_parament = [[32, 32, 32, 128],
                     [32, 16, 16, 128],
                     [32, 32, 16, 256],
                     [32, 32, 16, 256],
                     [32, 16, 16, 256],
                     [32, 32, 8, 256],
                     [32, 32, 32, 256],
                     [8, 16, 32, 128]]

    for j in range(1, 9):  # the individual subject number corresponding to the current training and test data
        x_train, x_eval, x_test, y_train, y_eval, y_test = per_subject_data(int(j))
        for i in range(1, 11):
            model = per_subject_model(int(list_parament[j - 1][0]), int(list_parament[j - 1][1]),
                                            int(list_parament[j - 1][2]), int(list_parament[j - 1][3]))
            per_subject_fit(model, x_train, x_eval, y_train, y_eval, 1000, j, i)

    nowTime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('Program end time：', nowTime)

    #  load the model with the best validation accuracy for testing
    for j in range(1, 9):
        x_train, x_eval, x_test, y_train, y_eval, y_test = per_subject_data(int(j))

        path = './model/best_model/per_subject_' + str(j) + '.h5'
        model_best = load_model(path)

        score = model_best.evaluate(x_test, y_test)
        print("loss:", score[0])
        print("accuracy:", score[1])
        print("the kappa index: %.4f" % cohen_kappa_score(np.argmax(y_test, axis=1),
                                                          np.argmax(model_best.predict(x_test), axis=1)))
        y_score = model_best.predict(x_test)
        auc_score = roc_auc_score(y_test, y_score)
        print("1D-CNN auc_score:", auc_score)
