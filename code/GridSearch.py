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
# #     Filename         : GridSearch.py
# #     Description      : For more information, please refer to our paper
# #                        "Electroencephalogram-Based Detection for Visually Induced Motion Sickness via
# #                        One-Dimensional Convolutional Neural Network"
# #   ----------------------------------------------------------------------------------------------
# #       Revision   |     DATA     |   Authors                                   |   Changes
# #   ----------------------------------------------------------------------------------------------
# #         1.00     |  2020-04-16  |   Shanshan Cui                              |   Initial version
# #   ----------------------------------------------------------------------------------------------
# # =========================================================================================================
#
# # -*- coding: utf-8 -*-


from sklearn.model_selection import train_test_split
from keras.layers import Input, Dense, Dropout, Conv1D, Flatten, concatenate, MaxPooling1D
from keras.models import Model, load_model
from keras import losses, utils
from sklearn.preprocessing import StandardScaler
import numpy as np
from keras.optimizers import SGD
from keras import layers
import datetime

# ------------------------------------------------------------------------------
# multiple  subject four classifier ((adjust parameters with grid search)
# ------------------------------------------------------------------------------

# ========================================================================
# read data, return to the training set, test set, labels
# ========================================================================

cls = 4

import pandas as pd

def readData():
    isFive = True  # Determine whether the classification label is binary or multiple
    path2 = r'C:\Users\Administrator\Desktop\论文\absolute_psd_downsampled.xlsx'  # absolute power spectrum file
    data2 = pd.read_excel(path2)
    if isFive:
        label = data2.iloc[:, 21]  # multi-label
    else:
        label = data2.iloc[:, 22]  # binary label

    X1 = data2.iloc[:, 1:21]
    data = np.asarray(X1)

    scaler = StandardScaler()
    data = scaler.fit_transform(data)

    data = data.reshape(-1, 20, 1)

    x_train, x_test, y_train, y_test = train_test_split(
        data, label,
        test_size=0.1,
        random_state=233,
        shuffle=True,stratify=label)

    y_train = utils.to_categorical(y_train, cls)
    y_test = utils.to_categorical(y_test, cls)

    return x_train, x_test, y_train, y_test

# ========================================================================
# Construct and return a one-dimensional convolution model
# ========================================================================

def Con1D_concate(filters1, filters2, filters3, lr, units_dense=128):

    input = Input(shape=(20, 1))

    x = Conv1D(filters=filters1, kernel_size=4, strides=2, padding='same', dilation_rate=1,
               activation='relu', use_bias=True, kernel_initializer='glorot_uniform',
               bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
               activity_regularizer=None, kernel_constraint=None, bias_constraint=None,
               name='conv1D_1')(input)
    x = layers.BatchNormalization()(x)
    x = MaxPooling1D(2)(x)
    x = Flatten()(x)


    x_x = Conv1D(filters=filters2, kernel_size=6, strides=3, padding='same', dilation_rate=1,
                 activation='relu', use_bias=True, kernel_initializer='glorot_uniform',
                 bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
                 activity_regularizer=None, kernel_constraint=None, bias_constraint=None,
                 name='conv1D_2')(input)
    x_x = layers.BatchNormalization()(x_x)
    x_x = MaxPooling1D(2)(x_x)
    x_x = Flatten()(x_x)

    x_x_x = Conv1D(filters=filters3, kernel_size=2, strides=1, padding='same', dilation_rate=1,
                   activation='relu', use_bias=True, kernel_initializer='glorot_uniform',
                   bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
                   activity_regularizer=None, kernel_constraint=None, bias_constraint=None,
                   name='conv1D_3')(input)
    x_x_x = layers.BatchNormalization()(x_x_x)
    x_x_x = MaxPooling1D(2)(x_x_x)
    x_x_x = Flatten()(x_x_x)

    x = concatenate([x, x_x, x_x_x])

    x = Dense(units=512, activation='relu', name='dense_1')(x)
    x = Dropout(0.3)(x)

    feature = Flatten()(input)

    x = concatenate([feature, x])

    x = Dense(units=units_dense, activation='relu', name='dense_2')(x)
    x = Dropout(0.3)(x)
    pred = Dense(cls, activation='softmax', name='dense_3')(x)

    model = Model(input,pred)
    model.compile(loss=losses.categorical_crossentropy,
                  optimizer=SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True),
                  metrics=['accuracy'])
    return model

def EEG_1d():
    x_train, x_test, y_train, y_test = readData()

    x_training, x_val, y_training, y_val = train_test_split(
        x_train, y_train,
        test_size=0.1,
        random_state=233,
        shuffle=True,stratify=y_train)

    best_score=0
    for units_dense in [256, 128]:
        for filters1 in [8, 16, 32]:
            for filters2 in [8, 16, 32]:
                for filters3 in [8, 16, 32]:
                            model = Con1D_concate(filters1, filters2, filters3, 0.01, units_dense)
                            model.fit(x=x_training, y=y_training,
                                      batch_size=32,
                                      epochs=200, verbose=0,
                                      validation_split=0.1)
                            score = model.evaluate(x_val, y_val)
                            if score[1] > best_score:
                                best_score = score[1]
                                best_parameters = {'filters1': filters1, "filters2": filters2, 'filters3': filters3, "units_dense": units_dense}
                                with open("C:\\Users\Administrator\Desktop\论文\egg1D.txt", "a+") as f:
                                    f.write(str(filters1))
                                    f.write(str(filters2))
                                    f.write(str(filters3))
                                    f.write(str(units_dense))
                                    f.write(str(best_score))
                                    f.write('\n')
                                print(best_parameters)
    print(best_parameters)
    print(best_score)

# ------------------------------------------------------------------------------
# multiple  subject 2 classifier ((adjust parameters with grid search)
# ------------------------------------------------------------------------------

def readData_2classifier():
    isFive = False  # Determine whether the classification label is binary or multiple
    path2 = r'C:\Users\Administrator\Desktop\论文\absolute_psd_downsampled.xlsx'  # absolute power spectrum file
    data2 = pd.read_excel(path2)
    if isFive:
        label = data2.iloc[:, 21]  # multi-label
    else:
        label = data2.iloc[:, 22]  # binary label

    X1 = data2.iloc[:, 1:21]
    data = np.asarray(X1)

    scaler = StandardScaler()
    data = scaler.fit_transform(data)

    data=data.reshape(-1,20,1)

    x_train, x_test, y_train, y_test = train_test_split(
        data, label,
        test_size=0.1,
        random_state=233,
        shuffle=True,stratify=label)

    y_train = utils.to_categorical(y_train, 2)
    y_test = utils.to_categorical(y_test, 2)

    return x_train, x_test, y_train, y_test

def Con1D_concate_2classifier(filters1, filters2, filters3, lr, units_dense=128):

    input = Input(shape=(20, 1))

    x = Conv1D(filters=filters1, kernel_size=4, strides=2, padding='same', dilation_rate=1,
               activation='relu', use_bias=True, kernel_initializer='glorot_uniform',
               bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
               activity_regularizer=None, kernel_constraint=None, bias_constraint=None,
               name='conv1D_1')(input)
    x = layers.BatchNormalization()(x)
    x = MaxPooling1D(2)(x)
    x = Flatten()(x)


    x_x = Conv1D(filters=filters2, kernel_size=6, strides=3, padding='same', dilation_rate=1,
                 activation='relu', use_bias=True, kernel_initializer='glorot_uniform',
                 bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
                 activity_regularizer=None, kernel_constraint=None, bias_constraint=None,
                 name='conv1D_2')(input)
    x_x = layers.BatchNormalization()(x_x)
    x_x = MaxPooling1D(2)(x_x)
    x_x = Flatten()(x_x)

    x_x_x = Conv1D(filters=filters3, kernel_size=2, strides=1, padding='same', dilation_rate=1,
                   activation='relu', use_bias=True, kernel_initializer='glorot_uniform',
                   bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
                   activity_regularizer=None, kernel_constraint=None, bias_constraint=None,
                   name='conv1D_3')(input)
    x_x_x = layers.BatchNormalization()(x_x_x)
    x_x_x = MaxPooling1D(2)(x_x_x)
    x_x_x = Flatten()(x_x_x)

    x = concatenate([x, x_x, x_x_x])

    x = Dense(units=512, activation='relu', name='dense_1')(x)
    x = Dropout(0.3)(x)

    feature = Flatten()(input)

    x = concatenate([feature, x])

    x = Dense(units=units_dense, activation='relu', name='dense_2')(x)
    x = Dropout(0.3)(x)
    pred = Dense(2, activation='softmax', name='dense_3')(x)

    model = Model(input,pred)
    model.compile(loss=losses.binary_crossentropy,
                  optimizer=SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True),
                  metrics=['accuracy'])
    return model

def EEG_1d_2classifier():

    x_train, x_test, y_train, y_test = readData_2classifier()
    x_training, x_val, y_training, y_val = train_test_split(
        x_train, y_train,
        test_size=0.1,
        random_state=233,
        shuffle=True,stratify=y_train)

    best_score=0
    for units_dense in [256, 128]:
        for filters1 in [8, 16, 32]:
            for filters2 in [8, 16, 32]:
                for filters3 in [8, 16, 32]:
                            model = Con1D_concate_2classifier(filters1, filters2, filters3, 0.01, units_dense)
                            model.fit(x=x_training, y=y_training,
                                      batch_size=32,
                                      epochs=200, verbose=0,
                                      validation_split=0.1,
                                      )
                            score = model.evaluate(x_val, y_val)
                            if score[1] > best_score:
                                best_score = score[1]
                                best_parameters = {'filters1': filters1, "filters2": filters2,'filters3': filters3, "units_dense": units_dense}
                                with open("C:\\Users\Administrator\Desktop\论文\egg1D.txt", "a+") as f:
                                    f.write(str(filters1))
                                    f.write(str(filters2))
                                    f.write(str(filters3))
                                    f.write(str(units_dense))
                                    f.write(str(best_score))
                                    f.write('\n')
                                print(best_parameters)
    print(best_parameters)
    print(best_score)

if __name__ == '__main__':
    nowTime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('Program start time：', nowTime)
    print("The program is running, please wait：")

    EEG_1d_2classifier()  # multiple subject 2 classification  GridSearch

    EEG_1d()  # multiple subject 4 classification GridSearch

