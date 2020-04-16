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
# #     Filename         : contrast_experiment_RBP_FEEG or DeepCNN.py
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


from __future__ import print_function
import numpy as np
import pandas as pd
import datetime
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Input, concatenate
from keras.layers import Conv1D, MaxPooling1D
from keras.optimizers import SGD
from keras import losses,utils
from keras import layers
from sklearn.preprocessing import StandardScaler

# ========================================================================================================================================
#deeper network. Increases the depth of a one-dimensional convolutional neural network, contrast this with the proposed network
# ========================================================================================================================================

# ====================================================================================
# Construct a deeper one-dimensional convolution model and return the model
# ====================================================================================

def Con1D():
    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=3, strides=2, padding='same', dilation_rate=1,
           activation='relu', use_bias=True, kernel_initializer='glorot_uniform',
           bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
           activity_regularizer=None, kernel_constraint=None, bias_constraint=None,
           name='conv1D_1',input_shape=(20,1)))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(2))

    model.add(Conv1D(filters=8, kernel_size=3, strides=1, padding='same', dilation_rate=1,
                     activation='relu', use_bias=True, kernel_initializer='glorot_uniform',
                     bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
                     activity_regularizer=None, kernel_constraint=None, bias_constraint=None,
                     name='conv1D_2'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(2))

    model.add(Conv1D(filters=16, kernel_size=3, strides=1, padding='same', dilation_rate=1,
                     activation='relu', use_bias=True, kernel_initializer='glorot_uniform',
                     bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
                     activity_regularizer=None, kernel_constraint=None, bias_constraint=None,
                     name='conv1D_3'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(2))

    model.add(Flatten())

    model.add(Dense(512,activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(128,activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(2,activation='softmax'))
    return model

# ========================================================================
# read data and train the deeper one-dimensional convolution model
# ========================================================================

def Deeper_Conv1d():
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

    data = data.reshape(-1, 20, 1)

    x_train, x_test, y_train, y_test = train_test_split(
        data, label,
        test_size=0.1,
        random_state=233,
        shuffle=True, stratify=label)

    y_train = utils.to_categorical(y_train, 2)
    y_test = utils.to_categorical(y_test, 2)

    model = Con1D()

    model.compile(loss=losses.binary_crossentropy,
                  optimizer=SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True),
                  metrics=['accuracy'])
    model.fit(x=x_train, y=y_train,
              batch_size=32,
              epochs=200, verbose=0,
              validation_split=0.1)

    score = model.evaluate(x_test, y_test)
    print("loss:",score[0])
    print("accuracy:",score[1])
    print("the kappa index: %.4f" % cohen_kappa_score(np.argmax(y_test, axis=1),
                                                      np.argmax(model.predict(x_test), axis=1)))

# ========================================================================
#Construct the proposed one-dimensional convolution model and return the model
# ========================================================================

def Con1D_concate(filters1, filters2, filters3,absolute=1):

    if absolute==1:
        input = Input(shape=(40, 1))
    else:
        input = Input(shape=(20, 1))


    x = Conv1D(filters=filters1, kernel_size=4, strides=2, padding='same', dilation_rate=1,
               activation='relu', use_bias=True, kernel_initializer='glorot_uniform',
               bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
               activity_regularizer=None, kernel_constraint=None, bias_constraint=None,
               name='conv1D_1')(input)
    x=layers.BatchNormalization()(x)
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
    x = Dense(512, activation='relu', name='dense_1')(x)
    x = Dropout(0.3)(x)

    feature = Flatten()(input)
    x = concatenate([feature, x])

    x = Dense(128, activation='relu', name='dense_2')(x)
    x = Dropout(0.3)(x)
    pred = Dense(2, activation='softmax', name='dense_3')(x)

    model = Model(input,pred)

    return model

# =================================================================================================================
# read RBP，FEEG data，respectively， and train the proposed one-dimensional convolution model， respectively
# =================================================================================================================

def EEG_1d(absolute=1):
    isFive=False
    path1 = r'C:\Users\Administrator\Desktop\论文\relative_psd_downsampled.xlsx'
    path2 = r'C:\Users\Administrator\Desktop\论文\absolute_psd_downsampled.xlsx'
    data1 = pd.read_excel(path1)
    data2 = pd.read_excel(path2)
    if isFive:
        label = data1.iloc[:, 21]
    else:
        label = data1.iloc[:, 22]
    X1 = data2.iloc[:, 1:21]
    X2 = data1.iloc[:, 1:21]
    if absolute==1:
        data = pd.concat((X1, X2), axis=1)
        data = np.asarray(data)
    elif absolute == 2:
        data = X2
        data = np.asarray(data)
    else:
        data = X1
        data = np.asarray(data)

    scaler = StandardScaler()
    data = scaler.fit_transform(data)

    if absolute == 1:
        data = data.reshape(-1, 40, 1)
    else:
        data=data.reshape(-1,20,1)

    x_train, x_test, y_train, y_test = train_test_split(
        data, label,
        test_size=0.1,
        random_state=233,
        shuffle=True,stratify=label)

    y_train = utils.to_categorical(y_train, 2)
    y_test = utils.to_categorical(y_test, 2)

    model = Con1D_concate(32, 8, 16, absolute)

    model.compile(loss=losses.binary_crossentropy,
                  optimizer=SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True),
                  metrics=['accuracy'])
    model.fit(x=x_train, y=y_train,
              batch_size=32,
              epochs=200, verbose=0,
              validation_split=0.1)

    score = model.evaluate(x_test, y_test)
    print("loss:",score[0])
    print("accuracy:",score[1])
    print("the kappa index: %.4f" % cohen_kappa_score(np.argmax(y_test, axis=1),
                                                      np.argmax(model.predict(x_test), axis=1)))

if __name__ == '__main__':
    nowTime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('Program start time：', nowTime)
    print("The program is running, please wait：")

    for i in range(1, 11):
        Deeper_Conv1d()  # deeper one-dimensional convolution

    for i in range(1,11):
        EEG_1d(absolute=2)  # RBP data
        EEG_1d(absolute=1)  # FEEG data (RBP + ABP)





