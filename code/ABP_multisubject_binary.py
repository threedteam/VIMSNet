# =========================================================================================================
#       C Q UC Q UC Q UC Q U          C Q UC Q UC Q U              C Q U          C Q U
# C Q U               C Q U     C Q U               C Q U          C Q U          C Q U
# C Q U                         C Q U               C Q U          C Q U          C Q U
# C Q U                         C Q U               C Q U          C Q U          C Q U
# C Q U                         C Q U               C Q U          C Q U          C Q U
# C Q U                         C Q UC Q UC Q U     C Q U          C Q U          C Q U
# C Q U               C Q U     C Q U          C Q UC Q U          C Q U          C Q U
#      C Q UC Q UC Q U               C Q UC Q UC Q U                    C Q UC Q U
#                                              C Q UC Q U
#
#     Corresponding author：Ran Liu
#     Address: College of Computer Science, Chongqing University, 400044, Chongqing, P.R.China
#     Phone: +86 136 5835 8706
#     Fax: +86 23 65111874
#     Email: ran.liu_cqu@qq.com
#
#     Filename         : ABP_multisubject_binary.py
#     Description      : For more information, please refer to our paper
#                        "Electroencephalogram-Based Detection for Visually Induced Motion Sickness via
#                        One-Dimensional Convolutional Neural Network"
#   ----------------------------------------------------------------------------------------------
#       Revision   |     DATA     |   Authors                                   |   Changes
#   ----------------------------------------------------------------------------------------------
#         1.00     |  2020-02-29  |   Shanshan Cui                              |   Initial version
#   ----------------------------------------------------------------------------------------------
# =========================================================================================================

# -*- coding: utf-8 -*-


from sklearn.model_selection import train_test_split
from keras.layers import Input, Dense, Dropout, Conv1D, Flatten, concatenate, MaxPooling1D
from keras.models import Model
from keras import losses, utils
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from keras.optimizers import SGD
from keras import layers
import keras
from sklearn.metrics import cohen_kappa_score, roc_auc_score
import datetime

# ========================================================================
# read data, return to the training set, test set, labels
# ========================================================================

def readData():
    isFive=False  # Determine whether the classification label is binary or multiple
    path2 = r'C:\Users\Administrator\Desktop\论文\absolute_psd_segment.xlsx'  # absolute power spectrum file
    data2 = pd.read_excel(path2)
    if isFive:
        label = data2.iloc[:, 21]  # multi-label
    else:
        label = data2.iloc[:, 22]  # binary label

    X1 = data2.iloc[:, 1:21]
    data = np.asarray(X1)

    scaler = StandardScaler()  # data standardization
    data = scaler.fit_transform(data)

    data=data.reshape(-1, 20, 1)

    x_train, x_test, y_train, y_test = train_test_split(
        data, label,
        test_size=0.1,
        random_state=233,
        shuffle=True, stratify=label)

    y_train = utils.to_categorical(y_train, 2)
    y_test = utils.to_categorical(y_test, 2)

    return  x_train, x_test, y_train, y_test

# ========================================================================
# Construct and return a one-dimensional convolution model
# ========================================================================

def Con1D_concate(filters1, filters2, filters3, lr):
    # The value of those parameter(filters1, filters2, filters3, lr) is determined by the grid search method

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

    x = Dense(128, activation='relu', name='dense_2')(x)
    x = Dropout(0.3)(x)
    pred = Dense(2, activation='softmax', name='dense_3')(x)

    model = Model(input, pred)
    model.compile(loss=losses.binary_crossentropy,
                  optimizer=SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True),
                  metrics=['accuracy'])
    return model

# ========================================================================
# Train the one-dimensional convolution model
# ========================================================================

def EEG_1d(num=1):

    x_train, x_test, y_train, y_test = readData()  # prepare training and test data

    model = Con1D_concate(32, 8, 16, 0.01)  # generating model

    callbacks_list = [
        keras.callbacks.ModelCheckpoint(
            filepath='D:\\modelSave\\eleven_changeinput_test0.1_ABP_leaf_1dconv_ModelCheckpoint_' + str(num) + '.h5',
            monitor='val_accuracy',
            verbose=0, save_best_only=True, save_weights_only=False,
            mode='max', period=1
        )]

    model.fit(x=x_train, y=y_train,batch_size=32,epochs=200, verbose=0,
              validation_split=0.1, callbacks=callbacks_list, shuffle=True)

    path = 'D:\\modelSave\\eleven_changeinput_test0.1_ABP_leaf_1dconv_' + str(num) + '.h5'
    model.save(path)

    score = model.evaluate(x_test, y_test)
    print("loss:", score[0])
    print("accuracy:", score[1])
    print("the kappa index: %.4f" % cohen_kappa_score(np.argmax(y_test, axis=1),
                                                      np.argmax(model.predict(x_test), axis=1)))
    y_score=model.predict(x_test)
    auc_score = roc_auc_score(y_test, y_score)
    print("1D-CNN auc_score:", auc_score)

if __name__ == '__main__':
    nowTime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('Program start time：', nowTime)
    print("The program is running, please wait：")

    for i in range(1, 11):
        EEG_1d(num=i)


