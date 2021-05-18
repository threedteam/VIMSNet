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
# #     Filename         : loocv.py
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

import torch as nn
import torch.optim as optim
from torchvision import transforms, models, datasets
import torchvision
from torch import nn
import torch
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras import utils
import datetime
from sklearn.metrics import cohen_kappa_score, roc_auc_score


"""LOOCV"""


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def loocv_data(test_subject_no):
    print(test_subject_no)
    isFive = False  # Determine whether the classification label is binary or multiple (four-level)
    path2 = r'./data/absolute_psd_segment.xlsx'
    data2 = pd.read_excel(path2)

    if isFive:
        label = data2.loc[data2['subject_no'] != test_subject_no, 'VIMSL']  # multi-label (four-level)
    else:
        label = data2.loc[data2['subject_no'] != test_subject_no, 'VIMSL.1']  # binary label

    x_train = data2.loc[data2['subject_no'] != test_subject_no, 'alpha_absolute1': 'theta_absolute4']

    x_train = np.asarray(x_train)
    y_train = np.asarray(label)

    x_test = data2.loc[data2['subject_no'] == test_subject_no, 'alpha_absolute1': 'theta_absolute4']
    y_test = data2.loc[data2['subject_no'] == test_subject_no, 'VIMSL.1']  # binary label

    x_test = np.asarray(x_test)
    y_test = np.asarray(y_test)

    # Normalization
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)

    scaler = StandardScaler()
    x_test = scaler.fit_transform(x_test)

    x_train = x_train.reshape(-1, 1, 20)
    x_test = x_test.reshape(-1, 1, 20)

    # splitting training set and validation set
    x_train, x_eval, y_train, y_eval = train_test_split(
        x_train, y_train,
        test_size=0.1,
        random_state=233,
        shuffle=True, stratify=y_train)

    y_train = utils.to_categorical(y_train, 2)
    y_eval = utils.to_categorical(y_eval, 2)
    y_test = utils.to_categorical(y_test, 2)

    x_train = torch.from_numpy(x_train)
    y_train = torch.from_numpy(y_train)
    x_train = torch.tensor(x_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    train_ds = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

    x_eval = torch.from_numpy(x_eval)
    y_eval = torch.from_numpy(y_eval)
    x_eval = torch.tensor(x_eval, dtype=torch.float32)
    y_eval = torch.tensor(y_eval, dtype=torch.float32)
    val_ds = TensorDataset(x_eval, y_eval)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=True)

    x_test = torch.from_numpy(x_test)
    y_test = torch.from_numpy(y_test)
    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)
    test_ds = TensorDataset(x_test, y_test)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=True)

    return train_loader, val_loader, test_loader


class CNN_1D(nn.Module):
    def __init__(self):
        super(CNN_1D, self).__init__()
        self.Block1 = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=32,
                kernel_size=4,
                stride=2,
                padding=1,
                dilation=1
            ),
            nn.BatchNorm1d(32),
            nn.Dropout(0.3),
            nn.ReLU(),
        )

        self.Block2 = nn.Sequential(
            nn.Conv1d(1, 8, 6, 2, 2, 1),
            nn.BatchNorm1d(8),
            nn.Dropout(0.3),
            nn.ReLU(),
        )

        self.Block3 = nn.Sequential(
            nn.Conv1d(1, 8, 2, 1, 1, 1),
            nn.BatchNorm1d(8),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        self.squeeze = nn.AdaptiveMaxPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(inplace=True),
            nn.Linear(5, 10),
            nn.Sigmoid(),
        )

        self.dense_1 = nn.Sequential(
            nn.Linear(48*10, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )

        self.dense_2 = nn.Sequential(
            nn.Linear(532, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )

        self.out = nn.Linear(128, 2)
        nn.init.constant(self.out.bias, 0.1)
        torch.nn.init.kaiming_normal_(self.out.weight)

    def forward(self, x):
        x_1 = self.Block1(x)
        x_2 = self.Block2(x)
        x_3 = self.Block3(x)

        x_cat = torch.cat([x_1, x_2, x_3], 1)

        a = x_cat.permute(0, 2, 1)
        squeeze = self.squeeze(a)
        squeeze = squeeze.view(-1, 10)
        excitation = self.excitation(squeeze)
        excitation = excitation.view(-1, 1, 10)

        scale = torch.mul(x_cat, excitation, out=None)

        input_dense = scale.view(scale.size(0), -1)

        input_dense = self.dense_1(input_dense)

        x_input_dense = x.view(x.size(0), -1)

        input_dense = torch.cat([input_dense, x_input_dense], 1)

        input_dense = self.dense_2(input_dense)

        out = self.out(input_dense)

        return out


def weight_init(m):
    if isinstance(m, nn.Conv1d):
        torch.nn.init.kaiming_normal_(m.weight)
        nn.init.constant_(m.bias, 0.0)


loss_func = nn.CrossEntropyLoss()
epochs = 20


def CNN_1D_fit_eval(model, train_loader, val_loader, max_checkpoint_acc, subject_nun, num):
    bias_list = (param for name, param in model.named_parameters() if name[-4:] == 'bias' or "bn" in name)
    others_list = (param for name, param in model.named_parameters() if name[-4:] != 'bias' and "bn" not in name)
    parameters = [{'params': bias_list, 'weight_decay': 0},
                  {'params': others_list}]
    # optimizer = torch.optim.SGD(parameters, lr=0.0125, momentum=0.9, weight_decay=0.001)
    # optimizer = torch.optim.Adam(parameters, lr=0.0125, weight_decay=0.0001)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.0125, momentum=0.9, weight_decay=0.001)

    max_val_acc = 0
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_corrects = 0.0
        total_train = 0
        total_train = torch.tensor(total_train, dtype=torch.float32)
        for step, (data, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            data = data.to(device)
            targets = targets.to(device)
            output = model(data)
            loss = loss_func(output, targets.argmax(1))
            loss.backward()
            optimizer.step()

            train_loss += loss
            train_corrects += (output.argmax(1) == targets.argmax(1)).sum()
            total_train += data.size(0)
        epoch_loss = train_loss / len(train_loader)
        epoch_acc = train_corrects / total_train

        # print('epoch: {}; 训练损失Loss: {:.4f}； 训练精度Acc: {:.4f}'.format(epoch, epoch_loss, epoch_acc))

        model.eval()
        eval_loss = 0.0
        eval_corrects = 0.0
        total_val = 0
        total_val = torch.tensor(total_val, dtype=torch.float32)
        for step, (data, targets) in enumerate(val_loader):
            data = data.to(device)
            targets = targets.to(device)
            output = model(data)
            loss = loss_func(output, targets.argmax(1))
            eval_loss += loss
            eval_corrects += (output.argmax(1) == targets.argmax(1)).sum()
            total_val += data.size(0)
        epoch_val_loss = eval_loss / len(val_loader)
        epoch_val_acc = eval_corrects / total_val

        if epoch_val_acc > max_val_acc:
            path_model = './model/loocv_' + str(subject_nun) + '_' + str(num) + '_model.pth'
            torch.save(model, path_model)
            max_val_acc = epoch_val_acc

        if epoch_val_acc > max_checkpoint_acc:
            path_model = './model/loocv_max_' + str(subject_nun) + '_model.pth'
            torch.save(model, path_model)
            max_checkpoint_acc = epoch_val_acc

        # print('验证损失Loss: {:.4f}； 验证精度Acc: {:.4f}'.format(epoch_val_loss, epoch_val_acc))
    return max_checkpoint_acc


def AdaBN(test_loader, model):
    state_dict = model.state_dict()
    state_dict['Block1.1.running_mean'] = torch.zeros(32)
    state_dict['Block1.1.running_var'] = torch.zeros(32)
    state_dict['Block1.1.num_batches_tracked'] = torch.tensor(0)
    state_dict['Block2.1.running_mean'] = torch.zeros(8)
    state_dict['Block2.1.running_var'] = torch.zeros(8)
    state_dict['Block2.1.num_batches_tracked'] = torch.tensor(0)
    state_dict['Block3.1.running_mean'] = torch.zeros(8)
    state_dict['Block3.1.running_var'] = torch.zeros(8)
    state_dict['Block3.1.num_batches_tracked'] = torch.tensor(0)
    model.load_state_dict(state_dict)

    model.train()
    with torch.no_grad():
        for data, targets in test_loader:
            data = data.to(device)
            targets = targets.to(device)
            output = model(data)


def CNN_1D_test(model, test_loader):
    model.eval()
    test_loss = 0.0
    test_acc = 0.0
    kappa = 0.0
    auc = 0.0
    total_test = 0
    total_test = torch.tensor(total_test, dtype=torch.float32)
    with torch.no_grad():
        for data, targets in test_loader:
            data = data.to(device)
            targets = targets.to(device)
            output = model(data)
            test_loss += loss_func(output, targets.argmax(1))
            test_acc += (output.argmax(1) == targets.argmax(1)).sum()
            kappa += cohen_kappa_score(targets.argmax(1).cuda().data.cpu().numpy(), output.argmax(1).cuda().data.cpu().numpy())
            auc += roc_auc_score(targets.cuda().data.cpu().numpy(), output.cuda().data.cpu().numpy())
            total_test += data.size(0)

    print('test loss: {:.4f}； test acc: {:.4f}; Kappa:  {:.4f}; AUC:  {:.4f};'.format
          (test_loss/len(test_loader), test_acc/total_test, kappa/len(test_loader), auc/len(test_loader)))

    return test_acc/total_test,  kappa/len(test_loader), auc/len(test_loader)


if __name__ == '__main__':
    nowTime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('Program start time：', nowTime)
    print("The program is running, please wait：")

    list_acc_all = []
    list_kappa_all = []
    list_auc_all = []

    for j in range(1, 9):  # subject number used as test set
        max_checkpoint_acc = 0
        train_loader, val_loader, test_loader = loocv_data(j)
        for i in range(10):
            model = CNN_1D().to(device)
            model.apply(weight_init)
            max_checkpoint_acc_return = CNN_1D_fit_eval(model, train_loader, val_loader, max_checkpoint_acc, j, i)
            max_checkpoint_acc = max_checkpoint_acc_return

    #  load the model with the best validation accuracy for testing
    for j in range(1, 9):
        train_loader, val_loader, test_loader = loocv_data(j)

        # load the model that has been processed by AdaBN
        path_model = './model/best_model/loocv_' + str(j) + '.pth'
        model_best = torch.load(path_model)
        acc, kappa, auc = CNN_1D_test(model_best, test_loader)

        # # load the model that has not been processed by AdaBN
        # path_model = './model/best_model/loocv_' + str(j) + '.pth'
        # model_best = torch.load(path_model)
        # AdaBN(test_loader, model_best)
        # path_model = './model/best_model/loocv_' + str(j) + '.pth'
        # torch.save(model_best, path_model)
        # path_model = './model/best_model/loocv_' + str(j) + '.pth'
        # model_best = torch.load(path_model)
        # acc, kappa, auc = CNN_1D_test(model_best, test_loader)

