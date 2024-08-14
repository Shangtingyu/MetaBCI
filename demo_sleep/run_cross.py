# -*- coding: utf-8 -*-
# Organization: Hangzhou Meidang Digital Health Technology Co.
# Authors: Shangtingyu
# Date: 2024/7/25
# License: MIT License
import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score

import demo_sleep
from demo_sleep import save_res_pre
from metabci.brainda import datasets
from metabci.brainda.algorithms.deep_learning import np_to_th, TinySleepNet, DeepSleepNet, EEGNet
from metabci.brainda.algorithms.deep_learning.attnsleepnet import AttnSleepNet
from metabci.brainda.algorithms import deep_learning
from metabci.brainda.algorithms.utils.model_selection import EnhancedStratifiedKFold
from metabci.brainda.utils.performance import Performance


def cross_train_model(datas, model_name=AttnSleepNet,num_channels=1, num_classes=5, n_splits=5, model_selection=EnhancedStratifiedKFold):
    """
    Train the AttnSleep model using cross-validation.
    Parameters:
    datas (list): List of input datas, like [labels, data]
    n_splits (int): Number of folds for cross-validation.
    model_selection: Cross-validation model selection algorithm.
    Returns:
    path
    """
    label, input_data = datas[0], datas[1]
    random_state = 42
    X_train = np_to_th(input_data)
    y_train = np_to_th(label).long()
    kf = model_selection(n_splits=n_splits, shuffle=True, random_state=random_state)
    best_accs = []
    models = []
    for fold, (train_index, _, val_index) in enumerate(kf.split(X_train, y_train)):
        if model_name == AttnSleepNet:
            model = AttnSleepNet(num_channels, num_classes)
        elif model_name == DeepSleepNet:
            model = DeepSleepNet(num_classes)
        elif model_name == TinySleepNet:
            model = TinySleepNet(num_classes)
        elif model_name == EEGNet:
            X_train = X_train.to(torch.double)
            model = EEGNet(num_channels, 3000, num_classes)
        else:
            print('No such model')

        X_train_fold = X_train[train_index]
        y_train_fold = y_train[train_index]
        X_val_fold = X_train[val_index]
        y_val_fold = y_train[val_index]
        model.fit(X_train_fold, y_train_fold, valid_data=(X_val_fold, y_val_fold))

        best_acc = getattr(model, 'best_valid_acc', None)
        best_accs.append(best_acc)
        models.append(model)
        print(f"Fold {fold + 1} completed.")

    max_number = max(best_accs)
    max_index = best_accs.index(max_number)

    return models[max_index]


def main():
    path = r'D:\sleep-data\ST'  # 原始数据raw_data存储地址，没有则会自动下载
    dataPath = r'D:\sleep-data\ST\EEG Fpz-Cz'  # 数据预处理后的npz_data存储地址
    os.makedirs(path, exist_ok=True)
    os.makedirs(dataPath, exist_ok=True)
    num_classes = 5    # 睡眠分期的分类任务，支持2-5类
    num_channels = 1   # 睡眠分期的通道个数
    subjects = list(range(35))  # None则代表处理所有被试
    test_subjects = list(range(30, 35))
    train_subjects = list(set(subjects) - set(test_subjects))
    select_ch = ["EEG Fpz-Cz"]  # None则代表使用默认通道

    # 数据预处理与AI模型加载
    # 支持数据集:Sleep_telemetry可以替换为Sleep_cassette、SHHS、Mros、MSP、Apples
    sleep = datasets.Sleep_telemetry(dataPath=path)
    sleep.save_processed_data(update_path=dataPath, subjects=subjects, select_ch=select_ch)
    data = sleep.get_processed_data(update_path=dataPath, subjects=train_subjects, num_classes=num_classes)
    # 支持模型: AttnSleepNet、TinySleepNet、DeepSleepNet、EEGNet
    model_name = AttnSleepNet  # 选择模型
    model = cross_train_model(data, model_name, num_channels, num_classes)

    # 独立测试集评估模型
    test_datas = sleep.get_processed_data(update_path=dataPath, subjects=test_subjects, num_classes=num_classes)
    test_X, test_y = test_datas[1], test_datas[0]
    if model_name == EEGNet:
        test_X = test_X.astype(np.double)
    pre_y = model.predict(test_X)
    per = Performance(estimators_list=["Acc"])
    print(per.evaluate(pre_y, test_y))
    f1_scores = f1_score(pre_y, test_y, average=None)
    f1_scores = [round(num, 2) for num in f1_scores]
    print(f'F1 Scores for each class: {f1_scores}')
    # 绘制测试集的饼图
    demo_sleep.plotAnalyze(pre_y)
    # 绘制测试集的混淆矩阵
    demo_sleep.plot_confusion_matrix(pre_y, test_y)

    # 绘制特定被试的睡眠趋势图
    pre_subjects = [30]
    pre_datas = sleep.get_processed_data(update_path=dataPath, subjects=pre_subjects, num_classes=num_classes)
    pre_data = pre_datas[1]  # 0是标签，1是数据
    if model_name == EEGNet:
        pre_data = pre_data.astype(np.double)
    y_pre = model.predict(pre_data)
    y_pre_sm = demo_sleep.smooth(y_pre)  # 睡眠规则后处理平滑睡眠分期结果
    fig, axs = plt.subplots(3, 1, figsize=(15, 10))
    demo_sleep.plotTime(axs[0], pre_datas[0], flag_modi=False, color="black", name="PSG true label")
    demo_sleep.plotTime(axs[1], y_pre, flag_modi=False, color="GoldenRod", name="prediction")
    demo_sleep.plotTime(axs[2], y_pre_sm, flag_modi=False, color="GoldenRod", name="smooth prediction")
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
