# -*- coding: utf-8 -*-
# Organization: Hangzhou Meidang Digital Health Technology Co.
# Authors: Shangtingyu
# Date: 2024/7/25
# License: MIT License
import os

from matplotlib import pyplot as plt
from sklearn.metrics import f1_score

import demo_sleep
from demo_sleep import save_res_pre
from metabci.brainda import datasets
from metabci.brainda.algorithms.deep_learning import np_to_th, TinySleepNet, DeepSleepNet
from metabci.brainda.algorithms.deep_learning.attnsleepnet import AttnSleepNet
from metabci.brainda.algorithms import deep_learning
from metabci.brainda.algorithms.utils.model_selection import EnhancedStratifiedKFold
from metabci.brainda.utils.performance import Performance


def cross_train_model(datas, n_splits=5, model_params=(1, 5), model_selection=EnhancedStratifiedKFold):
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
    names = []
    for fold, (train_index, _, val_index) in enumerate(kf.split(X_train, y_train)):
        model = AttnSleepNet(model_params[0], model_params[1])
        X_train_fold = X_train[train_index]
        y_train_fold = y_train[train_index]
        X_val_fold = X_train[val_index]
        y_val_fold = y_train[val_index]
        model.fit(X_train_fold, y_train_fold, valid_data=(X_val_fold, y_val_fold))

        best_acc = getattr(model, 'best_valid_acc', None)
        best_accs.append(best_acc)
        names.append(model.name)
        print(f"Fold {fold + 1} completed.")

    max_number = max(best_accs)
    max_index = best_accs.index(max_number)
    name = names[max_index]
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(script_dir, name, 'params.pt')
    return path


def main():
    path = r'D:\sleep-data\ST'  # 原始数据raw_data存储地址，没有则会自动下载
    dataPath = r'D:\sleep-data\ST\EEG Fpz-Cz'  # 数据预处理后的npz_data存储地址
    os.makedirs(dataPath, exist_ok=True)
    subjects = list(range(1))  # None则代表处理所有被试
    select_ch = ["EEG Fpz-Cz"]  # None则代表使用默认通道
    # 睡眠分期的分类任务，支持2-5类
    num_classes = 5
    num_channels = 1
    # 数据预处理与加载
    # 支持数据集:Sleep_telemetry可以替换为Sleep_cassette、SHHS、Mros、MSP、Apples
    sleep = datasets.Sleep_telemetry(dataPath=path)
    sleep.save_processed_data(update_path=dataPath, subjects=subjects, select_ch=select_ch)
    data = sleep.get_processed_data(update_path=dataPath, subjects=subjects, num_classes=num_classes)
    model_params = (num_channels, num_classes)  # 模型的通道和分类门数
    model_path = cross_train_model(data, model_params=model_params)
    # 评估模型
    test_subjects = list(range(2))
    test_datas = sleep.get_processed_data(update_path=dataPath, subjects=test_subjects, num_classes=num_classes)
    test_X, test_y = test_datas[1], test_datas[0]
    pre_y = save_res_pre(test_X, model_path, AttnSleepNet(num_channels, num_classes))
    per = Performance(estimators_list=["Acc"])
    print(per.evaluate(pre_y, test_y))
    f1_scores = f1_score(pre_y, test_y, average=None)
    f1_scores = [round(num, 2) for num in f1_scores]
    print(f'F1 Scores for each class: {f1_scores}')

    # 绘制饼图
    demo_sleep.plotAnalyze(pre_y)
    # 绘制混淆矩阵
    demo_sleep.plot_confusion_matrix(pre_y, test_y)
    # 绘制睡眠趋势图的被试
    pre_subjects = [0]
    pre_data = sleep.get_processed_data(update_path=dataPath, subjects=pre_subjects, num_classes=num_classes)
    model = deep_learning.AttnSleepNet(model_params[0], model_params[1])
    model.initialize()
    model.load_params(f_params=model_path)
    y_pre = model.predict(pre_data[1])
    y_pre_sm = demo_sleep.smooth(y_pre)
    fig, axs = plt.subplots(3, 1, figsize=(15, 10))
    demo_sleep.plotTime(axs[0], pre_data[0], flag_modi=False, color="black", name="PSG true label")
    demo_sleep.plotTime(axs[1], y_pre, flag_modi=False, color="GoldenRod", name="prediction")
    demo_sleep.plotTime(axs[2], y_pre_sm, flag_modi=False, color="GoldenRod", name="smooth prediction")
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
