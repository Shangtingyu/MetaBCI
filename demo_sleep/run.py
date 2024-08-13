# -*- coding: utf-8 -*-
# Organization: Hangzhou Meidang Digital Health Technology Co.
# Authors: Shangtingyu
# Date: 2024/7/9
# License: MIT License


import os
from matplotlib import pyplot as plt
import demo_sleep
from metabci.brainda.algorithms import deep_learning
from metabci.brainda import datasets
from metabci.brainda.algorithms.utils.model_selection import EnhancedStratifiedKFold


def main():
    path = r'D:\sleep-data\ST'  # 原始数据raw_data存储地址，没有则会自动下载
    dataPath = r'D:\sleep-data\ST\EEG Fpz-Cz'  # 数据预处理后的npz_data存储地址
    os.makedirs(dataPath, exist_ok=True)
    subjects = [0,1]     # None则代表处理所有被试
    pre_subjects = [0]  # 绘制睡眠趋势图的被试
    select_ch = ["EEG Fpz-Cz"]  # None则代表使用默认通道
    num_classes = 5  # 睡眠分期的分类任务，支持2-5类

    # 数据预处理与加载
    # 支持数据集:Sleep_telemetry可以替换为Sleep_cassette、SHHS、Mros、MSP、Apples
    sleep = datasets.Sleep_telemetry(dataPath=path)
    sleep.save_processed_data(update_path=dataPath, subjects=subjects, select_ch=select_ch)
    data = sleep.get_processed_data(update_path=dataPath, subjects=subjects, num_classes=num_classes)
    X = deep_learning.np_to_th(data[1])
    y = deep_learning.np_to_th(data[0]).long()
    kf = EnhancedStratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_index, _, test_index = next(kf.split(X, y))
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]

    # AI模型加载与训练
    # model = deep_learning.DeepSleepNet(num_classes=num_classes)
    # model = deep_learning.TinySleepNet(num_classes=num_classes)
    # model = deep_learning.eegnet(len(select_ch),3000,num_classes=num_classes)
    model = deep_learning.AttnSleepNet(len(select_ch), num_classes=num_classes)
    model.fit(X_train, y_train, test_data=(X_test, y_test))

    # 模型预测与结果展示
    y_predict = model.predict(X_test)
    demo_sleep.plot_confusion_matrix(y_predict, y_test.numpy())
    demo_sleep.plotAnalyze(y_predict)
    pre_data = sleep.get_processed_data(update_path=dataPath, subjects=pre_subjects, num_classes=num_classes)
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
