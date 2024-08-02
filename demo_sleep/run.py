# -*- coding: utf-8 -*-
# Organization: Hangzhou Meidang Digital Health Technology Co.
# Authors: Shangtingyu
# Date: 2024/7/9
# License: MIT License
import os
from matplotlib import pyplot as plt
from demo_sleep.show import plotAnalyze, plotTime
from demo_sleep.smooth import smooth
from metabci.brainda.algorithms.deep_learning import np_to_th, EEGNet
from metabci.brainda.algorithms.deep_learning.AttnSleep import AttnSleep
from metabci.brainda.algorithms.deep_learning.deepsleepnet import DeepSleepNet
from metabci.brainda.datasets.sleep_cassette import Sleep_cassette
from metabci.brainda.datasets.msp import MSP
from metabci.brainda.datasets.shhs import SHHS
from metabci.brainda.datasets.sleep_telemetry import Sleep_telemetry
from metabci.brainda.utils.performance import Performance


path = r'/data/xingjain.zhang/sleep/0_rawdata/sleep-ST'  # 原始数据raw_data存储地址，没有则会自动下载
dataPath = r'/data/xingjain.zhang/sleep/1_npzdata/msp/C3_M2'  # 数据预处理后的npz_data存储地址
os.makedirs(dataPath, exist_ok=True)
train_subjects = None  # None则代表处理所有被试
test_subjects = [11]
select_ch = ["EEG Fpz-Cz"]  # None则代表使用单通道"EEG Fpz-Cz"
num_classes = 5  # 睡眠分期的分类任务，支持2-5类
sleep = MSP(dataPath=path)
# sleep.save_processed_data(update_path=dataPath, subjects=train_subjects, select_ch=select_ch)
print("Data preprocessing is complete.")
train_data = sleep.get_processed_data(update_path=dataPath, subjects=train_subjects, num_classes=num_classes)
test_data = sleep.get_processed_data(update_path=dataPath, subjects=test_subjects, num_classes=num_classes)
X_test = np_to_th(test_data[1])
y_test = np_to_th(test_data[0])

X_train = np_to_th(train_data[1])
y_train = np_to_th(train_data[0])
y_test = y_test.long()
y_train = y_train.long()

# model = AttnSleep(3, num_classes=num_classes)
model = DeepSleepNet(num_classes)
model.fit(X_train, y_train)
y_predict = model.predict(X_test)
y_predict_sm = smooth(y_predict)
performance = Performance(estimators_list=["Acc", "TPR", "FNR", "TNR"], isdraw=True)
print(performance.evaluate(test_data[0], y_predict))
plotAnalyze(y_predict_sm)
fig, axs = plt.subplots(3, 1, figsize=(15, 10))
plotTime(axs[0], test_data[0], flag_modi=False, color="black", name="PSG true label")
plotTime(axs[1], y_predict, flag_modi=False, color="GoldenRod", name="prediction")
plotTime(axs[2], y_predict_sm, flag_modi=False, color="GoldenRod", name="smooth prediction")
plt.tight_layout()
plt.show()


