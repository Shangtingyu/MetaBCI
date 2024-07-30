# -*- coding: utf-8 -*-
# Organization: Hangzhou Meidang Digital Health Technology Co.
# Authors: Shangtingyu
# Date: 2024/7/9
# License: MIT License

from metabci.brainda.algorithms.deep_learning import np_to_th
from metabci.brainda.algorithms.deep_learning.eegnet import EEGNet
from metabci.brainda.datasets.sleep_telemetry import Sleep_telemetry

dataPath = r'C:\Users\86130\Desktop\哈哈哈哈'
# Processed npz data address
path = r'C:\Users\86130\Desktop\哈哈哈哈'
sleep = Sleep_telemetry(dataPath)
sleep.save_processed_data(update_path=path)
subjects = list(range(1, 6))
data = sleep.get_processed_data(update_path=path, subjects=subjects)
label, input_data = data[0], data[1]
# check the shape of data
model = EEGNet(1, 3000, 5)
X_train = np_to_th(input_data)
y_train = np_to_th(label)
print(X_train.shape)
# The value of the label in the fit function must be in long format.
y_train = y_train.long()
model.fit(X_train, y_train)
