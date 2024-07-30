# -*- coding: utf-8 -*-
# Organization: Hangzhou Meidang Digital Health Technology Co.
# Authors: Shangtingyu
# Date: 2024/7/9
# License: MIT License

from metabci.brainda.algorithms.deep_learning import np_to_th
from metabci.brainda.algorithms.deep_learning.AttnSleep import AttnSleep
from metabci.brainda.datasets.sleep_telemetry import Sleep_telemetry


dataPath = r'C:\Users\86130\Desktop\哈哈哈哈'
path = r'/data/xingjain.zhang/sleep/1_npzdata/SC/01_SC_FPZ-Cz'
sleep = Sleep_telemetry(dataPath)
subjects = [3]
data = sleep.get_processed_data(update_path=path, subjects=subjects)
label, input_data = data[0], data[1]
model = AttnSleep(1, 5)
X_train = np_to_th(input_data)
y_train = np_to_th(label)
y_train = y_train.long()
model.fit(X_train, y_train)
