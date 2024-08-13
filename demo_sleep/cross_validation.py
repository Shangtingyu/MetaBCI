# -*- coding: utf-8 -*-
# Organization: Hangzhou Meidang Digital Health Technology Co.
# Authors: Shangtingyu
# Date: 2024/7/25
# License: MIT License
import os

import numpy as np
import demo_sleep
from demo_sleep import save_res_pre
from metabci.brainda.algorithms.deep_learning import np_to_th
from metabci.brainda.algorithms.deep_learning.attnsleepnet import AttnSleepNet
from metabci.brainda.algorithms.utils.model_selection import EnhancedStratifiedKFold
from metabci.brainda.datasets.sleep_telemetry import Sleep_telemetry
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
        model.fit(X_train_fold, y_train_fold, test_data=(X_val_fold, y_val_fold))

        best_acc = getattr(model, 'best_valid_acc', None)
        best_accs.append(best_acc)
        names.append(model.model_name)
        print(f"Fold {fold + 1} completed.")
    # 找到最大数字
    max_number = max(best_accs)
    # 找到最大数字的索引
    max_index = best_accs.index(max_number)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(script_dir, names[max_index], 'params.pt')
    return path


def main():
    npz_path = r'/data/xingjain.zhang/sleep/1_npzdata/ST'  # npz数据存储地址
    sleep = Sleep_telemetry()
    subjects = list(range(30))
    data = sleep.get_processed_data(update_path=npz_path, subjects=subjects, num_classes=2)
    model_params = (1, 2)  # 模型的通道和分类门数
    path = cross_train_model(data, model_params=model_params)
    print(path)
    test_subjects = list(range(30, 35))
    test_datas = sleep.get_processed_data(update_path=npz_path, subjects=test_subjects, num_classes=2)
    test_X, test_y = test_datas[1], test_datas[0]
    pre_y = save_res_pre(test_X, path, AttnSleepNet(1, 2))
    per = Performance(estimators_list=["Acc"])
    print(per.evaluate(pre_y, test_y))


if __name__ == '__main__':
    main()
