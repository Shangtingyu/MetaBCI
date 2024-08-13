# -*- coding: utf-8 -*-
# Organization: Hangzhou Meidang Digital Health Technology Co.
# Authors: Shangtingyu
# Date: 2024/7/25
# License: MIT License

import numpy as np
import demo_sleep
from metabci.brainda.algorithms.deep_learning import np_to_th
from metabci.brainda.algorithms.deep_learning.attnsleepnet import AttnSleepNet
from metabci.brainda.algorithms.utils.model_selection import EnhancedStratifiedKFold
from metabci.brainda.datasets.sleep_telemetry import Sleep_telemetry


def cross_train_model(datas, n_splits=5, model_params=(1, 5), model_selection=EnhancedStratifiedKFold):
    """
    Train the AttnSleep model using cross-validation.
    Parameters:
    datas (list): List of input datas, like [labels, data]
    n_splits (int): Number of folds for cross-validation.
    model_selection: Cross-validation model selection algorithm.
    Returns:
    None
    """
    label, input_data = datas[0], datas[1]
    random_state = 42
    X_train = np_to_th(input_data)
    y_train = np_to_th(label).long()
    kf = model_selection(n_splits=n_splits, shuffle=True, random_state=random_state)
    all_pres = []
    all_labels = []

    for fold, (train_index, _, val_index) in enumerate(kf.split(X_train, y_train)):
        model = AttnSleepNet(model_params[0], model_params[1])
        X_train_fold = X_train[train_index]
        y_train_fold = y_train[train_index]
        X_val_fold = X_train[val_index]
        y_val_fold = y_train[val_index]
        model.fit(X_train_fold, y_train_fold, test_data=(X_val_fold, y_val_fold))
        val_preds = model.predict(X_val_fold)
        val_labels = y_val_fold.cpu().numpy()
        all_pres.extend(val_preds)
        all_labels.extend(val_labels)
        print(f"Fold {fold + 1} completed.")
    all_pres = np.asarray(all_pres)
    all_labels = np.asarray(all_labels)
    demo_sleep.plot_confusion_matrix(all_pres, all_labels)


def main():
    npz_path = r'D:\sleep-data\ST\EEG Fpz-Cz'  # npz数据存储地址
    sleep = Sleep_telemetry()
    subjects = list(range(30))
    data = sleep.get_processed_data(update_path=npz_path, subjects=subjects,num_classes=2)
    model_params = (1, 2)  # 模型的通道和分类门数
    cross_train_model(data, model_params=model_params)


if __name__ == '__main__':
    main()
