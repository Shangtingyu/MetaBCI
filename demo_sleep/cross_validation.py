# -*- coding: utf-8 -*-
# Organization: Hangzhou Meidang Digital Health Technology Co.
# Authors: Shangtingyu
# Date: 2024/7/25
# License: MIT License

from collections import Counter
from metabci.brainda.algorithms.deep_learning import np_to_th
from metabci.brainda.algorithms.deep_learning.AttnSleep import AttnSleep
from metabci.brainda.algorithms.utils.model_selection import EnhancedStratifiedKFold
from metabci.brainda.datasets.sleep_telemetry import Sleep_telemetry


def cross_train_model(datas, n_splits=5, model_params=(1, 5), model_selection=EnhancedStratifiedKFold):
    """
    Train the AttnSleep model using cross-validation.
    Parameters:
    datas (list): List of input datas, like [labels, data]
    n_splits (int): Number of folds for cross-validation.
    train_selection: train model selection algorithm.
    model_selection: cross model selection algorithm.
    Returns:
    None
    """
    label, input_data = datas[0], datas[1]
    random_state = 42
    X_train = np_to_th(input_data)
    y_train = np_to_th(label).long()
    kf = model_selection(n_splits=n_splits, shuffle=True, random_state=random_state)
    for fold, (train_index, _, val_index) in enumerate(kf.split(X_train, y_train)):
        model = AttnSleep(model_params[0], model_params[1])
        X_train_fold = X_train[train_index]
        y_train_fold = y_train[train_index]
        X_val_fold = X_train[val_index]
        y_val_fold = y_train[val_index]
        model.fit(X_train_fold, y_train_fold, valid_data=(X_val_fold, y_val_fold))
        print(f"Fold {fold + 1} completed.")


def main():
    npz_path = r'C:\Users\86130\Desktop\ST\3_ch'
    sleep = Sleep_telemetry()
    data = sleep.get_processed_data(update_path=npz_path, subjects=[7])
    cross_train_model(data, model_params=(3, 5))


if __name__ == '__main__':
    main()
