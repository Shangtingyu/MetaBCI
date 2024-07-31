# -*- coding: utf-8 -*-
# Organization: Hangzhou Meidang Digital Health Technology Co.
# Authors: Shangtingyu
# Date: 2024/7/25
# License: MIT License
import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, classification_report, cohen_kappa_score, accuracy_score

from metabci.brainda.algorithms.deep_learning import np_to_th
from metabci.brainda.algorithms.deep_learning.AttnSleep import AttnSleep
from metabci.brainda.algorithms.utils.model_selection import EnhancedStratifiedKFold
from metabci.brainda.datasets.sleep_cassette import SleepCassette
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

    # Initialize lists to store predictions and true labels for all folds
    all_pres = []
    all_labels = []

    for fold, (train_index, _, val_index) in enumerate(kf.split(X_train, y_train)):
        model = AttnSleep(model_params[0], model_params[1])
        X_train_fold = X_train[train_index]
        y_train_fold = y_train[train_index]
        X_val_fold = X_train[val_index]
        y_val_fold = y_train[val_index]

        # Train the model
        model.fit(X_train_fold, y_train_fold, valid_data=(X_val_fold, y_val_fold))

        # Get predictions on the validation set
        val_preds = model.predict(X_val_fold)
        val_labels = y_val_fold.cpu().numpy()  # True labels

        # Append predictions and labels to the lists
        all_pres.extend(val_preds)
        all_labels.extend(val_labels)

        print(f"Fold {fold + 1} completed.")

    # Compute and save the confusion matrix
    all_labels = np.array(all_labels).astype(int)
    all_pres = np.array(all_pres).astype(int)
    # r = classification_report(all_labels, all_pres, digits=5, output_dict=True)
    cm = confusion_matrix(all_labels, all_pres)
    # df = pd.DataFrame(r)
    # df["cohen"] = cohen_kappa_score(all_labels, all_pres)
    # df["accuracy"] = accuracy_score(all_labels, all_pres)
    # df = df * 100
    # file_name = self.config["name"] + "_classification_report.xlsx"
    # report_Save_path = os.path.join(save_dir, file_name)
    # df.to_excel(report_Save_path)
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")  # 格式化时间戳
    folder_name = "confusion_matrix"

    # 创建文件夹（如果不存在的话）
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # 生成保存路径
    cm_file_name = f"confusion_matrix_{timestamp}.torch"
    cm_Save_path = os.path.join(folder_name, cm_file_name)

    # 保存混淆矩阵
    torch.save(cm, cm_Save_path)
    print(f"Confusion matrix saved as {cm_file_name} in the folder {folder_name}.")


def main():
    npz_path = r'/data/xingjain.zhang/sleep/1_npzdata/SC/01_SC_FPZ-Cz'
    sleep = Sleep_telemetry(npz_path)
    subjects = list(range(30))
    data = sleep.get_processed_data(update_path=npz_path, subjects=subjects,num_classes=2)
    cross_train_model(data, model_params=(1, 2))


if __name__ == '__main__':
    main()
