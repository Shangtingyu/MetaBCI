# -*- coding: utf-8 -*-
"""
1.Function： 读取pytorch保存的混淆矩阵torch
2.Author：xingjian.zhang
3.Time： 20240528
4.Others：这代码运行会有奇怪的问题，提示OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
         解决办法：先运行一下别的带有matplotlib的函数例如compare_same_EEG,这个问题就解决了。
"""
import os
from datetime import datetime
from sklearn.metrics import confusion_matrix
import torch
import numpy as np
import matplotlib.pyplot as plt


def plot_confusion_matrix(label_pre: np.ndarray, label_true: np.ndarray):
    """
    生成混淆矩阵并且可视化

    Parameters:
    label_pre (ndarray): 预测标签
    label_true (ndarray): 真实标签
    """
    all_labels = np.array(label_true).astype(int)
    all_pres = np.array(label_pre).astype(int)
    cm = confusion_matrix(all_labels, all_pres)
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")  # 格式化时间戳
    folder_name = "confusion_matrix"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    cm_file_name = f"confusion_matrix_{timestamp}.torch"
    cm_Save_path = os.path.join(folder_name, cm_file_name)
    torch.save(cm, cm_Save_path)
    print(f"Confusion matrix saved as {cm_file_name} in the folder {folder_name}.")
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['font.weight'] = 'bold'
    matrix = torch.load(cm_Save_path)
    labels = []
    class_totals = matrix.sum(axis=1, keepdims=True)
    num_classes = len(class_totals)
    if num_classes == 5:
        labels = ['Wake', 'N1', 'N2', 'N3', 'REM']
    elif num_classes == 4:
        labels = ['Wake', 'Light Sleep', 'Deep Sleep',  'REM']
    elif num_classes == 3:
        labels = ['Wake', 'NREM', 'REM']
    elif num_classes == 2:
        labels = ['Wake', 'Sleep']
    else:
        print("数据分类任务数存在错误")

    percentage_confusion_matrix = (matrix / class_totals) * 100

    fig, ax = plt.subplots()
    cax = ax.matshow(percentage_confusion_matrix, cmap=plt.cm.Blues)
    plt.colorbar(cax)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_yticklabels(labels, fontsize=12)
    plt.setp(ax.get_xticklabels(), ha="center")
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i == j:
                txt_color = 'white'  # Set color to white for diagonal cells
            else:
                txt_color = 'black'  # Set color to black for non-diagonal cells
            ax.text(j, i, f'{percentage_confusion_matrix[i, j]:.2f}%',
                    ha='center', va='center', color=txt_color, fontsize=12, fontweight='bold')

    # Set labels for the x and y axes
    plt.xlabel('Predicted Label', fontsize=16)
    plt.ylabel('True Label', fontsize=16)

    # Display the plot
    plt.show()
    plt.clf()


