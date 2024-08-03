# -*- coding: utf-8 -*-
"""
1.Function： 读取pytorch保存的混淆矩阵torch
2.Author：xingjian.zhang
3.Time： 20240528
4.Others：这代码运行会有奇怪的问题，提示OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
         解决办法：先运行一下别的带有matplotlib的函数例如compare_same_EEG,这个问题就解决了。
"""

import torch
import numpy as np
import matplotlib.pyplot as plt


def plot_confusion_matrix(readPath:str):
    """
    读取并可视化混淆矩阵

    Parameters:
    readPath (str): 混淆矩阵文件的路径
    """
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['font.weight'] = 'bold'
    confusion_matrix = torch.load(readPath)

    labels = ['Wake', 'N1', 'N2', 'N3', 'REM']

    class_totals = confusion_matrix.sum(axis=1, keepdims=True)
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

    percentage_confusion_matrix = (confusion_matrix / class_totals) * 100

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


# Example usage
readPath = r"D:\python\Pycharm\metabci_0804\demo_sleep\01_SC_FPZ-Cz_confusion_matrix.torch"
plot_confusion_matrix(readPath)
