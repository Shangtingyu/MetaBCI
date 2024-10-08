# -*- coding: utf-8 -*-
# Organization: Hangzhou Meidang Digital Health Technology Co.
# Authors: xingjian.zhang,Shangtingyu
# Date: 2024/7/5
# License: MIT License
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
from copy import deepcopy
from datetime import datetime

from skorch.dataset import ValidSplit

from metabci.brainda.algorithms.deep_learning.base import SkorchNet, NeuralNetClassifierNoLog
from skorch.callbacks import LRScheduler, EpochScoring, Checkpoint, EarlyStopping, Callback
import torch.optim as optim


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),  # inplace=True表示在计算ReLU时会修改输入的张量，而不是创建一个新的张量。这可以节省内存。
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()  # b是batchsize, c是channel
        y = self.avg_pool(x).view(b, c)  # .view(b, c) 操作将结果的形状调整为 (b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)  # y.expand_as(x) 来将 y 的形状扩展成与 x 相同的形状。


class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=16):
        super(SEBasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(planes, planes, 1)
        self.bn2 = nn.BatchNorm1d(planes)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class GELU(nn.Module):
    # for older versions of PyTorch.  For new versions you can use nn.GELU() instead.
    def __init__(self):
        super(GELU, self).__init__()

    def forward(self, x):
        x = torch.nn.functional.gelu(x)
        return x


class MRCNN(nn.Module):
    def __init__(self, afr_reduced_cnn_size):
        super(MRCNN, self).__init__()
        drate = 0.5
        self.GELU = GELU()
        self.features1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=50, stride=6, bias=False, padding=24),
            nn.BatchNorm1d(64),
            self.GELU,
            nn.MaxPool1d(kernel_size=8, stride=2, padding=4),
            nn.Dropout(drate),
            nn.Conv1d(64, 128, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(128),
            self.GELU,

            nn.Conv1d(128, 128, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(128),
            self.GELU,

            nn.MaxPool1d(kernel_size=4, stride=4, padding=2)
        )

        self.features2 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=400, stride=50, bias=False, padding=200),
            nn.BatchNorm1d(64),
            self.GELU,
            nn.MaxPool1d(kernel_size=4, stride=2, padding=2),
            nn.Dropout(drate),

            nn.Conv1d(64, 128, kernel_size=7, stride=1, bias=False, padding=3),
            nn.BatchNorm1d(128),
            self.GELU,

            nn.Conv1d(128, 128, kernel_size=7, stride=1, bias=False, padding=3),
            nn.BatchNorm1d(128),
            self.GELU,

            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )
        self.dropout = nn.Dropout(drate)
        self.inplanes = 128
        self.AFR = self._make_layer(SEBasicBlock, afr_reduced_cnn_size, 1)

    def _make_layer(self, block, planes, blocks, stride=1):  # makes residual SE block
        # planes: 表示特征通道数的数量，它影响了模块内部的卷积层和规范化层的通道数。
        # blocks: 表示要在层中重复多少个基本块（block）。
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.features1(x)
        x2 = self.features2(x)
        x_concat = torch.cat((x1, x2), dim=2)
        x_concat = self.dropout(x_concat)
        x_concat = self.AFR(x_concat)
        return x_concat


class MRCNN_2CH(nn.Module):
    """在原MRCNN基础上，增加双通道数据，在AFR结构前合并"""

    def __init__(self, afr_reduced_cnn_size):
        super(MRCNN_2CH, self).__init__()
        drate = 0.5
        self.GELU = GELU()
        self.features1_EEG = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=50, stride=6, bias=False, padding=24),
            nn.BatchNorm1d(64),
            self.GELU,
            nn.MaxPool1d(kernel_size=8, stride=2, padding=4),
            nn.Dropout(drate),
            nn.Conv1d(64, 128, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(128),
            self.GELU,
            nn.Conv1d(128, 128, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(128),
            self.GELU,
            nn.MaxPool1d(kernel_size=4, stride=4, padding=2)
        )
        self.features1_EOG = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=50, stride=6, bias=False, padding=24),
            nn.BatchNorm1d(64),
            self.GELU,
            nn.MaxPool1d(kernel_size=8, stride=2, padding=4),
            nn.Dropout(drate),
            nn.Conv1d(64, 128, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(128),
            self.GELU,
            nn.Conv1d(128, 128, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(128),
            self.GELU,
            nn.MaxPool1d(kernel_size=4, stride=4, padding=2)
        )

        self.features2_EEG = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=400, stride=50, bias=False, padding=200),
            nn.BatchNorm1d(64),
            self.GELU,
            nn.MaxPool1d(kernel_size=4, stride=2, padding=2),
            nn.Dropout(drate),
            nn.Conv1d(64, 128, kernel_size=7, stride=1, bias=False, padding=3),
            nn.BatchNorm1d(128),
            self.GELU,
            nn.Conv1d(128, 128, kernel_size=7, stride=1, bias=False, padding=3),
            nn.BatchNorm1d(128),
            self.GELU,
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )
        self.features2_EOG = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=400, stride=50, bias=False, padding=200),
            nn.BatchNorm1d(64),
            self.GELU,
            nn.MaxPool1d(kernel_size=4, stride=2, padding=2),
            nn.Dropout(drate),
            nn.Conv1d(64, 128, kernel_size=7, stride=1, bias=False, padding=3),
            nn.BatchNorm1d(128),
            self.GELU,
            nn.Conv1d(128, 128, kernel_size=7, stride=1, bias=False, padding=3),
            nn.BatchNorm1d(128),
            self.GELU,
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        self.dropout = nn.Dropout(drate)
        self.inplanes = 128
        self.AFR = self._make_layer(SEBasicBlock, afr_reduced_cnn_size, 1)

    def _make_layer(self, block, planes, blocks, stride=1):  # makes residual SE block
        # planes: 表示特征通道数的数量，它影响了模块内部的卷积层和规范化层的通道数。
        # blocks: 表示要在层中重复多少个基本块（block）。
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x_EEG = torch.unsqueeze(x[:, 0, :], 1)
        x_EOG = torch.unsqueeze(x[:, 1, :], 1)

        x1_EEG = self.features1_EEG(x_EEG)
        x2_EEG = self.features2_EEG(x_EEG)
        x_concat_EEG = torch.cat((x1_EEG, x2_EEG), dim=2)

        x1_EOG = self.features1_EOG(x_EOG)
        x2_EOG = self.features2_EOG(x_EOG)
        x_concat_EOG = torch.cat((x1_EOG, x2_EOG), dim=2)

        x_concat = torch.cat((x_concat_EEG, x_concat_EOG), dim=2)

        x_concat = self.dropout(x_concat)
        x_concat = self.AFR(x_concat)
        return x_concat


def attention(query, key, value, dropout=None):
    "Implementation of Scaled dot product attention"
    d_k = query.size(-1)  # 这一行代码获取了查询（query）张量的最后一个维度的大小，通常用于表示查询和键（key）的特征维度
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  # 这一行代码计算了查询（query）和键（key）之间的点积matmul得分

    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class CausalConv1d(torch.nn.Conv1d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=True):
        self.__padding = (kernel_size - 1) * dilation

        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.__padding,
            dilation=dilation,
            groups=groups,
            bias=bias)

    def forward(self, input):
        result = super(CausalConv1d, self).forward(input)
        if self.__padding != 0:
            return result[:, :, :-self.__padding]
        return result


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, afr_reduced_cnn_size, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        # clones 的函数来创建多个相同的卷积层
        # CausalConv1d 因果卷积
        self.convs = clones(CausalConv1d(afr_reduced_cnn_size, afr_reduced_cnn_size, kernel_size=7, stride=1), 3)
        self.linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value):
        "Implements Multi-head attention"
        nbatches = query.size(0)

        # Q、K、V
        query = query.view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        key = self.convs[1](key).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        value = self.convs[2](value).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        x, self.attn = attention(query, key, value, dropout=self.dropout)

        # contiguous() 是一个PyTorch张量（Tensor）的方法，用于返回一个具有连续内存布局的新张量。
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)

        return self.linear(x)


class LayerNorm(nn.Module):
    "Construct a layer normalization module."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        # 创建两个可学习的模型参数 a_2 和 b_2，
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps  # eps代表为了避免出现零作为分母，添加的小正数值。

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)  # 计算了输入张量 x 沿着最后一个维度（通常是特征维度）的均值。keepdim=True 参数确保结果具有与输入相同的维度。
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerOutput(nn.Module):
    '''
    A residual connection followed by a layer norm.
    '''

    def __init__(self, size, dropout):
        super(SublayerOutput, self).__init__()
        self.norm = LayerNorm(size)  # 层归一化是在特征维度上进行的
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class TCE(nn.Module):
    '''
    Transformer Encoder

    It is a stack of N layers.
    '''

    def __init__(self, layer, N):
        super(TCE, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class EncoderLayer(nn.Module):
    '''
    An encoder layer

    Made up of self-attention and a feed forward layer.
    Each of these sublayers have residual and layer norm, implemented by SublayerOutput.
    '''

    def __init__(self, size, self_attn, feed_forward, afr_reduced_cnn_size, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer_output = clones(SublayerOutput(size, dropout), 2)
        self.size = size
        self.conv = CausalConv1d(afr_reduced_cnn_size, afr_reduced_cnn_size, kernel_size=7, stride=1, dilation=1)

    def forward(self, x_in):
        "Transformer Encoder"
        query = self.conv(x_in)
        x = self.sublayer_output[0](query, lambda x: self.self_attn(query, x_in, x_in))  # Encoder self-attention
        return self.sublayer_output[1](x, self.feed_forward)


class PositionwiseFeedForward(nn.Module):
    "Positionwise feed-forward network."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        "Implements FFN equation."
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class BestEpochCallback(Callback):
    def __init__(self):
        self.best_epoch = None
        self.best_valid_acc = None

    def on_epoch_end(self, net, dataset_train, dataset_valid, **kwargs):
        valid_acc = net.history[-1]['valid_acc']
        epoch = net.history[-1]['epoch']

        if self.best_valid_acc is None or valid_acc > self.best_valid_acc:
            self.best_valid_acc = valid_acc
            self.best_epoch = epoch

    def on_train_end(self, net, **kwargs):
        if self.best_epoch is not None:
            print(f"Best validation accuracy achieved at epoch {self.best_epoch}: {self.best_valid_acc}")
            net.best_epoch = self.best_epoch
            net.best_valid_acc = self.best_valid_acc



class SkorchNet_sleep:
    def __init__(self, module):
        self.module = module


    def __call__(self, *args, **kwargs):
        model = self.module(*args, **kwargs)
        self.net = NeuralNetClassifierNoLog(
            module=model.float(),
            criterion=nn.CrossEntropyLoss(),
            optimizer=optim.AdamW,
            optimizer__weight_decay=1e-4,
            batch_size=128,
            lr=1e-3,
            max_epochs=300,
            device="cuda" if torch.cuda.is_available() else "cpu",
            iterator_train__shuffle=True,
            callbacks=[
                ("train_acc", EpochScoring("accuracy", name="train_acc", on_train=True, lower_is_better=False)),
                ("lr_scheduler", LRScheduler(policy="MultiStepLR", milestones=[20], gamma=0.2)),
                ("estoper", EarlyStopping(monitor="valid_acc", patience=20, lower_is_better=False)),
                ("best_epoch", BestEpochCallback()),
            ],
            verbose=True,
        )
        return self.net



@SkorchNet_sleep
class AttnSleep_1CH(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        N = 2  # TCE模型的克隆数量
        d_model = 80  # 用于表示模型中全连接层的神经元个数
        d_ff = 120  # 前馈神经网络的维度，通常用于处理注意力层的输出
        h = 5  # 多头注意力机制中的注意力头的数量, 这里代表划分为子空间的个数
        dropout = 0.3  # 30%的丢弃率
        afr_reduced_cnn_size = 30  # SE block中的通道数

        self.mrcnn = MRCNN(afr_reduced_cnn_size)
        attn = MultiHeadedAttention(h, d_model, afr_reduced_cnn_size)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.tce = TCE(EncoderLayer(d_model, deepcopy(attn), deepcopy(ff), afr_reduced_cnn_size, dropout), N)

        self.fc = nn.Linear(d_model * afr_reduced_cnn_size, num_classes)

    def forward(self, x):
        x_feat = self.mrcnn(x)
        # print("MRCNN输出的tensor维度{}".format(x_feat.shape))
        encoded_features = self.tce(x_feat)
        encoded_features = encoded_features.contiguous().view(encoded_features.shape[0], -1)
        final_output = self.fc(encoded_features)
        return final_output


@SkorchNet_sleep
class AttnSleep_2CH(nn.Module):
    """双通道输入,分别通过MRCNN+AFR+TCE后合并，再输入分类模块"""

    def __init__(self, num_classes):
        super().__init__()

        N = 2  # TCE模型的克隆数量
        d_model = 80  # 用于表示TCE模型中全连接层的神经元个数
        d_ff = 120  # 前馈神经网络的维度，通常用于处理注意力层的输出
        h = 5  # 多头注意力机制中的注意力头的数量, 这里代表划分为子空间的个数
        dropout = 0.3  # TCE模型中的丢弃率30%，MRCNN是50%
        afr_reduced_cnn_size = 30  # SE block中的通道数

        self.mrcnn_EEG = MRCNN(afr_reduced_cnn_size)
        self.mrcnn_EOG = MRCNN(afr_reduced_cnn_size)
        attn_EEG = MultiHeadedAttention(h, d_model, afr_reduced_cnn_size)
        attn_EOG = MultiHeadedAttention(h, d_model, afr_reduced_cnn_size)
        ff_EEG = PositionwiseFeedForward(d_model, d_ff, dropout)
        ff_EOG = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.tce_EEG = TCE(EncoderLayer(d_model, deepcopy(attn_EEG), deepcopy(ff_EEG), afr_reduced_cnn_size, dropout),
                           N)
        self.tce_EOG = TCE(EncoderLayer(d_model, deepcopy(attn_EOG), deepcopy(ff_EOG), afr_reduced_cnn_size, dropout),
                           N)

        self.fc1 = nn.Linear(d_model * afr_reduced_cnn_size * 2, d_model)
        self.fc2 = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x_EEG = torch.unsqueeze(x[:, 0, :], 1)
        x_EOG = torch.unsqueeze(x[:, 1, :], 1)

        x_feat_EEG = self.mrcnn_EEG(x_EEG)
        x_feat_EOG = self.mrcnn_EOG(x_EOG)

        encoded_features_EEG = self.tce_EEG(x_feat_EEG)
        encoded_features_EEG = encoded_features_EEG.contiguous().view(encoded_features_EEG.shape[0], -1)

        encoded_features_EOG = self.tce_EOG(x_feat_EOG)
        encoded_features_EOG = encoded_features_EOG.contiguous().view(encoded_features_EOG.shape[0], -1)

        encoded_features = torch.cat((encoded_features_EEG, encoded_features_EOG), dim=1)
        fc_output = self.fc1(encoded_features)
        final_output = self.fc2(fc_output)
        return final_output


@SkorchNet_sleep
class AttnSleep_3CH(nn.Module):
    """
    三通道输入,分别通过MRCNN+AFR+TCE后合并，再输入分类模块
    在AttnSleep_3CH_S2基础上，增加一层全连接层，减少三个模态的数据融合
    """

    def __init__(self, num_classes):
        super().__init__()

        N = 2  # TCE模型的克隆数量
        d_model = 80  # 用于表示TCE模型中全连接层的神经元个数
        d_ff = 120  # 前馈神经网络的维度，通常用于处理注意力层的输出
        h = 5  # 多头注意力机制中的注意力头的数量, 这里代表划分为子空间的个数
        dropout = 0.3  # TCE模型中的丢弃率10%，MRCNN是50%
        afr_reduced_cnn_size = 30  # SE block中的通道数

        self.mrcnn_EEG = MRCNN(afr_reduced_cnn_size)
        self.mrcnn_EOG = MRCNN(afr_reduced_cnn_size)
        self.mrcnn_EMG = MRCNN(afr_reduced_cnn_size)
        attn_EEG = MultiHeadedAttention(h, d_model, afr_reduced_cnn_size)
        ff_EEG = PositionwiseFeedForward(d_model, d_ff, dropout)
        attn_EOG = MultiHeadedAttention(h, d_model, afr_reduced_cnn_size)
        ff_EOG = PositionwiseFeedForward(d_model, d_ff, dropout)
        attn_EMG = MultiHeadedAttention(h, d_model, afr_reduced_cnn_size)
        ff_EMG = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.tce_EEG = TCE(EncoderLayer(d_model, deepcopy(attn_EEG), deepcopy(ff_EEG), afr_reduced_cnn_size, dropout),
                           N)
        self.tce_EOG = TCE(EncoderLayer(d_model, deepcopy(attn_EOG), deepcopy(ff_EOG), afr_reduced_cnn_size, dropout),
                           N)
        self.tce_EMG = TCE(EncoderLayer(d_model, deepcopy(attn_EMG), deepcopy(ff_EMG), afr_reduced_cnn_size, dropout),
                           N)

        self.fc1 = nn.Linear(d_model * afr_reduced_cnn_size * 3, d_model)
        self.fc2 = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x_EEG = torch.unsqueeze(x[:, 0, :], 1)
        x_EOG = torch.unsqueeze(x[:, 1, :], 1)
        x_EMG = torch.unsqueeze(x[:, 2, :], 1)

        x_feat_EEG = self.mrcnn_EEG(x_EEG)
        x_feat_EOG = self.mrcnn_EOG(x_EOG)
        x_feat_EMG = self.mrcnn_EMG(x_EMG)

        encoded_features_EEG = self.tce_EEG(x_feat_EEG)
        encoded_features_EEG = encoded_features_EEG.contiguous().view(encoded_features_EEG.shape[0], -1)

        encoded_features_EOG = self.tce_EOG(x_feat_EOG)
        encoded_features_EOG = encoded_features_EOG.contiguous().view(encoded_features_EOG.shape[0], -1)

        encoded_features_EMG = self.tce_EMG(x_feat_EMG)
        encoded_features_EMG = encoded_features_EMG.contiguous().view(encoded_features_EMG.shape[0], -1)

        encoded_features = torch.cat((encoded_features_EEG, encoded_features_EOG, encoded_features_EMG), dim=1)

        fc_output = self.fc1(encoded_features)
        final_output = self.fc2(fc_output)

        return final_output


class AttnSleepNet:
    """
    A class to select and instantiate a specific AttnSleep model based on the number of channels and classes.

    Args:
    ch (int): The number of channels (1, 2, or 3).
    num_classes (int): The number of classes (2,3,4,5).

    Returns:
    nn.Module: An instantiated AttnSleep model.

    Example:
    data : tensor(epoch, channel, data)
    >>> model = AttnSleepNet(3, 5)
    >>> model.fit(data, label)
    Selects the dual-channel model and sets the number of classes to 5.
    """

    def __new__(cls, ch, num_classes):
        if ch == 1:
            instance = AttnSleep_1CH(num_classes)
        elif ch == 2:
            instance = AttnSleep_2CH(num_classes)
        elif ch == 3:
            instance = AttnSleep_3CH(num_classes)
        else:
            raise ValueError("Invalid selection")

        # Ensure the attributes are set correctly
        instance.original_class = cls
        instance.ch = ch
        instance.num_classes = num_classes

        return instance
