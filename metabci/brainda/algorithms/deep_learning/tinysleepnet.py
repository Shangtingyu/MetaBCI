from collections import OrderedDict
import torch.nn as nn
from metabci.brainda.algorithms.deep_learning.attnsleepnet import SkorchNet_sleep


@SkorchNet_sleep
class TinySleepNet(nn.Module):
    """
    A class to select and instantiate a specific TinySleepNet model based on the number of classes.

    Parameters
        num_classes (int): The number of classes (2,3,4,5)

    Returns:
    nn.Module: An instantiated Sleep model.

    Example:
    data : tensor(epoch, channel, data)
    >>> model = TinySleepNet(5)
    >>> model.fit(data, label)
    Selects the dual-channel model and sets the number of classes to 5.
    """
    def __init__(self, num_classes: int):
        """

        :param num_classes (int): The number of classes (2,3,4,5)
        """
        super().__init__()
        self.padding_edf = {  # same padding in tensorflow
            'conv1': (22, 22),
            'max_pool1': (2, 2),
            'conv2': (3, 4),
            'max_pool2': (0, 1),
        }
        first_filter_size = int(100 / 2.0)  # 100/2 = 50, 与以往使用的Resnet相比，这里的卷积核更大
        first_filter_stride = int(100 / 16.0)  # todo 与论文不同，论文给出的stride是100/4=25
        self.cnn = nn.Sequential(
            nn.ConstantPad1d(self.padding_edf['conv1'], 0),  # conv1
            nn.Sequential(OrderedDict([
                ('conv1', nn.Conv1d(in_channels=1, out_channels=128, kernel_size=first_filter_size, stride=first_filter_stride,
                      bias=False))
            ])),
            nn.BatchNorm1d(num_features=128, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True),
            nn.ConstantPad1d(self.padding_edf['max_pool1'], 0),  # max p 1
            nn.MaxPool1d(kernel_size=8, stride=8),
            nn.Dropout(p=0.5),
            nn.ConstantPad1d(self.padding_edf['conv2'], 0),  # conv2
            nn.Sequential(OrderedDict([
                ('conv2',
                 nn.Conv1d(in_channels=128, out_channels=128, kernel_size=8, stride=1, bias=False))
            ])),
            nn.BatchNorm1d(num_features=128, eps=0.001, momentum=0.01),

            nn.ReLU(inplace=True),
            nn.ConstantPad1d(self.padding_edf['conv2'], 0),  # conv3
            nn.Sequential(OrderedDict([
                ('conv3',nn.Conv1d(in_channels=128, out_channels=128, kernel_size=8, stride=1, bias=False))
            ])),
            nn.BatchNorm1d(num_features=128, eps=0.001, momentum=0.01),

            nn.ReLU(inplace=True),
            nn.ConstantPad1d(self.padding_edf['conv2'], 0),  # conv4
            nn.Sequential(OrderedDict([
                ('conv4', nn.Conv1d(in_channels=128, out_channels=128, kernel_size=8, stride=1, bias=False))
            ])),
            nn.BatchNorm1d(num_features=128, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True),
            nn.ConstantPad1d(self.padding_edf['max_pool2'], 0),  # max p 2
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Flatten(),
            nn.Dropout(p=0.5),
        )

        self.rnn = nn.LSTM(input_size=2048, hidden_size=128, num_layers=1, batch_first=True)
        self.rnn_dropout = nn.Dropout(p=0.5)  # todo 是否需要这个dropout?
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.cnn(x)
        x = self.rnn(x)
        # 解包
        x = x[0]
        x = self.rnn_dropout(x)
        x = self.fc(x)

        return x
