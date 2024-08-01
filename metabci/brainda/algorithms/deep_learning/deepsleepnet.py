import torch
import torch.nn as nn
from metabci.brainda.algorithms.deep_learning.AttnSleep import SkorchNet_sleep


@SkorchNet_sleep
class DeepSleepNet(nn.Module):
    """
    A class to select and instantiate a specific DeepSleepNet model based on the number of classes.

    Parameters
        num_classes (int): The number of classes (2,3,4,5)

    Returns:
    nn.Module: An instantiated Sleep model.

    Example:
    data : tensor(epoch, channel, data)
    >>> model = DeepSleepNet(5)
    >>> model.fit(data, label)
    Selects the dual-channel model and sets the number of classes to 5.
    """
    def __init__(self, num_classes: int):
        """
        :param num_classes (int): The number of classes (2,3,4,5)
        """
        super().__init__()

        Fs = 100

        self.cnn1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=int(Fs / 2), stride=int(Fs / 16), bias=False),
            nn.BatchNorm1d(num_features=64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=8, stride=8),
            nn.Dropout(p=0.5),

            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=8, stride=1, padding=(8 - 1) // 2, bias=False),
            nn.BatchNorm1d(num_features=128),
            nn.ReLU(),

            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=8, stride=1, padding=(8 - 1) // 2, bias=False),
            nn.BatchNorm1d(num_features=128),
            nn.ReLU(),

            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=8, stride=1, padding=(8 - 1) // 2, bias=False),
            nn.BatchNorm1d(num_features=128),
            nn.ReLU(),

            nn.MaxPool1d(kernel_size=4, stride=4)
        )

        self.cnn2 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=int(Fs * 4), stride=int(Fs / 2), bias=False),
            nn.BatchNorm1d(num_features=64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Dropout(0.5),

            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=6, stride=1, padding=(6 - 1) // 2, bias=False),
            nn.BatchNorm1d(num_features=128),
            nn.ReLU(),

            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=6, stride=1, padding=(6 - 1) // 2, bias=False),
            nn.BatchNorm1d(num_features=128),
            nn.ReLU(),

            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=6, stride=1, padding=(6 - 1) // 2, bias=False),
            nn.BatchNorm1d(num_features=128),
            nn.ReLU(),

            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.dropout = nn.Dropout(0.5)
        self.lstm1 = nn.LSTM(512, 512, bidirectional=True)
        self.lstm2 = nn.LSTM(1024, 512, bidirectional=True)
        self.fc1 = nn.Linear(2432, 1024)
        self.fc2 = nn.Linear(2432, 512)
        self.final_layer = nn.Linear(1024, num_classes)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def forward(self, x):
        x_repeat = x.clone()
        x = self.cnn1(x)
        x_repeat = self.cnn2(x_repeat)
        x = torch.cat((x, x_repeat), dim=2)
        x = x.view(-1, self.num_flat_features(x))
        x = self.dropout(x)
        x_orin = x.clone()
        x_orin = self.fc1(x_orin)
        x = self.fc2(x)

        x, _ = self.lstm1(x)
        x = self.dropout(x)
        x, _ = self.lstm2(x)
        x = self.dropout(x)

        x = torch.add(x, x_orin)
        x = self.dropout(x)
        x = self.final_layer(x)

        return x



