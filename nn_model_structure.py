"""
Author: Zejun Gong
Date: 12/NOV/2021
"""
import torch.nn as nn

class BinaryClassification(nn.Module):
    def __init__(self):

        super(BinaryClassification, self).__init__()
        self.layer_1 = nn.Linear(768, 768)
        self.layer_2 = nn.Linear(768, 768)
        self.layer_3 = nn.Linear(768, 768)
        self.layer_4 = nn.Linear(768, 1536)
        self.layer_5 = nn.Linear(1536, 1536)
        self.layer_6 = nn.Linear(1536, 1536)
        self.layer_7 = nn.Linear(1536, 1536)
        self.layer_out = nn.Linear(1536, 1)
        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(p=0.2)
        self.batchnorm1 = nn.BatchNorm1d(768)
        self.batchnorm2 = nn.BatchNorm1d(768)
        self.batchnorm3 = nn.BatchNorm1d(768)
        self.batchnorm4 = nn.BatchNorm1d(1536)
        self.batchnorm5 = nn.BatchNorm1d(1536)
        self.batchnorm6 = nn.BatchNorm1d(1536)
        self.batchnorm7 = nn.BatchNorm1d(1536)
    def forward(self, inputs):

        x = self.relu(self.layer_1(inputs))
        x = self.batchnorm1(x)
        x = self.relu(self.layer_2(x))
        x = self.batchnorm2(x)
        x = self.relu(self.layer_3(x))
        x = self.batchnorm3(x)
        x = self.relu(self.layer_4(x))
        x = self.batchnorm4(x)
        x = self.dropout(x)
        x = self.relu(self.layer_5(x))
        x = self.batchnorm5(x)
        x = self.relu(self.layer_6(x))
        x = self.batchnorm6(x)
        x = self.relu(self.layer_7(x))
        x = self.batchnorm7(x)
        x = self.dropout(x)
        x = self.layer_out(x)

        return x
