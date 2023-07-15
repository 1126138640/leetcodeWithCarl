import torch
from torch import nn
from torch import functional as F


class ResUnit(nn):
    def __init__(self, in_channels, out_channels, shortcut, e=0.5):
        super(ResUnit, self).__init__()
        inter_channels = out_channels * e
        self.stage1 = nn.ModuleList[
            nn.Conv2d(in_channels=in_channels, out_channels=inter_channels, kernel_size=(1, 1),),
            nn.BatchNorm2d(num_features=in_channels),
            nn.ReLU()
        ]
        self.stage2 = nn.ModuleList[
            nn.Conv2d(in_channels=inter_channels, out_channels=inter_channels, kernel_size=(3, 3)),
            nn.BatchNorm2d(num_features=inter_channels),
            nn.ReLU()
        ]
        self.stage3 = nn.ModuleList[
            nn.Conv2d(in_channels=inter_channels, out_channels=out_channels, kernel_size=(1, 1)),
            nn.BatchNorm2d(num_features=inter_channels),
        ]
        self.shortcut_stage = nn.ModuleList[
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1)),
            nn.BatchNorm2d(num_features=in_channels)
        ]
        self.shortcut = shortcut

    def forward(self, x):
        identity = self.shortcut_stage(x) if self.shortcut else x
        out = self.stage1(x)
        out = self.stage2(out)
        out = self.stage3(out)
        return out + identity
