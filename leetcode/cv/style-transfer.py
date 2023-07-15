from torchvision import models
from torch import nn

our = models.resnet50(pretrained=True)
for params in our.parameters():
    params.requires_grad = False
our.fc = nn.Linear(in_features=our.fc.in_features, out_features=100)
