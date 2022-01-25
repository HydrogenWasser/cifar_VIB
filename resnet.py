import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import torchvision

class resnet(nn.Module):
    def __init__(self, device):
        super(resnet, self).__init__()
        self.device = device
        self.batch_size = 100
        self.loss_func = nn.CrossEntropyLoss(reduce=False)
        self.features = torchvision.models.resnet34(pretrained= True)
        self.Encoder = nn.Sequential(nn.Linear(in_features=512, out_features=1024),
                                               nn.ReLU(),
                                               nn.Linear(in_features=1024, out_features=1024),
                                               nn.ReLU(),
                                               nn.Linear(in_features=1024, out_features=256),
                                               nn.ReLU(),
                                               nn.Linear(in_features=256, out_features=10)
                                     )

        self.features.fc = self.Encoder

    def forward(self, x):
        y_pre = self.features(x)
        return y_pre

    def batch_loss(self, y_pre, y_batch):
        cross_entropy_loss = self.loss_func(y_pre, y_batch)
        loss = torch.mean(cross_entropy_loss, dim=0)
        return loss

class EMA(nn.Module):
    def __init__(self, mu):
        super(EMA, self).__init__()
        self.mu = mu
        self.shadow = {}

    def register(self, name, val):
        self.shadow[name] = val.clone()

    def forward(self, name, x):
        assert name in self.shadow
        new_average = self.mu * x + (1.0 - self.mu) * self.shadow[name]
        self.shadow[name] = new_average.clone()
        return new_average
