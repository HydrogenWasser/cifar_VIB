import math
import random
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.distributions.normal import Normal
import matplotlib.pyplot as plt

class NormalVGG(nn.Module):

    """
    VGG16 VIB version
    """
    def __init__(self, device):
        super(NormalVGG, self).__init__()
        self.num_class = 10
        self.loss_func = nn.CrossEntropyLoss()
        self.device = device
        self.batch_size = 100
        #self.prior = Normal(torch.zeros(1, self.z_dim).to(self.device), torch.ones(1, self.z_dim).to(self.device))
        # conv1
        net = models.vgg16(pretrained=True)
        net.classifier = nn.Sequential()
        self.features = net

        self.classifier = nn.Sequential(  # 定义自己的分类层
            nn.Linear(in_features=512 * 7 * 7, out_features=1024),  # 自定义网络输入后的大小。
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(in_features=1024, out_features=1024),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(in_features=1024, out_features=self.num_class)
        )


    def forward(self, x):   # output: 32 * 32 * 3
        x = self.features(x)
        x = x.view(x.size(0), -1)
        output = self.classifier(x)
        return output
        # x (batchsize, 512)

    def batch_loss(self, output, y_batch):

        loss = self.loss_func(output, y_batch)
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
