from torch import nn
import os
import torch
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision
from torch.utils.data import TensorDataset, Dataset, DataLoader
from causalVIBcifar import *
from NormalVGG import *
import numpy as np
import fgsm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

batch_size = 50
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#  data pre-treatment
data_transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    "val": transforms.Compose([transforms.Resize(256),
                               transforms.CenterCrop(224),
                               transforms.ToTensor(),
                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

# load train data
trainset = torchvision.datasets.CIFAR10('data',
                                        train=True,
                                        download=False,
                                        transform=data_transform["train"])

trainloader = DataLoader(dataset=trainset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=0)

# load test data
testset = torchvision.datasets.CIFAR10('data',
                                       train=False,
                                       download=False,
                                       transform=data_transform["val"])
testloader = DataLoader(dataset=testset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=0)

# load model and load parameter into the model
net = torchvision.models.resnet34(pretrained= True)
net.to(device) # net into cuda

# change fc layer structure
inchannel = net.fc.in_features
net.fc = nn.Linear(inchannel, 10)  # change the number of class

# #对于模型的每个权重，使其不进行反向传播，即固定参数
# for param in net.parameters():
#     param.requires_grad = False
# #但是参数全部固定了，也没法进行学习，所以我们不固定最后一层，即全连接层
# for param in net.fc.parameters():
#     param.requires_grad = True

# define ooptimizer and loss function
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0001)
# training
best_acc = 0.0
save_path = './weights/resNet34.pth'
for epoch in range(1):
    # training
    net.train()
    running_loss = 0.0
    for step, data in enumerate(trainloader, start=0):
        images, labels = data
        optimizer.zero_grad()
        logits = net(images.to(device))
        loss = loss_function(logits, labels.to(device))
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        # print train process
        rate = (step + 1) / len(trainloader)
        a = "*" * int(rate * 50)
        b = "." * int((1 - rate) * 50)
        print("\rtrain loss: {:^3.0f}%[{}->{}]{:.4f}".format(int(rate * 100), a, b, loss), end="")

net.eval()  # change into test model
acc = 0.0  # accumulate accurate number / epoch
with torch.no_grad():
    for val_data in testloader:
        val_images, val_labels = val_data
        outputs = net(val_images.to(device))  # eval model only have last output layer
        # loss = loss_function(outputs, test_labels)
        predict_y = torch.max(outputs, dim=1)[1]
        acc += (predict_y == val_labels.to(device)).sum().item()
    val_accurate = acc / batch_size
    if val_accurate > best_acc:
        best_acc = val_accurate
        torch.save(net.state_dict(), save_path)
    print('[epoch %d] train_loss: %.3f  test_accuracy: %.3f' %
          (epoch + 1, running_loss / step, val_accurate))

