from torch import nn
import os
import torch
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision
from torch.utils.data import TensorDataset, Dataset, DataLoader
from causalVIBcifar import *
from resnet_IB import *
from resnet import *
import numpy as np
import fgsm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

"                                                     "
"               Daten Vorbereitung                    "
"                                                     "
batch_size = 100

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
trainset = torchvision.datasets.CIFAR10('data', train=True, download=False,
                                        transform=data_transform["train"])

train_loader = DataLoader(dataset=trainset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=0)

# load test data
testset = torchvision.datasets.CIFAR10('data',
                                       train=False,
                                       download=False,
                                       transform=data_transform["val"])
test_loader = DataLoader(dataset=testset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=0)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

"                                                     "
"               Modell Speichung                      "
"                                                     "

def save(net, name):
    path = './model'
    if not os.path.exists(path):
        os.mkdir(path)
    net_path = path + '/' + name +'.pkl'
    net = net.cpu()
    torch.save(net.state_dict(), net_path)
    net.to(device)

def load(net, name):
    net_path = './model/' + name +'.pkl'
    net.load_state_dict(torch.load(net_path))
    net.to(device)
    return net

"                                                     "
"               Modell Trainierung                    "
"                                                     "
def resnetIB_Train(model, ema, num_epoch):

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)

    for name, param in model.named_parameters():
        if param.requires_grad:
            ema.register(name, param.data)

    for epoch in range(num_epoch):
        loss_bei_epoch = []
        accuracy_bei_epoch = []
        I_X_T_bei_epoch = []
        I_Y_T_bei_epoch = []

        if epoch % 2 == 0 and epoch > 0:
            scheduler.step()

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            y_pre = model(x_batch)
            y_prediction = torch.max(y_pre, dim=1)[1]
            accuracy = torch.mean((y_prediction == y_batch).float())
            accuracy_bei_epoch.append(accuracy.item())
            # print(accuracy)
            loss, I_X_T, I_Y_T = model.batch_loss(x_batch, y_batch, num_samples=12)
            loss_bei_epoch.append(loss.item())
            I_X_T_bei_epoch.append(I_X_T.item())
            I_Y_T_bei_epoch.append(I_Y_T.item())

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            for name, param in model.named_parameters():
                if (param.requires_grad):
                    ema(name, param.data)

        # if(epoch%5 == 0):
        print("EPOCH: ", epoch, ", loss: ", np.mean(loss_bei_epoch), ", Accuracy: ", np.mean(accuracy_bei_epoch), ", I_X_T: ",  np.mean(I_X_T_bei_epoch), ", I_Y_T: ", np.mean(I_Y_T_bei_epoch))
    save(model, "ResnetIB")

def resnetIB_eval(model):
    model = load(model, "ResnetIB")
    model.eval()
    accuracy_ = []
    I_X_T_ = []
    I_Y_T_ = []

    for x_batch, y_batch in test_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        y_pre = model(x_batch)
        y_prediction = torch.max(y_pre, dim=1)[1]
        accuracy = torch.mean((y_prediction == y_batch).float())
        accuracy_.append(accuracy.item())


    # if(epoch%5 == 0):
    print("TEST, Accuracy: ", np.mean(accuracy_))

def resnet_Train(model, ema, num_epoch):

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)

    for name, param in model.named_parameters():
        if param.requires_grad:
            ema.register(name, param.data)

    for epoch in range(num_epoch):
        loss_bei_epoch = []
        accuracy_bei_epoch = []

        if epoch % 2 == 0 and epoch > 0:
            scheduler.step()

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            y_pre = model(x_batch)
            y_prediction = torch.max(y_pre, dim=1)[1]
            accuracy = torch.mean((y_prediction == y_batch).float())
            accuracy_bei_epoch.append(accuracy.item())

            loss = model.batch_loss(y_pre, y_batch)
            loss_bei_epoch.append(loss.item())

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            for name, param in model.named_parameters():
                if (param.requires_grad):
                    ema(name, param.data)

        # if(epoch%5 == 0):
        print("EPOCH: ", epoch, ", loss: ", np.mean(loss_bei_epoch), ", Accuracy: ", np.mean(accuracy_bei_epoch))
    save(model, "normalResnet")
#
def NormalVGG_eval(model):
    model = load(model, "normalResnet")
    model.eval()
    accuracy_ = []

    for x_batch, y_batch in test_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        y_pre= model(x_batch)

        y_prediction = torch.max(y_pre, dim=1)[1]
        accuracy = torch.mean((y_prediction == y_batch).float())
        accuracy_.append(accuracy.item())


    # if(epoch%5 == 0):
    print("TEST, Accuracy: ", np.mean(accuracy_))

def causalVGG_adver(beta, model, epsilon):
    model = load(model, str(beta)+"causalVGG")
    model.eval()
    adver_image_obtain = fgsm.attack_model(model=model)
    adver_image_obtain.is_causal = True
    accuracy_clean = []
    accuracy_adver = []
    for x_batch, y_batch in test_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        y_pre = model(x_batch)
        y_prediction = torch.max(y_pre, dim=1)[1]
        accuracy = torch.mean((y_prediction == y_batch).float())
        accuracy_clean.append(accuracy.item())

        perturbed_x_batch = adver_image_obtain.generate(x_batch, eps=epsilon, y=y_batch)

        y_pre = model(perturbed_x_batch)
        y_prediction = torch.max(y_pre, dim=1)[1]
        accuracy = torch.mean((y_prediction == y_batch).float())
        accuracy_adver.append(accuracy.item())

    # if(epoch%5 == 0):
    print("TEST, epsilon: ", epsilon, " Clean Accuracy: ", np.mean(accuracy_clean), ", Adversial Accuracy: ",  np.mean(accuracy_adver))

def normalVGG_adver(model, epsilon):
    model = load(model, "NormalVGG")
    model.eval()
    adver_image_obtain = fgsm.attack_model(model=model)
    adver_image_obtain.is_causal = False
    accuracy_clean = []
    accuracy_adver = []
    for x_batch, y_batch in test_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        y_pre = model(x_batch)
        y_prediction = torch.max(y_pre, dim=1)[1]
        accuracy = torch.mean((y_prediction == y_batch).float())
        accuracy_clean.append(accuracy.item())

        perturbed_x_batch = adver_image_obtain.generate(x_batch, eps=epsilon, y=y_batch)

        y_pre = model(perturbed_x_batch)
        y_prediction = torch.max(y_pre, dim=1)[1]
        accuracy = torch.mean((y_prediction == y_batch).float())
        accuracy_adver.append(accuracy.item())

    # if(epoch%5 == 0):
    print("TEST, Clean Accuracy: ", np.mean(accuracy_clean), ", Adversial Accuracy: ",  np.mean(accuracy_adver))
