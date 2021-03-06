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

"                                                     "
"               Daten Vorbereitung                    "
"                                                     "

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
batch_size = 100
samples_amount = 12

train_data = datasets.CIFAR10('data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=batch_size)
test_data = datasets.CIFAR10('data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_data, batch_size=batch_size)

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
def causalVGG_Train(beta, model, ema, num_epoch):

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)
    model.give_beta(beta)

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

            y_pre, z_scores, features, logits, mean_Cs, std_Cs, y_logits_s = model(x_batch)
            y_prediction = torch.max(y_pre, dim=1)[1]
            accuracy = torch.mean((y_prediction == y_batch).float())
            accuracy_bei_epoch.append(accuracy.item())
            # print(accuracy)
            loss, I_X_T, I_Y_T = model.train_batch_loss(logits, features, z_scores, y_logits_s, mean_Cs, std_Cs, y_batch, num_samples=12)
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
    save(model, str(beta)+"CifarCausalVGG")

def causalVGG_eval(beta, model):
    model = load(model, str(beta)+"CifarCausalVGG")
    model.eval()
    accuracy_ = []
    I_X_T_ = []
    I_Y_T_ = []

    for x_batch, y_batch in test_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        y_pre, z_scores, features, logits, mean_Cs, std_Cs, y_logits_s = model(x_batch)
        loss, I_X_T, I_Y_T = model.train_batch_loss(logits, features, z_scores, y_logits_s, mean_Cs, std_Cs, y_batch, num_samples=12)
        I_X_T_.append(I_X_T.item())
        I_Y_T_.append(I_Y_T.item())

        y_prediction = torch.max(y_pre, dim=1)[1]
        accuracy = torch.mean((y_prediction == y_batch).float())
        accuracy_.append(accuracy.item())


    # if(epoch%5 == 0):
    print("TEST, Accuracy: ", np.mean(accuracy_), ", I_X_T: ",  np.mean(I_X_T_), ", I_Y_T: ", np.mean(I_Y_T_))

def NormalVGG_Train(model, ema, num_epoch):

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
    save(model, "NormalVGG")

def NormalVGG_eval(model):
    model = load(model,"NormalVGG")
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

        y_pre, z_scores, features, logits, mean_Cs, std_Cs, y_logits_s = model(x_batch)
        y_prediction = torch.max(y_pre, dim=1)[1]
        accuracy = torch.mean((y_prediction == y_batch).float())
        accuracy_clean.append(accuracy.item())

        perturbed_x_batch = adver_image_obtain.generate(x_batch, eps=epsilon, y=y_batch)

        y_pre, z_scores, features, logits, mean_Cs, std_Cs, y_logits_s = model(perturbed_x_batch)
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
