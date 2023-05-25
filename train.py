import random
from distutils.util import strtobool

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
import torch
import torchvision.datasets as datasets
from torchvision.transforms import ToTensor, Compose, Resize, Normalize 
from torch.utils.tensorboard import SummaryWriter
import os
import torch.nn as nn
from ResNet18 import ResNet18


def load_data():
    data_transform = Compose([Resize((224, 224)),ToTensor(), Normalize((.1307,), (.3081,))])
    mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=data_transform)
    mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=data_transform)
    training_loader = torch.utils.data.DataLoader(mnist_trainset, batch_size=128, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(mnist_testset, batch_size=128, shuffle=False)
    return training_loader, validation_loader

def loss_on_test(test_loader, net, criterion,device="cuda:0"):
    running_loss = 0.0
    net.eval()
    for i, (inputs, labels) in enumerate(test_loader):
        inputs, labels = inputs.to(device), labels.to(device)        
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        running_loss += loss.item()
    
    net.train()
    return running_loss/i

def train(trainloaer, testloader, hyperparameters):
    
    device= hyperparameters["device"]
    EPOCHS = hyperparameters["EPOCHS"]
    criterion = hyperparameters["criterion"]
    net = hyperparameters["net"].to(device)
    optimizer = hyperparameters["optimizer"]
    seed = hyperparameters["seed"]
    EXP_NAME = hyperparameters["EXP_NAME"]
    
    torch.manual_seed(seed)
    writer = SummaryWriter(os.path.join("./","runs", EXP_NAME))

    for epoch in tqdm(range(EPOCHS)):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, (inputs, labels) in tqdm(enumerate(trainloaer)):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()


            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            del inputs
            del labels


        writer.add_scalar("Loss/train", running_loss/i, epoch)
        writer.add_scalar("Loss/test", loss_on_test(validation_loader, net,criterion), epoch)







if __name__ == "__main__":

    
    print(10*"#", " Starting Training Script",10*"#")
    net = ResNet18()
    hyperparameters = {
        "device" : torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),
        "EPOCHS": int(6e5 / 256),
        "criterion" : nn.CrossEntropyLoss(),
        "net": net,
        "optimizer": torch.optim.SGD(net.parameters(),lr = 1e-1, weight_decay=0.0001, momentum=0.9),
        "seed" : 42,
        "EXP_NAME": "test" + str(random.randint(0,1000000))
    }
    
    training_loader,validation_loader = load_data()
    net = train(training_loader, validation_loader, hyperparameters)