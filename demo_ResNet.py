import torch
import torch.nn as nn
import torch.utils.data as Data
import numpy as np
from pylib.dataLoader import My_MNIST
from pylib.CNN import ResNet
from pylib.CNN import ResidualBlock
from pylib.CNN import train
from pylib.CNN import test
from pylib.CNN import adjust_learning_rate
import os

test_data_path = "mnist/mnist_test/mnist_test_data"
test_label_path = "mnist/mnist_test/mnist_test_label"
train_data_path = "mnist/mnist_train/mnist_train_data"
train_label_path = "mnist/mnist_train/mnist_train_label"
BATCH_SIZE = 50
test_set = My_MNIST(test_data_path, test_label_path, train=False)
train_set = My_MNIST(train_data_path, train_label_path, train = True)
train_loader = Data.DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = Data.DataLoader(dataset=test_set, batch_size=BATCH_SIZE, shuffle=True)

resnet = ResNet(ResidualBlock, [2, 2, 2, 2]).cuda()

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
lr = 0.001
optimizer = torch.optim.Adam(resnet.parameters(), lr=lr)
EPOCH = 50
# Training
#train(epoch, model, train_loader, loss_f, optimizer)
for epoch in range(EPOCH):
    train(epoch, resnet, train_loader, criterion, optimizer)
    test(epoch, resnet, test_loader, criterion)
    adjust_learning_rate(optimizer)
