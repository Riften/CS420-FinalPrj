import torch
import torch.nn as nn
import torch.utils.data as Data
import numpy as np
from pylib.dataLoader import My_MNIST
from pylib.CNN import CNN
from pylib.CNN import train
from pylib.CNN import test
from pylib.CNN import adjust_learning_rate
import os

test_data_path = "mnist/mnist_test/mnist_test_data"
test_label_path = "mnist/mnist_test/mnist_test_label"
train_data_path = "mnist/mnist_train/mnist_train_data"
train_label_path = "mnist/mnist_train/mnist_train_label"
cnn = CNN().cuda()
test_set = My_MNIST(test_data_path, test_label_path, train=False)
train_set = My_MNIST(train_data_path, train_label_path, train = True)
EPOCH = 10
BATCH_SIZE = 50
LR = 0.001

optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()
train_loader = Data.DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = Data.DataLoader(dataset=test_set, batch_size=BATCH_SIZE, shuffle=True)
for epoch in range(EPOCH):
    train(epoch, cnn, train_loader, loss_func, optimizer)
    test(epoch, cnn, test_loader, loss_func)
    adjust_learning_rate(optimizer)