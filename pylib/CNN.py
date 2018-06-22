import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

'''
Class: CNN
The CNN model used to do classification on MINIST dataset
'''
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(  # input shape (1, 45, 45)
            nn.Conv2d(
                in_channels=1,      # input height
                out_channels=16,    # n_filters
                kernel_size=5,      # filter size
                stride=1,           # filter movement/step
                padding=2,      #  padding=(kernel_size-1)/2 when stride=1
            ),      # output shape (16, 45, 45)
            nn.ReLU(),    # activation
            nn.MaxPool2d(kernel_size=2),    # output shape (16, 22, 22)
        )
        self.conv2 = nn.Sequential(  # input shape (16, 22, 22)
            nn.Conv2d(16, 32, 5, 1, 2),  # output shape (32, 22, 22)
            nn.ReLU(),  # activation
            nn.MaxPool2d(2),  # output shape (32, 11, 11)
        )
        self.out = nn.Linear(32 * 11 * 11, 10)   # fully connected layer, output 10 classes
 
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)   # 展平多维的卷积图成 (batch_size, 32 * 7 * 7)
        output = self.out(x)
        return output


# 3x3 Convolution
# It is used in Residual Block
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)

# Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

# ResNet Module
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.conv = conv3x3(1, 16)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 16, layers[0])
        self.layer2 = self.make_layer(block, 32, layers[0], 2)
        self.layer3 = self.make_layer(block, 64, layers[1], 2)
        self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64, num_classes)

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

'''
Function: train
Train CNN model useing data loaded by data loader.
Input:
    epoch: The corrent epoch
    model: Instance of CNN model.
    train_loader: Instance of torch.utils.data.DataLoader class.
    loss_f: loss function
    optimizer: The optimizer from torch.optim package.
'''
def train(epoch, model, train_loader, loss_f, optimizer):
    for batch_idx, (data, target) in enumerate(train_loader):  
        data, target = data.cuda(), target.cuda()  #Load data in GPU device.
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()   #Reset the gradient of optimizer
        output = model(data)  

        loss = loss_f(output, target)  
        loss.backward()  
        optimizer.step()  
        if batch_idx % 200== 0:  #Print the training result every 200 batches
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(  
                epoch, batch_idx * len(data), len(train_loader.dataset),  
                100. * batch_idx / len(train_loader), loss.data[0])) 

'''
Function test
Test CNN model.
'''
def test(epoch, model, test_loader, loss_f):
    test_loss = 0  
    correct = 0  
    for data, target in test_loader:
        data, target = data.cuda(), target.cuda()  
        data, target = Variable(data, volatile=True), Variable(target)  
        output = model(data)  
        test_loss += loss_f(output, target).data[0] # sum up batch loss  
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability  
        correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()  
    test_loss /= len(test_loader.dataset)  
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(  
        test_loss, correct, len(test_loader.dataset),  
        100. * correct / len(test_loader.dataset))) 

'''
Function used to reduce the learning rate.
It will change the 'lr' parameter of optimizer instance.
'''
def adjust_learning_rate(optimizer, decay_rate=.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay_rate