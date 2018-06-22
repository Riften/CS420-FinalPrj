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