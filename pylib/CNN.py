import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

'''

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
                padding=2,      # 如果想要 con2d 出来的图片长宽没有变化, padding=(kernel_size-1)/2 当 stride=1
            ),      # output shape (16, 45, 45)
            nn.ReLU(),    # activation
            nn.MaxPool2d(kernel_size=2),    # 在 2x2 空间里向下采样, output shape (16, 22, 22)
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
#cnn = CNN()
#print(cnn)  # net architecture