import numpy as np
import torch.utils.data as Data
import torch

class npLoader():
    def __init__(self):
        self.trainNum = 60000
        self.testNum = 10000
        self.fw = 45 #figure width
    def loadData(self, dataPath, train = True):
        data = np.fromfile(dataPath, dtype = np.uint8)
        if train:
            data_num = self.trainNum
        else:
            data_num = self.testNum
        data = data.reshape(data_num, self.fw, self.fw)
        return data
    
    def loadLabel(self, labelPath, train = True):
        label = np.fromfile(labelPath, dtype = np.uint8)
        return label

class My_MNIST(Data.Dataset):
    def __init__(self, data_path, label_path, train=True, transform=None, target_transform=None, download=False):
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.test_num = 10000
        self.train_num = 60000
        if train:
            self.data = torch.Tensor(np.fromfile(data_path,dtype=np.uint8).reshape(60000,1,45,45))/255
        else:
            self.data = torch.Tensor(np.fromfile(data_path,dtype=np.uint8).reshape(10000,1,45,45))/255
        self.label = torch.LongTensor(np.fromfile(label_path,dtype=np.uint8))
        
 
    def __getitem__(self, index):
        return self.data[index], self.label[index]
 
    def __len__(self):
        if self.train:
            return 60000
        else:
            return 10000