import numpy as np

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