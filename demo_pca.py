from pylib.traditional import pcaRed
import os
import pickle
import threading
from pylib.dataLoader import npLoader

loader = npLoader()
test_data_path = "mnist/mnist_test/mnist_test_data"
test_label_path = "mnist/mnist_test/mnist_test_label"
train_data_path = "mnist/mnist_train/mnist_train_data"
train_label_path = "mnist/mnist_train/mnist_train_label"
train_data = loader.loadData(train_data_path, train = True, transform = False)
test_data = loader.loadData(test_data_path, train = False, transform = False)

#Doing PCA to reduce data into different dimension.
components = [50,75,100,125,150,175,200]
threads = [0]*7
for i in xrange(7):
    threads[i] =threading.Thread(target=pcaRed,args=(train_data,test_data, components[i], 'output/pca'))
    threads[i].start()