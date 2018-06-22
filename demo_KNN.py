import numpy as np
import threading
import pickle
from sklearn.neighbors import KNeighborsClassifier
import os

#KNN on Different PCA Dimension
components = [50,75,100,125,150,175,200]
threads = [0]*7
datadir = 'output/pca'
resdir = 'output/KNN'
train_label = loader.loadLabel(train_label_path, train = True)
test_label = loader.loadLabel(test_label_path, train =False)
for (i,c) in enumerate(components):
    trainpath = "pca_%d_train_data.pkl" % c
    testpath = "pca_%d_test_data.pkl" % c
    trainpath = os.path.join(datadir, trainpath)
    testpath = os.path.join(datadir, testpath)
    train_data = pickle.load(open(trainpath, 'r'))
    test_data = pickle.load(open(testpath, 'r'))
    
    reportpath = 'KNN_report_%d.txt' % c
    modelpath = 'KNN_model_%d.pkl' % c
    reportpath = os.path.join(resdir, reportpath)
    modelpath = os.path.join(resdir, modelpath)
    model = KNeighborsClassifier(n_neighbors=10)
    threads[i] =threading.Thread(target = traditioanl_classifier, \
                                 args=(model, train_data, train_label, test_data, test_label, reportpath, modelpath))
    threads[i].start()