from sklearn.decomposition import PCA
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import pickle

'''
Use sklearn package to do PCA.
Note that we will only fit PCA model on training data.
'''
def pcaRed(train_data, test_data, n_components, outdir):
    modelpath = "pca_%d_model.pkl" % n_components
    trainpath = "pca_%d_train_data.pkl" % n_components
    testpath = "pca_%d_test_data.pkl" % n_components
    modelpath = os.path.join(outdir, modelpath)
    trainpath = os.path.join(outdir, trainpath)
    testpath = os.path.join(outdir, testpath)
    print "PCA === %d" % n_components
    pca = PCA(n_components = n_components)
    
    print "...fit %d" % n_components
    pca.fit(train_data)
    
    print "...transform %d" % n_components
    newtrain = pca.transform(train_data)
    newtest = pca.transform(test_data)
    
    print "...save"
    pickle.dump(pca, open(modelpath, 'wb'))
    pickle.dump(newtrain, open(trainpath, 'wb'))
    pickle.dump(newtest, open(testpath, 'wb'))
    print "...... Write model to %s" % modelpath
    print "...... Write train data to %s" % trainpath
    print "...... Write test data to %s" % testpath

'''
Function: traditional_classifier
Use sklearn package to classify data.
Input:
    model: Classifier from sklearn package.
    train/test_data/label: Data and label loaded by npLoader from dataLoader.py.
    reportpath: Path to save result report.
    modelpath: Path to store the trained model.
'''
def traditioanl_classifier(model, train_data, train_label, test_data, test_label, reportpath, modelpath = ''):
    print '...fit'
    model.fit(train_data, train_label)
    
    print '...predict'
    train_pred = model.predict(train_data)
    test_pred = model.predict(test_data)
    
    print '...Write report to %s' % reportpath
    r = open(reportpath, 'w')
    r.write("==== Train ====\n")
    r.write(classification_report(train_label,train_pred)+'\n')
    r.write('accuracy: '+str(accuracy_score(train_label, train_pred))+'\n')
    r.write("==== Test ====\n")
    r.write(classification_report(test_label,test_pred)+'\n')
    r.write('accuracy: '+str(accuracy_score(test_label, test_pred))+'\n')
    r.close()
    
    if modelpath!='':
        print 'Write model to %s' % modelpath
        m = open(modelpath, 'wb')
        pickle.dump(model, m)
        m.close()