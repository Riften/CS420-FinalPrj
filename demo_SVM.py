import pickle
from sklearn.svm import SVC
from pylib.traditional import traditioanl_classifier
import threading

#SVM with different kernel
train_data = pickle.load(open('output/pca/pca_50_train_data.pkl', 'r'))
test_data = pickle.load(open('output/pca/pca_50_test_data.pkl', 'r'))
train_label = loader.loadLabel(train_label_path, train = True)
test_label = loader.loadLabel(test_label_path, train =False)
kernels = ['rbf', 'poly', 'sigmoid']
#kernels = ['linear']
outdir = 'output/SVM'
threads = [0]*3
for (i,kernel) in enumerate(kernels):
    reportpath = 'SVM_%s_report_50' % kernel
    modelpath = 'SVM_%s_model_50' % kernel
    reportpath = os.path.join(outdir, reportpath)
    modelpath = os.path.join(outdir, modelpath)
    
    model = SVC(kernel = kernel)
    threads[i] =threading.Thread(target = traditioanl_classifier, \
                                 args=(model, train_data, train_label, test_data, test_label, reportpath, modelpath))
    threads[i].start()