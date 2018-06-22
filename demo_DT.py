import numpy as np
import threading
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from pylib.traditional import traditioanl_classifier
from sklearn.metrics import zero_one_loss
import matplotlib.pyplot as plt
from pylib.dataLoader import npLoader
import os

loader = npLoader()
#Decision Tree and Boosting
train_data = pickle.load(open('output/pca/pca_50_train_data.pkl', 'r'))
test_data = pickle.load(open('output/pca/pca_50_test_data.pkl', 'r'))
train_label = loader.loadLabel(train_label_path, train = True)
test_label = loader.loadLabel(test_label_path, train =False)

learning_rate = 1.
n_estimators = 1000  #Try to boosting 1000 models together
outdir = 'output/DecisionTree'

#Deepdecision Tree With depth 100
dt = DecisionTreeClassifier(max_depth=100, min_samples_leaf=1)

#Decision Tree with very deep depth.
dtd = DecisionTreeClassifier()

#Basic model of boosting
dt_stump = DecisionTreeClassifier(max_depth=10,min_samples_leaf=1)
dt_stump.fit(train_data, train_label)
#dt_stump_err = 1.0 - dt_stump.score(X_test, y_test)

#Boosting model
ada_discrete = AdaBoostClassifier(
    base_estimator=dt_stump,
    learning_rate=learning_rate,
    n_estimators=n_estimators,
    algorithm="SAMME")

#ada_discrete.fit(X_train, y_train)
reportpath1 = os.path.join(outdir, 'Decision_deep_report.txt')
reportpath2 = os.path.join(outdir, 'Decision_100_report.txt')
reportpath3 = os.path.join(outdir, 'Decision_boosting_stump_10_report.txt')

modelpath1 = os.path.join(outdir, 'Decision_deep_model.pkl')
modelpath2 = os.path.join(outdir, 'Decision_100_model.pkl')
modelpath3 = os.path.join(outdir, 'Decision_boosting_stump_10_model.pkl')

traditioanl_classifier(dtd, train_data, train_label, test_data, test_label, reportpath1, modelpath1)
traditioanl_classifier(dt, train_data, train_label, test_data, test_label, reportpath2, modelpath2)
traditioanl_classifier(ada_discrete, train_data, train_label, test_data, test_label, reportpath3, modelpath3)

dt_err = 1.0 - dt.score(test_data, test_label)
dt_stump_err = 1.0 - dt_stump.score(test_data, test_label)
ada_discrete_test_err = np.zeros((n_estimators,))
for i, test_pred in enumerate(ada_discrete.staged_predict(test_data)):
    ada_discrete_test_err[i] = zero_one_loss(test_label, test_pred)

ada_discrete_train_err = np.zeros((n_estimators,))
for i, train_pred in enumerate(ada_discrete.staged_predict(train_data)):
    ada_discrete_train_err[i] = zero_one_loss(train_label, train_pred)


fig = plt.figure(figsize=(6,4))
ax = fig.add_subplot(111)
ax.plot([1, n_estimators], [dt_stump_err] * 2, 'k-',
        label='Depth = 10')
ax.plot([1, n_estimators], [dt_err] * 2, 'k--',
        label='Depth = 100')
ax.plot(np.arange(n_estimators) + 1, ada_discrete_test_err,
        label='Boosting 1000 Trees - test',
        color='red')
ax.plot(np.arange(n_estimators) + 1, ada_discrete_train_err,
        label='Boosting 1000 Trees - train',
        color='blue')
leg = ax.legend(loc='upper right', fancybox=True)
leg.get_frame().set_alpha(0.7)
ax.set_xlabel('nunber of estimators')
ax.set_ylabel('error rate')

fig.savefig('output/DecisionTree/treeBoosting.pdf')