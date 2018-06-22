# pylib
Folder pylib contains all the functions we write. We can use any function in any file by "from pylib.file import function"

## dataLoader.py
Several classes used to load data and label.
### class npLoader
Class used to load data for traditional classification models.

How to use:
```Python
from pylib.dataLoader import npLoader
loader = npLoader()
data = loader.loadData('mnist/mnist_test/mnist_test_data', train = False)
label = loader.loadLabel('mnist/mnist_test/mnist_test_label', train = False)
```
### class My_MNIST
A class inherited from <code>torch.utils.data.Dataset</code>. Used to load data used for deep learning models based on pytorch.

How to use:
```Python
test_set = My_MNIST(test_data_path, test_label_path, train=False)
train_set = My_MNIST(train_data_path, train_label_path, train = True)
train_loader = Data.DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = Data.DataLoader(dataset=test_set, batch_size=BATCH_SIZE, shuffle=True)
```

## CNN.py
Definition of CNN model and ResNet model. 

It also contains the train and test function for CNN and ResNet model.

## traditional.py
Functions used to do PCA and fit traditional classifiers using sklearn package.

## visual.py
Visualization Tools.
