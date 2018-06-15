# 2018 CS420 Final Prj
The rest of codes will be uploaded soon.

## Package required
- numpy
- sklearn
- pytorch

## pylib
Folder pylib contains all the functions we write. We can use any function in any file by "from pylib.file import function"

### <font color=#0099ff>dataLoader.py</font>
Several classes used to load data and label.
#### class npLoader
Class used to load data for traditional classification models.

How to use:
```Python
from pylib.dataLoader import npLoader
loader = npLoader()
data = loader.loadData('mnist/mnist_test/mnist_test_data', train = False)
label = loader.loadLabel('mnist/mnist_test/mnist_test_label', train = False)
```
#### class My_MNIST
A class inherited from <code>torch.utils.data.Dataset</code>. Used to load data used for deep learning models based on pytorch.


### visual.py
Visualization Tools.
