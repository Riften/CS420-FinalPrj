# 2018 CS420 Final Prj
test

## pylib
Folder pylib contains all the functions we write. We can use any function in any file by "from pylib.file import function"

### dataLoader.py
Several classes used to load data and label.
'''Python
from pylib.dataLoader import npLoader
loader = npLoader()
data = loader.loadData('mnist/mnist_test/mnist_test_data', train = False)
label = loader.loadLabel('mnist/mnist_test/mnist_test_label', train = False)
'''
