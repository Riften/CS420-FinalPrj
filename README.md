# 2018 CS420 Final Prj
The rest of codes will be uploaded soon.

## Package required
- numpy
- sklearn
- pytorch

## pylib
Folder pylib contains all the functions we write. We can use any function in any file by "from pylib.file import function"

### dataLoader.py
Several classes used to load data and label.
#### npLoader()
Function used to load data for traditional classification models.
#### 
How to use:

```Python
from pylib.dataLoader import npLoader
loader = npLoader()
data = loader.loadData('mnist/mnist_test/mnist_test_data', train = False)
label = loader.loadLabel('mnist/mnist_test/mnist_test_label', train = False)
```
### visual.py
Visualization Tools.
