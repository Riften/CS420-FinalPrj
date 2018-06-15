import numpy as np
from PIL import Image
import random
from pylib.dataLoader import npLoader
import os

'''
Function: visWrongData
Visualize and save the samples misclassified by a trained traditional model.
Input:
    model: traditional classification model
    n: Number of misclassified samples you want to visualize.
    data_raw: Image data loaded from data file using dataLoader.
    data: Corresponding data used to test model.
    label: Corresponding labels of data.
    outdir: Directory path to save result.
Output:
    'n' misclassified samples saved in 'outdir'.
'''
def visWrongData(model, n, data_raw, data, label, outdir=''):
    pred = model.predict(data)      #Predict
    pos = np.where(pred!=label)[0]  #Find the index of the misclassified samples

    #Sample n samples.
    #If there is not enough samples, we will visualize all of them.
    if pos.shape[0]<n:
        n = pos.shape[0]
    sampos = random.sample(pos, n)
    
    for (i,p) in enumerate(sampos):
        im = Image.fromarray(data_raw[p])
        if outdir!='':
            #Name of files:
            #Index_TruthLabel_ClassifiedLabel
            filepath = os.path.join(outdir, '%d_%d_%d.png'%(i,label[p],pred[p]))
            im.save(filepath)