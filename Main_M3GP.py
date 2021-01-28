import pandas

from m3gp.M3GP import M3GP
from sys import argv
from m3gp.Constants import *
import os

from sklearn.model_selection import train_test_split

import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score

# 
# By using this file, you are agreeing to this product's EULA
#
# This product can be obtained in https://github.com/jespb/Python-M3GP
#
# Copyright ©2019 J. E. Batista
#



filename= "file51e4e5741f2.csv"

ds = pandas.read_csv(DATASETS_DIR+filename)
class_header = ds.columns[-1]

Tr_X, Te_X, Tr_Y, Te_Y = train_test_split(ds.drop(columns=[class_header]), ds[class_header], 
		train_size=TRAIN_FRACTION, random_state=42)

# Train a model
m3gp = M3GP("Entropy With Diagonal")
m3gp.fit(Tr_X, Tr_Y)

pred = m3gp.predict(Te_X)
pred = [float(x) for x in pred]
print( balanced_accuracy_score(pred, Te_Y) )


