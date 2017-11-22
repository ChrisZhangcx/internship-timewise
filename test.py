# -*- coding:utf8 -*-
import numpy as np
from Preprocessing import ProcessingMethod as pm
from sklearn.preprocessing import scale

y = np.array([[1,2,3,4,5,6,7,8,9],[2,3,4,5,6,7,8,9,0]])
print scale(y)
y = np.transpose(y)
print scale(y)
print pm.discretization(y, kinds=3)
