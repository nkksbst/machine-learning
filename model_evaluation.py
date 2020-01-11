# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 09:57:53 2020
@author: Monikka Busto
"""

import numpy as np
from sklearn import metrics

# binary classification
y_truth = np.random.randint(0,2, size = 5)
y_pred = np.ones(5, dtype=np.int32)

metrics.accuracy_score(y_truth, y_pred)
metrics.precision_score(y_truth, y_pred)
metrics.recall_score(y_truth, y_pred)

true_positive = np.sum((y_pred == 1) * (y_truth == 1))
true_negative = np.sum((y_pred == 0) * (y_truth == 0))
false_positive = np.sum((y_pred == 1) * (y_truth == 0))
false_negative = np.sum((y_pred == 0) * (y_truth == 1))
accuracy = (true_positive + true_negative)/len(y_truth)
precision = true_positive/(true_positive + false_positive)
recall = true_positive/(true_positive + false_negative)

# regression
