#!/usr/bin/env python
# -*- coding: utf-8 -*-

from time import time
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

start_time = time()

csp = scipy.io.loadmat('./data/training/csp-min-trials.mat')['data']

# csp_1 = scipy.io.loadmat('./data/training/csp-left-right.mat')['data']
# csp_2 = scipy.io.loadmat('./data/training/csp-left-foot.mat')['data']
# csp_3 = scipy.io.loadmat('./data/training/csp-right-foot.mat')['data']

eval_signal = scipy.io.loadmat('./data/training/x_train_eval.mat')['data']
y_test = scipy.io.loadmat('./data/training/y_train_eval.mat')['data'][0]

X_train = scipy.io.loadmat('./data/training/x_filtered.mat')['data']

y_train = scipy.io.loadmat('./data/training/y_train.mat')['data'][0]

clf = SVC(kernel='linear')
clf.fit(X_train, y_train)

y_pred = []
X_test_data = []

for i in range(len(eval_signal)):
	X_test = []
	filtered = np.array([])
	for j in range(len(eval_signal[i])):
		filt_1 = np.convolve(csp[0][j], eval_signal[i][j])
		filt_2 = np.convolve(csp[1][j], filt_1)
		filtered = np.concatenate([filtered, np.convolve(csp[2][j], filt_2) ])
	X_test = [filtered]
	X_test_data.append(filtered)
	y_pred.append(clf.predict(X_test)[0])

y_pred = np.array(y_pred)

print(y_test, y_pred)
score = accuracy_score(y_test, y_pred)

print(score)

elapsed_time = time() - start_time
print("Elapsed time: %.10f seconds." % elapsed_time)
