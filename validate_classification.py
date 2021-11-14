#!/usr/bin/env python
# -*- coding: utf-8 -*-

from time import time
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import scikitplot as skplt


start_time = time()

csp = scipy.io.loadmat('./data/training/csp-min-trials.mat')['data']

# csp_1 = scipy.io.loadmat('./data/training/csp-left-right.mat')['data']
# csp_2 = scipy.io.loadmat('./data/training/csp-left-foot.mat')['data']
# csp_3 = scipy.io.loadmat('./data/training/csp-right-foot.mat')['data']

eval_signal = scipy.io.loadmat('./data/training/x_train_eval.mat')['data']
y_test = scipy.io.loadmat('./data/training/y_train_eval.mat')['data'][0]

X_train = scipy.io.loadmat('./data/training/x_filtered.mat')['data']
y_train = scipy.io.loadmat('./data/training/y_train_parted.mat')['data'][0]

clf = SVC(kernel='linear')
clf.fit(X_train, y_train)

print('model fitted')

y_pred = []

for i in range(len(eval_signal)):
	X_test = []
	filtered = np.array([])
	for j in range(len(eval_signal[i])):
		filt_1 = np.convolve(csp[0][j], eval_signal[i][j])
		filt_2 = np.convolve(csp[1][j], filt_1)
		filtered = np.concatenate([filtered, np.convolve(csp[2][j], filt_2) ])
	X_test = [filtered]
	y_pred.append(clf.predict(X_test)[0])

print(y_test[:50], y_pred[:50])
print(accuracy_score(y_test, y_pred)*100)

skplt.metrics.plot_roc_curve(y_test, y_test)
plt.show()

# print(y_train, y_pred)

elapsed_time = time() - start_time
print("Elapsed time: %.10f seconds." % elapsed_time)
