#!/usr/bin/env python
# -*- coding: utf-8 -*-

from time import time
import scipy.io
import numpy as np

start_time = time()

csp = scipy.io.loadmat('./data/training/csp-min-trials.mat')['data']

X_train = scipy.io.loadmat('./data/training/x_train.mat')['data']

x = []

# Concatenate and Get convolved CSP output
for i in range(len(X_train)):
	filtered = np.array([]) 
	for j in range(len(X_train[i])):
		filt_1 = np.convolve(csp[0][j], X_train[i][j])
		filt_2 = np.convolve(csp[1][j], filt_1)
		filtered = np.concatenate([filtered, np.convolve(csp[2][j], filt_2) ])
	x.append(filtered)

print(len(x))

x_data = {"data": x, "label": "labels_calib"}

scipy.io.savemat('./data/training/x_filtered.mat', x_data)

elapsed_time = time() - start_time
print("Elapsed time: %.10f seconds." % elapsed_time)
