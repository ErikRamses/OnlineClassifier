#!/usr/bin/env python
# -*- coding: utf-8 -*-

from time import time
import scipy.io
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

start_time = time()

csp = scipy.io.loadmat('./data/training/csp-min-trials.mat')['data']
# csp = scipy.io.loadmat('./data/training/csp-1600.mat')['data']

eval_signal = signal.unit_impulse(100, 'mid')
filtered = np.array([])
X_test = []

print(csp.shape)

channel = 0

filt_1 = np.convolve(csp[0][channel], eval_signal)
filt_2 = np.convolve(csp[1][channel], filt_1)
filtered = np.concatenate([filtered, np.convolve(csp[2][channel], filt_2) ])

L = 1000

# yf = np.fft.fft(filtered)
w, h = signal.freqz(filtered)

# plt.plot(H(filtered))
plt.title('Frequency response 1 trials/class')
plt.plot(w/np.pi,20*np.log10(abs(h)))
plt.xlabel('Normalized frequency (times x pi rad/sample)')
plt.ylabel('Magnitude (dB)')
plt.ylim([-40, 60])
plt.show()
# csp_1 = scipy.io.loadmat('./data/training/csp-left-right.mat')['data']
# csp_2 = scipy.io.loadmat('./data/training/csp-left-foot.mat')['data']
# csp_3 = scipy.io.loadmat('./data/training/csp-right-foot.mat')['data']

'''

eval_signal = scipy.io.loadmat('./data/training/x_test.mat')['data']

X_train = scipy.io.loadmat('./data/training/x_filtered.mat')['data']

y_train = scipy.io.loadmat('./data/training/y_train.mat')['data'][0]

# y_test = scipy.io.loadmat('./true_labels/BCICIV_eval_ds1a_1000Hz_true_y.mat')['true_y']

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

# print(accuracy_score(y_train, y_pred))
score = clf.score(X_test_data, y_pred)

print(score)

'''

elapsed_time = time() - start_time
print("Elapsed time: %.10f seconds." % elapsed_time)
