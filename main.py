#!/usr/bin/env python
# -*- coding: utf-8 -*-

import scipy.io
import numpy as np
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt
from scipy.signal import freqz
import scipy.linalg as la

#import mne

# Loading data from matfiles filtered with bandpass to 1000Hz
eeg_calib = scipy.io.loadmat('./data/BCICIV_1calib_1000Hz_mat/BCICIV_calib_ds1a_1000Hz.mat')
eeg_eval = scipy.io.loadmat('./data/BCICIV_1eval_1000Hz_mat/BCICIV_eval_ds1a_1000Hz.mat')

#path_file = './data/BCICIV_1eval_1000Hz_mat/BCICIV_eval_ds1a_1000Hz.mat'
#Definition of channel types and names.
#sfreq = 2500  # Sampling frequency
#ch_types = ['eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eog']
#ch_names = ['c3', 'c4', 'p3', 'p4', 'o1', 'o2', 'eog']

#info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

#raw_data = io.loadmat(path_file)

#Bandpass filtering methods with butter method

def butter_bandpass(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = lfilter(b, a, data)
    return y

# CSP takes any number of arguments, but each argument must be a collection of trials associated with a task
# That is, for N tasks, N arrays are passed to CSP each with dimensionality (# of trials of task N) x (feature vector)
# Trials may be of any dimension, provided that each trial for each task has the same dimensionality,
# otherwise there can be no spatial filtering since the trials cannot be compared
def CSP(*tasks):
	if len(tasks) < 2:
		print("Must have at least 2 tasks for filtering.")
		return (None,) * len(tasks)
	else:
		filters = ()
		# CSP algorithm
		# For each task x, find the mean variances Rx and not_Rx, which will be used to compute spatial filter SFx
		iterator = range(0,len(tasks))
		for x in iterator:
			# Find Rx
			Rx = covarianceMatrix(tasks[x][0])
			for t in range(1,len(tasks[x])):
				Rx += covarianceMatrix(tasks[x][t])
			Rx = Rx / len(tasks[x])

			# Find not_Rx
			count = 0
			not_Rx = Rx * 0
			for not_x in [element for element in iterator if element != x]:
				for t in range(0,len(tasks[not_x])):
					not_Rx += covarianceMatrix(tasks[not_x][t])
					count += 1
			not_Rx = not_Rx / count

			# Find the spatial filter SFx
			SFx = spatialFilter(Rx,not_Rx)
			filters += (SFx,)

			# Special case: only two tasks, no need to compute any more mean variances
			if len(tasks) == 2:
				filters += (spatialFilter(not_Rx,Rx),)
				break
		return filters

# covarianceMatrix takes a matrix A and returns the covariance matrix, scaled by the variance
def covarianceMatrix(A):
	Ca = np.dot(A,np.transpose(A))/np.trace(np.dot(A,np.transpose(A)))
	return Ca

# spatialFilter returns the spatial filter SFa for mean covariance matrices Ra and Rb
def spatialFilter(Ra,Rb):
	R = Ra + Rb
	E,U = la.eig(R)

	# CSP requires the eigenvalues E and eigenvector U be sorted in descending order
	ord = np.argsort(E)
	ord = ord[::-1] # argsort gives ascending order, flip to get descending
	E = E[ord]
	U = U[:,ord]

	# Find the whitening transformation matrix
	P = np.dot(np.sqrt(la.inv(np.diag(E))),np.transpose(U))

	# The mean covariance matrices may now be transformed
	Sa = np.dot(P,np.dot(Ra,np.transpose(P)))
	Sb = np.dot(P,np.dot(Rb,np.transpose(P)))

	# Find and sort the generalized eigenvalues and eigenvector
	E1,U1 = la.eig(Sa,Sb)
	ord1 = np.argsort(E1)
	ord1 = ord1[::-1]
	E1 = E1[ord1]
	U1 = U1[:,ord1]

	# The projection matrix (the spatial filter) may now be obtained
	SFa = np.dot(np.transpose(U1),P)
	return SFa.astype(np.float32)

# Got from https://github.com/spolsley/common-spatial-patterns

# Sample rate and desired cutoff frequencies (in Hz).
fs = 1000.0
lowcut = 0.05
highcut = 200.0

# Get items from calibration and evaluation files
items_calib = eeg_calib['cnt'].T
items = eeg_eval['cnt'].T

# Get time markers from calibration
time_markers = eeg_calib['mrk'][0][0][0][0]

# Get true values from each trial 
true_classes = eeg_calib['mrk'][0][0][1][0]

# Get info fro evaluation file
info = eeg_eval['nfo'][0][0]

freq = info[0]
classes = []
channels = []

for clase in info[1][0]:
	classes.append(clase[0])

for channel in info[2][0]:
	channels.append(channel[0])

tasks_left = []
tasks_foot = []
#y = butter_bandpass(items_calib[0], lowcut, highcut, fs, order=6)

start = 0
diffs = []

# Get the minimum difference between intervals
for i in range(len(time_markers)):
	diffs.append(time_markers[i]-start)
	start = time_markers[i]

difference = min(diffs)
eeg_complete = []

# Exclude last 16 trials and iterate inside time markers
for i in range(len(time_markers)-16):
	trials = []
	# Cut signals on regular intervals
	for item in items:
		trials.append(item[time_markers[i]-difference:time_markers[i]])
	if true_classes[i] == 1:
		tasks_foot.append(trials)
	else:
		tasks_left.append(trials)
	eeg_complete.append(trials)

# Passing tasks to CSP method
csp = CSP(tasks_left, tasks_foot)

x = []
y = []

# Concatenate and Get convolved CSP output
for i in range(len(eeg_complete)):
	if true_classes[i] == 1:
		index = 1
	else:
		index = 0
	filtered = np.array([])
	for j in range(len(eeg_complete[i])):
		filtered = np.concatenate([filtered, np.convolve(csp[index][j], eeg_complete[i][j]) ])
	x.append(filtered)
	y.append(index)
	
# Divide data into training and test sets
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.30)

# Single classification with Support Vector Classifier
from sklearn.svm import SVC
clf = SVC(kernel='linear')
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

# Single classification result
print( confusion_matrix(y_test,y_pred) )
# print( classification_report(y_test,y_pred) )

# Leave One Out with Support Vector Classifier
from sklearn.model_selection import LeaveOneOut

x = np.array(x)
y = np.array(y)

results = []

loo = LeaveOneOut()
for train, test in loo.split(x):
	X_train = x[train]
	y_train = y[train]
	X_test = x[test]
	y_test = y[test]

	clf = SVC(kernel='linear')
	clf.fit(X_train, y_train)
	y_pred = clf.predict(X_test)

	results.append( confusion_matrix(y_test,y_pred) )
	# print( classification_report(y_test,y_pred) )

# List of confusion matrix with true positives
print(results)