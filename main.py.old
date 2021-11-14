#!/usr/bin/env python
# -*- coding: utf-8 -*-

import scipy.io
import numpy as np
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt
from scipy.signal import freqz
import scipy.linalg as la
from sklearn.metrics import mean_squared_error
from matplotlib.collections import LineCollection

# Loading data from matfiles filtered with bandpass to 1000Hz
# eeg_calib = scipy.io.loadmat('./data/BCICIV_1calib_1000Hz_mat/BCICIV_calib_ds1a_1000Hz.mat')
# eeg_eval = scipy.io.loadmat('./data/BCICIV_1eval_1000Hz_mat/BCICIV_eval_ds1a_1000Hz.mat')

# Not filtered signal
# eeg_calib = scipy.io.loadmat('./data/BCICIV_1_mat/BCICIV_calib_ds1a.mat')
# eeg_eval = scipy.io.loadmat('./data/BCICIV_1_mat/BCICIV_eval_ds1a.mat')

# Loading data from matfiles filtered with bandpass to 1000Hz
eeg_calib = scipy.io.loadmat('./data/BCICIV_1calib_1000Hz_mat/BCICIV_calib_ds1a_1000Hz.mat')
eeg_eval = scipy.io.loadmat('./data/BCICIV_1eval_1000Hz_mat/BCICIV_eval_ds1a_1000Hz.mat')
# eeg_true = scipy.io.loadmat('./true_labels/BCICIV_eval_ds1a_1000Hz_true_y.mat')

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

#plt.plot(items[1], 'b',label='Ch CFC3 Vectorized')
#plt.title('Senales')
#plt.ylabel('Magnitud')
#plt.xlabel('Frecuencia')
#plt.legend(loc='upper right')
#plt.grid(True)
#plt.tight_layout()
#plt.show()

for clase in info[1][0]:
	classes.append(clase[0])

print(classes)

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

first = time_markers[0]

y_train = []

# Exclude last 16 trials and iterate inside time markers
for i in range(1, len(time_markers)):
	trials = []
	# Cut signals on regular intervals
	if((time_markers[i] - time_markers[i-1]) < difference+10):
		for item in items_calib:
			trials.append(item[time_markers[i]-difference:time_markers[i]])
		if true_classes[i] == 1:
			tasks_foot.append(trials)
		else:
			tasks_left.append(trials)

class_1 = []
class_2 = [] 

for trial in tasks_foot:
	for i in range(0, 8000, 1000):
		canales = []
		for channel in trial:
			canales.append(channel[i:i+1000])
		class_1.append(canales)
		y_train.append(1)

for trial in tasks_left:
	for i in range(0, 8000, 1000):
		canales = []
		for channel in trial:
			canales.append(channel[i:i+1000])
		class_2.append(canales)
		y_train.append(0)

# Passing tasks to CSP method
# csp = CSP(tasks_left, tasks_foot)
new_data = []

for i in range(len(items_calib)):
	new_data.append(items_calib[i][:10000])

new_data = np.asarray(new_data)

data = new_data
n_rows = len(data)
n_samples = len(data[0])

print(len(data[2]))

# Plot the EEG
fig, ax = plt.subplots()

ticklocs = []
t = np.arange(n_samples)
ax.set_xlim(0, n_samples)
ax.set_xticks(np.arange(n_samples))
dmin = data.min()
dmax = data.max()
dr = (dmax - dmin) * 0.7  # Crowd them a bit.
y0 = dmin
y1 = (n_rows - 1) * dr + dmax
ax.set_ylim(y0, y1)

segs = []
for i in range(n_rows):
    segs.append(np.column_stack((t, data[:, i])))
    ticklocs.append(i * dr)

offsets = np.zeros((n_rows, 2), dtype=float)
offsets[:, 1] = ticklocs

lines = LineCollection(segs, offsets=offsets, transOffset=None)
ax.add_collection(lines)

# Set the yticks to use axes coordinates on the y axis
ax.set_yticks(ticklocs)
ax.set_yticklabels(channels)

ax.set_xlabel('x')

plt.title('CSP Right Hand')
plt.tight_layout()
plt.show()

'''

parts = int( len(items[0]) / 1000 )
eval_signal = []
ini = 0

for chain in range(1, parts):
	trial = []
	for channel in items:
		trial.append(channel[ini:chain*1000])
	ini = chain*1000
	eval_signal.append(trial)

x_1 = []
x_0 = []

# Concatenate and Get convolved CSP output

for i in range(len(class_1)):
	filtered = np.array([])
	for j in range(len(class_1[i])):
		filtered = np.concatenate([filtered, np.convolve(csp[1][j], class_1[i][j]) ])
	x_1.append(filtered)

for i in range(len(class_2)):
	filtered = np.array([])
	for j in range(len(class_2[i])):
		filtered = np.concatenate([filtered, np.convolve(csp[1][j], class_2[i][j]) ])
	x_0.append(filtered)

X_train = np.concatenate([x_1, x_0])

x_1 = []
x_0 = []
y_test = []

for i in range(len(eval_signal)):
	filtered_1 = np.array([])
	filtered_0 = np.array([])
	for j in range(len(eval_signal[i])):
		filtered_1 = np.concatenate([filtered_1, np.convolve(csp[1][j], eval_signal[i][j]) ])
		filtered_0 = np.concatenate([filtered_0, np.convolve(csp[0][j], eval_signal[i][j]) ])
	x_1.append(filtered_1)
	y_test.append(1)
	x_0.append(filtered_0)
	y_test.append(0)

X_test = np.concatenate([x_1, x_0])

# Divide data into training and test sets
# from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.80)

# Single classification with Support Vector Classifier
from sklearn.svm import SVC
clf = SVC(kernel='linear')
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

des_func = clf.decision_function(X_test)

# Single classification result
print( confusion_matrix(y_test,y_pred) )

print( mean_squared_error(y_test,y_pred)  )

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

'''