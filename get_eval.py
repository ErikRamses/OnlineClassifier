#!/usr/bin/env python
# -*- coding: utf-8 -*-

import scipy.io
import numpy as np
from scipy import signal

classes = ['left', 'right', 'foot'] 
selected_channels = ['F3', 'Fz', 'F4', 'C3', 'Cz', 'C4', 'O1', 'Pz', 'O2']
difference = 8000

class EvalSignal(object):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

# Get the minimum difference between intervals
def minDiff(time_markers):
	start = 0
	diffs = []
	for i in range(len(time_markers)):
		diffs.append(time_markers[i]-start)
		start = time_markers[i]
	return min(diffs)

def getLabels(info):
	labels = []
	for clase in info:
		labels.append(clase[0])
	return labels

def getChannels(info):
	channels = []
	indexes = []
	for channel in info:
		channels.append(channel[0])
	for element in selected_channels:
		indexes.append(channels.index(element))
	return indexes

# Exclude last 16 trials and iterate inside time markers
def getXeval(items, channels):
	X_test = []
	
	ini = 0
	parts = int( len(items.data[0]) / 8000 )
	for i in range(1, parts):
		trials = []
		for item in items.data[channels]:
			f = signal.resample(item[ini:i*8000], 1600)
			trials.append(f)
		X_test.append(trials)
	return X_test

def preProcess(eeg_eval):
	# Get labels from clases into signal
	labels = getLabels(eeg_eval['nfo'][0][0][1][0])

	# Get labels from channels into signal
	channels = getChannels(eeg_eval['nfo'][0][0][2][0])

	# Get info from evaluation file
	items = EvalSignal(eeg_eval['cnt'].T, labels)
	test_data = getXeval(items, channels)
	return test_data

#Get all calib trials for autocalibration
#left, foot
eeg_eval_1a = scipy.io.loadmat('./data/BCICIV_1eval_1000Hz_mat/BCICIV_eval_ds1a_1000Hz.mat')
#left, right
eeg_eval_1b = scipy.io.loadmat('./data/BCICIV_1eval_1000Hz_mat/BCICIV_eval_ds1b_1000Hz.mat')
#left, foot
eeg_eval_1f = scipy.io.loadmat('./data/BCICIV_1eval_1000Hz_mat/BCICIV_eval_ds1f_1000Hz.mat')
#left, right
eeg_eval_1g = scipy.io.loadmat('./data/BCICIV_1eval_1000Hz_mat/BCICIV_eval_ds1g_1000Hz.mat')

# get x_test values from eeg continous data
X_test = preProcess(eeg_eval_1a)
x2 = preProcess(eeg_eval_1b)
x3 = preProcess(eeg_eval_1f)
x4 = preProcess(eeg_eval_1g)

X_test = np.concatenate([X_test, x2, x3, x4])

x_data = {"data": X_test, "label": "trials_eval"}

scipy.io.savemat('./data/training/x_test.mat', x_data)
print('eval data ready')
