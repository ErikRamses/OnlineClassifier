#!/usr/bin/env python
# -*- coding: utf-8 -*-

import scipy.io
import numpy as np
from scipy import signal

classes = ['left', 'right', 'foot'] 
selected_channels = ['F3', 'Fz', 'F4', 'C3', 'Cz', 'C4', 'O1', 'Pz', 'O2']
difference = 8000

class CalibSignal(object):
    def __init__(self, data, time_markers, true_classes, labels):
        self.data = data
        self.time_markers = time_markers
        self.true_classes = true_classes
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
def getXYTrain(items, channels):
	y_train = []
	x_train = []

	for i in range(1, len(items.time_markers)):
		trials = []
		# Cut signals on regular intervals
		if((items.time_markers[i] - items.time_markers[i-1]) < difference+10):
			for item in items.data[channels]:
				f = signal.resample(item[items.time_markers[i]-difference:items.time_markers[i]], 1600)
				trials.append(f)
			if items.true_classes[i] == 1:
				y_train.append(classes.index(items.labels[1]))
			else:
				y_train.append(classes.index(items.labels[0]))
			x_train.append(trials)
	return x_train, y_train

def preProcess(eeg_calib):
	# Get labels from clases into signal
	labels = getLabels(eeg_calib['nfo'][0][0][1][0])

	# Get labels from channels into signal
	channels = getChannels(eeg_calib['nfo'][0][0][2][0])

	# Get info fro evaluation file
	items = CalibSignal(eeg_calib['cnt'].T, eeg_calib['mrk'][0][0][0][0], eeg_calib['mrk'][0][0][1][0], labels)
	train_data = getXYTrain(items, channels)
	return train_data

#Get all calib trials for autocalibration
#left, foot
eeg_calib_1a = scipy.io.loadmat('./data/BCICIV_1calib_1000Hz_mat/BCICIV_calib_ds1a_1000Hz.mat')
#left, right
eeg_calib_1b = scipy.io.loadmat('./data/BCICIV_1calib_1000Hz_mat/BCICIV_calib_ds1b_1000Hz.mat')
#left, foot
eeg_calib_1f = scipy.io.loadmat('./data/BCICIV_1calib_1000Hz_mat/BCICIV_calib_ds1f_1000Hz.mat')
#left, right
eeg_calib_1g = scipy.io.loadmat('./data/BCICIV_1calib_1000Hz_mat/BCICIV_calib_ds1g_1000Hz.mat')

# get x_train and y_train values from eeg continous data
x_train, y_train = preProcess(eeg_calib_1a)
x2, y2 = preProcess(eeg_calib_1b)
x3, y3 = preProcess(eeg_calib_1f)
x4, y4 = preProcess(eeg_calib_1g)

x_train = np.concatenate([x_train, x2, x3, x4])
y_train = np.concatenate([y_train, y2, y3, y4])

x_data = {"data": x_train, "label": "trials_calib"}
y_data = {"data": y_train, "label": "classes_calib"}

scipy.io.savemat('./data/training/x_train.mat', x_data)
scipy.io.savemat('./data/training/y_train.mat', y_data)
print('calib data ready')







