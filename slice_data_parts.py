#!/usr/bin/env python
# -*- coding: utf-8 -*-

import scipy.io
import numpy as np
import sys

trials = int(sys.argv[1])

def split_list(alist, wanted_parts=1):
    length = len(alist)
    return [ alist[i*length // wanted_parts: (i+1)*length // wanted_parts] 
             for i in range(wanted_parts) ]

classes = ['left', 'right', 'foot']

x_train = scipy.io.loadmat('./data/training/x_train_eval.mat')['data']
y_train = scipy.io.loadmat('./data/training/y_train_eval.mat')['data'][0]

class_1 = []
class_2 = []
class_3 = []
x_train_parted = []
y_train_parted = []

for i in range(len(y_train)):
	trial_1 = []
	trial_2 = []
	trial_3 = []
	trial_4 = []
	trial_5 = []
	trial_6 = []
	trial_7 = []
	trial_8 = []
	for channel in x_train[i]:
		current_list = split_list(channel, wanted_parts=8)
		trial_1.append(current_list[0])
		trial_2.append(current_list[1])
		trial_3.append(current_list[2])
		trial_4.append(current_list[3])
		trial_5.append(current_list[4])
		trial_6.append(current_list[5])
		trial_7.append(current_list[6])
		trial_8.append(current_list[7])
	if y_train[i] == 0:
		class_1.append(trial_1)
		class_1.append(trial_2)
		class_1.append(trial_3)
		class_1.append(trial_4)
		class_1.append(trial_5)
		class_1.append(trial_6)
		class_1.append(trial_7)
		class_1.append(trial_8)
	elif y_train[i] == 1:
		class_2.append(trial_1)
		class_2.append(trial_2)
		class_2.append(trial_3)
		class_2.append(trial_4)
		class_2.append(trial_5)
		class_2.append(trial_6)
		class_2.append(trial_7)
		class_2.append(trial_8)
	else:
		class_3.append(trial_1)
		class_3.append(trial_2)
		class_3.append(trial_3)
		class_3.append(trial_4)
		class_3.append(trial_5)
		class_3.append(trial_6)
		class_3.append(trial_7)
		class_3.append(trial_8)
	x_train_parted.append(trial_1)
	x_train_parted.append(trial_2)
	x_train_parted.append(trial_3)
	x_train_parted.append(trial_4)
	x_train_parted.append(trial_5)
	x_train_parted.append(trial_6)
	x_train_parted.append(trial_7)
	x_train_parted.append(trial_8)
	y_train_parted.append(y_train[i])
	y_train_parted.append(y_train[i])
	y_train_parted.append(y_train[i])
	y_train_parted.append(y_train[i])
	y_train_parted.append(y_train[i])
	y_train_parted.append(y_train[i])
	y_train_parted.append(y_train[i])
	y_train_parted.append(y_train[i])

x_train_2 = {"data": x_train_parted, "label": "x_train_1_sec"}
y_train_2 = {"data": y_train_parted, "label": "y_train_1_sec"}

print(len(x_train_parted), len(y_train_parted))

scipy.io.savemat('./data/training/x_train_eval.mat', x_train_2)
scipy.io.savemat('./data/training/y_train_eval.mat', y_train_2)

trials_1 = {"data": class_1, "label": "left"}
trials_2 = {"data": class_2, "label": "right"}
trials_3 = {"data": class_3, "label": "foot"}

'''

x_train_parted = []
y_train_parted = []

for j in range(trials):
	y_train_parted.append(0)
	x_train_parted.append(class_1[j])
	y_train_parted.append(1)
	x_train_parted.append(class_2[j])
	y_train_parted.append(2)
	x_train_parted.append(class_3[j])

print(len(x_train_parted), len(y_train_parted))

x_data = {"data": x_train_parted, "label": "trials_calib"}
y_data = {"data": y_train_parted, "label": "classes_calib"}

scipy.io.savemat('./data/training/x_train_parted.mat', x_data)
scipy.io.savemat('./data/training/y_train_parted.mat', y_data)

'''

# scipy.io.savemat('./data/training/class_1.mat', trials_1)
# scipy.io.savemat('./data/training/class_2.mat', trials_2)
# scipy.io.savemat('./data/training/class_3.mat', trials_3)

print('slice data ready')
