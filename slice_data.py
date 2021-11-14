#!/usr/bin/env python
# -*- coding: utf-8 -*-

import scipy.io
import numpy as np
import sys

#trials = int(sys.argv[1])

classes = ['left', 'right', 'foot']

x_train = scipy.io.loadmat('./data/training/x_train.mat')['data']
y_train = scipy.io.loadmat('./data/training/y_train.mat')['data'][0]

class_1 = []
class_2 = []
class_3 = []
y_train_parted = []
x_train_parted = []

for i in range(len(y_train)):
	if y_train[i] == 0:
		class_1.append(x_train[i])
	elif y_train[i] == 1:
		class_2.append(x_train[i])
	else:
		class_3.append(x_train[i])

limit = len(class_1)

if len(class_2) < limit:
	limit = len(class_2)

if len(class_3) < limit:
	limit = len(class_3)

trials_1 = {"data": class_1[:limit], "label": "left"}
trials_2 = {"data": class_2[:limit], "label": "right"}
trials_3 = {"data": class_3[:limit], "label": "foot"}

# for j in range(trials):
# 	y_train_parted.append(0)
# 	x_train_parted.append(class_1[j])
# 	y_train_parted.append(1)
# 	x_train_parted.append(class_2[j])
# 	y_train_parted.append(2)
# 	x_train_parted.append(class_3[j])
# 
# x_data = {"data": x_train_parted, "label": "trials_calib"}
# y_data = {"data": y_train_parted, "label": "classes_calib"}

#scipy.io.savemat('./data/training/x_train_parted.mat', x_data)
#scipy.io.savemat('./data/training/y_train_parted.mat', y_data)

scipy.io.savemat('./data/training/class_1.mat', trials_1)
scipy.io.savemat('./data/training/class_2.mat', trials_2)
scipy.io.savemat('./data/training/class_3.mat', trials_3)
print('slice data ready')
