#!/usr/bin/env python
# -*- coding: utf-8 -*-

import scipy.io
import numpy as np

classes = ['left', 'right', 'foot']

x_train = scipy.io.loadmat('./data/training/x_train.mat')['data']
y_train = scipy.io.loadmat('./data/training/y_train.mat')['data'][0]

class_1 = []
class_2 = []
class_3 = []

for i in range(len(y_train)):
	if y_train[i] == 0:
		class_1.append(x_train[i])
	elif y_train[i] == 1:
		class_2.append(x_train[i])
	else:
		class_3.append(x_train[i])

trials_1 = {"data": class_1[:1], "label": "left"}
trials_2 = {"data": class_2[:1], "label": "right"}
trials_3 = {"data": class_3[:1], "label": "foot"}

scipy.io.savemat('./data/training/class_1.mat', trials_1)
scipy.io.savemat('./data/training/class_2.mat', trials_2)
scipy.io.savemat('./data/training/class_3.mat', trials_3)
print('slice data ready')
