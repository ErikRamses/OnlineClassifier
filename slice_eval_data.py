#!/usr/bin/env python
# -*- coding: utf-8 -*-

import scipy.io
import numpy as np

def split_list(alist, wanted_parts=1):
    length = len(alist)
    return [ alist[i*length // wanted_parts: (i+1)*length // wanted_parts] 
             for i in range(wanted_parts) ]

x_test = scipy.io.loadmat('./data/training/x_test.mat')['data']

eval_data = []

for i in range(len(x_test)):
	trial_1 = []
	trial_2 = []
	for channel in x_test[i]:
		current_list = split_list(channel, wanted_parts=2)
		trial_1.append(current_list[0])
		trial_2.append(current_list[1])

	eval_data.append(trial_1)
	eval_data.append(trial_2)

print(len(eval_data[0][0]))

trials_eval = {"data": eval_data, "label": "eval_data"}

scipy.io.savemat('./data/training/eval_x_test.mat', trials_eval)
print('slice data ready')
