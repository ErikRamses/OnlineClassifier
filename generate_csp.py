#!/usr/bin/env python
# -*- coding: utf-8 -*-

import scipy.io
import numpy as np
from scipy.signal import freqz
import scipy.linalg as la

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

class_1 = scipy.io.loadmat('./data/training/class_1.mat')['data']
class_2 = scipy.io.loadmat('./data/training/class_2.mat')['data']
class_3 = scipy.io.loadmat('./data/training/class_3.mat')['data']

print(len(class_1), len(class_2), len(class_3))

# Passing tasks to CSP method
csp = CSP(class_1, class_2, class_3)
#csp_1 = CSP(class_1, class_2)
#csp_2 = CSP(class_1, class_3)
#csp_3 = CSP(class_2, class_3)

csp_matrix = {"data": csp, "label": "left_right_foot"}

#csp_matrix1 = {"data": csp_1, "label": "left_right"}
#csp_matrix2 = {"data": csp_2, "label": "left_foot"}
#csp_matrix3 = {"data": csp_3, "label": "right_foot"}

scipy.io.savemat('./data/training/csp-1600.mat', csp_matrix)
 
#scipy.io.savemat('./data/training/csp-left-right.mat', csp_matrix1)
#scipy.io.savemat('./data/training/csp-left-foot.mat', csp_matrix2)
#scipy.io.savemat('./data/training/csp-right-foot.mat', csp_matrix3)

print("Csp ready")
