import numpy as np

def encode_labels( y, k):
	"""Encode labels into one-hot representation
	"""
	onehot = np.zeros((y.shape[0],k ))
	for idx, val in enumerate(y):
		onehot[idx,val] = 1.0
	return onehot
