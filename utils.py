import os
import sys
import math
import numpy as np 
from PIL import Image




def readImg(file):
	img = Image.open(file)
	return img


def batchGenerator(num_sample, batch_size, shuffle=True):
	idx = np.arange(0, num_sample, 1)
	if shuffle:
		np.random.shuffle(idx)

	batch_caches = np.array_split(train_idx, 
								math.ceil(num_sample / batch_size))
	return batch_caches