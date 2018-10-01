import os
import sys
import math
import numpy as np 
import pandas as pd 
from PIL import Image

import utils



class TrainLoader():
	def __init__(self, train_manifest, val_manifest, batch_size, shuffle=True):
		self.train_df = pd.read_csv(train_manifest, header=None)
		self.val_df = pd.read_csv(val_manifest, header=None)
		self.num_train_sample = len(self.train_df)
		self.num_val_sample = len(self.val_df)
		self.batch_size = batch_size
		self.shuffle = shuffle

	def fetchData(data_frame, batch_cache):
		df = self.data_frame.iloc(batch_cache).reset_index(drop=True)
		images = []
		for img_file in df[0].values:
			img = np.asarray(utils.readImg(img_file))
			if len(img.shape) == 2:
				img.resize(img.shape[0], img.shape[1], 1)
			images.append(img)
		labels = df[1].values 
		return images, labels




class TestLoader():
	def __init__(self, test_manifest, batch_size, shuffle=True):
		self.test_df = pd.read_csv(test_manifest, header=None)
		self.test_len = len(self.train_df)
		self.batch_size = batch_size
		self.shuffle = shuffle



	
