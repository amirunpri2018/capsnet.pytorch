import os
import sys
import math
import numpy as np 
import pandas as pd 
from PIL import Image


class DataLoader():
	def __init__(self, train_manifest, val_manifest, batch_size, shuffle):
		self.train_df = pd.read_csv(train_manifest, header=None)
		self.val_df = pd.read_csv(val_manifest, header=None)
		self.train_len = len(self.train_df)
		self.val_len = len(self.val_df)
		self.batch_size = batch_size
		self.shuffle = shuffle


	def reamImg(file):
		img = Image.open(file)
		return img


	def batchGenerator():
		train_idx = np.arange(0, self.train_len, 1)
		val_idx = np.arange(0, self.val_len, 1)
		if self.shuffle:
			np.random.shuffle(train_idx)

		train_batch_caches = np.array_split(train_idx, 
								math.ceil(self.train_len / self.batch_size))
		val_batch_caches = np.array_split(val_idx, 
								math.ceil(self.val_len / self.batch_size))
		return train_batch_caches, val_batch_caches

	def fetchTrainData(batch_cache):
		df = self.train_df.iloc(batch_cache).reset_index(drop=True)
		images = []
		for img_file in df[0].values:
			img = np.asarray(reamImg(img_file))
			if len(img.shape) == 2:
				img.resize(img.shape[0], img.shape[1], 1)
			images.append(img)
		labels = df[1].values 
		return images, labels

	def fetchValData(batch_cache):
		df = self.val_df.iloc(batch_cache).reset_index(drop=True)
		images = []
		for img_file in df[0].values:
			img = np.asarray(reamImg(img_file))
			if len(img.shape) == 2:
				img.resize(img.shape[0], img.shape[1], 1)
			images.append(img)
		labels = df[1].values 
		return images, labels
