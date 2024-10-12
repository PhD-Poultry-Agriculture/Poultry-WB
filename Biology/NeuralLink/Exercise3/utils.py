import struct
import numpy as np


def loadMNISTImages(path):
	with open(path,'rb') as gz:
		n = struct.unpack('I', gz.read(4))
		# Read magic number.
		if n[0] != 0x3080000:
			raise Exception('Invalid file: unexpected magic number.')
		# Read number of entries.
		n = struct.unpack('>I', gz.read(4))[0]
		
		crow = struct.unpack('>I', gz.read(4))[0]
		ccol = struct.unpack('>I', gz.read(4))[0]
		if crow != 28 or ccol != 28:
			raise Exception('Invalid file: expected 28 rows/cols per image.')
		# Read data.
		res = np.fromstring(gz.read(n * crow * ccol), dtype=np.uint8)
	
	res = res.reshape((n, crow, ccol))
	res = np.array(res, dtype=np.float32)
	res = res/255
	return res


def loadMNISTLabels(path):
	with open(path,'rb') as gz:
		n = struct.unpack('I', gz.read(4))
		# Read magic number.
		if n[0] != 0x1080000:
			raise Exception('Invalid file: unexpected magic number.')
		# Read number of entries.
		n = struct.unpack('>I', gz.read(4))[0]
		# Read labels.
		res = np.fromstring(gz.read(n), dtype=np.uint8)

	return res


def preprocess(images):
	'''
	reshape and substact mean.
	Input: images - n_images,rows,cols
	'''
	images = images.reshape(np.size(images, 0),  np.size(images, 1) * np.size(images, 2))
	images = images / np.expand_dims(np.max(images, axis=1), axis=1)
	processed = images - np.expand_dims(np.mean(images, axis=1), axis=1)
	return processed
