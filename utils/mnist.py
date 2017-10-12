import os
import gzip
import pickle
import numpy as np
import urllib.request


path = 'data/'
url = 'http://deeplearning.net/data/mnist/'
name = 'mnist.pkl.gz'

def one_hot_vectors(y, classes = 10):
	one_hot_list = []
	for num in y:
		vector = np.zeros_like(range(classes))
		vector[num] = 1
		one_hot_list.append(vector)
	return np.array(one_hot_list)

def dowload(url, name):
	data = urllib.request.urlopen(url + name)
	print('Downloading',name)
	with open(name, 'wb') as f:
		f.write(data.read())

def load_file(path, name):
	return pickle.load(gzip.open(path + name, 'rb'), encoding = 'latin1')
	
def _load(path = path, url = url, name = name):
	if len(os.listdir(path)) != 0:
		print('Loading', name)
		return load_file(path, name)

	else:
		dowload(url, name)
		print('Loading', name)
		return load_file(path, name)

def load():
	mnist_data = _load()
	x_train, y_train = mnist_data[0][0], one_hot_vectors(mnist_data[0][1])
	x_test = np.concatenate([mnist_data[1][0], mnist_data[2][0]])
	y_test = one_hot_vectors(np.concatenate([mnist_data[1][1], mnist_data[2][1]]))
	return (x_train, y_train, x_test, y_test)

