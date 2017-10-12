import numpy as np
from utils import mnist
import matplotlib.pyplot as plt

def save_fig(data, title, xlabel = 'epochs', ylabel = 'accuracy', upper_lim = False):
	plt.figure()
	plt.title(title)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.plot(data)
	plt.savefig(title + '.png')

def sigmoid(x, prime = False):
	sg = 1/(1 + np.exp(-x))
	if prime:
		return sg * (1 - sg)
	else:
		return sg

def relu(x, prime = False):
	if prime:
		x[x<0] = 0
		x[x>0] = 1
		return x
	else:
		x[x<0] = 0 
		return x

class NeuralNet:
	def __init__(self, input_len = 784, hidden_neurons = 1024, classes = 10, stddev = 1e-3):
		self.w1 = stddev * np.random.randn(input_len, hidden_neurons)
		self.b1 = np.zeros(hidden_neurons)
		self.w2 = stddev * np.random.randn(hidden_neurons, classes)
		self.b2 = np.zeros(classes)
		self.losses = []
		self.train_acc = []

	def forward(self, x, training = False):
		z1 = np.dot(self.w1.T, x) + self.b1
		h1 = relu(z1)
		z2 = np.dot(self.w2.T, h1) + self.b2
		out = sigmoid(z2)
		if training:
			return (z1, h1, z2, out)
		else:
			return out

	def predict(self, x):
		return np.argmax(self.forward(x))

	def acc(self, x, y):
		pred = [self.predict(x_) for x_ in x]
		cor = 0
		for pred_, y_ in zip(pred,y):
			if pred_ == np.argmax(y_):
				cor += 1
		cor /= len(x)
		return cor

	def fit(self, x, y, lr = 1e-2, epochs = 30, n_samples = 1500, acc = True):
		print('Beginning trainning process...')
		for epoch in range(epochs):
			# Sample from dataset
			idx = np.random.randint(len(x), size = n_samples)
			x_train, y_train = [x[i] for i in idx], [y[i] for i in idx]

			loss = 0
			cor = 0
			for x_, y_ in zip(x_train, y_train):
				# Backprop
				z1, h1, z2, out = self.forward(x_, training = True)
				
				error = out - y_ 
				delta_out = error * sigmoid(z2, prime = True)
				grad_w2, grad_b2 = np.dot(h1.reshape(len(h1),1), delta_out.reshape(len(delta_out),1).T), delta_out
				
				delta_h1 = np.dot(self.w2, delta_out) * relu(z1, prime = True)
				grad_w1, grad_b1 = np.dot(x_.reshape(len(x_),1), delta_h1.reshape(len(delta_h1),1).T), delta_h1


				# Gradient Descent
				self.w2 -= lr * grad_w2
				self.w1 -= lr * grad_w1
				self.b2 -= lr * grad_b2
				self.b1 -= lr * grad_b1	


				if np.argmax(out) == np.argmax(y_):
					cor += 1
				loss += .5 * np.mean((error ** 2))

			cor /= len(x_train)
			self.train_acc.append(cor)
			loss /= len(x_train)
			self.losses.append(loss)
			print('Epoch:', epoch, ' loss:', self.losses[-1], '\tacc:', self.train_acc[-1])		

# Loading MNIST data set
x_train, y_train, x_test, y_test = mnist.load()

net = NeuralNet()
net.fit(x_train, y_train)

# Test
model_acc = net.acc(x_test, y_test)
print("Model's accuracy:", model_acc)


plot = True
if plot:
	save_fig(net.train_acc, 'Train Accuracy')
	save_fig(net.losses, 'Losses', ylabel = 'loss')
