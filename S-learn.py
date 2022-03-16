# Libraries
import numpy as np


# Fully connected layer
class Layer_fc:
	# Init weights and biases, will be trained later
	def __init__(self, n_inputs, n_neurons):
		self.weights = np.random.randn(n_inputs, n_neurons)
		self.biases = np.zeros((1, n_neurons))
		
	# Forward pass of neurons, dot product of inputs.weights + biases
	def forward(self, inputs):
		self.output = np.dot(inputs, self.weights) + self.biases


# Rectified Linear Activation Function
# Fast way to have a non linear activation function
# Commonly used in hidden layers
class ReLU:
	
	def forward(self, inputs):
		# Linear function clipped under 0
		self.output = np.maximum(0, inputs)
