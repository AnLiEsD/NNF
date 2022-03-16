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


# An activation function is a way for us to activate or not a neuron
# The rectified linear activation function either return the output of the neuron or 0
# The ReLU activation function is a simple way for us to add non linearity to a function with limited calculation
# The RelU activation function is commonly used in hidden layers

def Activation_ReLU(inputs):
	if inputs < 0:
		return 0
	else:
		return inputs
