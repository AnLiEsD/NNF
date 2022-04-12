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


# Softmax activation function
# This is a non linear function used for classification
# Data is normalized and the output is used as a confidence score
class Softmax_Activation:

	def forward(self, inputs):
		# The exponentiation of values can cost a lot of ressources
		# By reducing the scale of our data the result will be smaller
		exp_values = np.exp(inputs - np.max(inputs, axis = 1, keepdims = True))
		# Making a probability distribution of the data
		prob = exp_values / np.sum(exp_values, axis = 1, keepdims = True)

		self.output = prob


# Cross_entropy
# This loss calculation is applied to a probability distribution
# We use the output data from our neurons and the desired output
neuron_outputs = np.array[[0.8, 0.1, 0.1],
						 [0.2, 0.7, 0.1],
						 [0.1, 0.3, 0.6]]

# Indexes of the desired values
targets = np.array[[1, 0, 0],
				   [0, 1, 0],
				   [0, 0, 1]]

# No need for computation if the shape is equal to 1
if len(targets.shape) == 1:
	confidence = neuron_outputs[range(len(neuron_outputs)), targets]

elif len(targets.shape) == 2:
	confidence = np.sum(neuron_outputs * targets, axis=1)

# Calculation of cross-entropy
loss = -log(confidence)

avg_loss = np.mean(loss)
print(avg_loss)
