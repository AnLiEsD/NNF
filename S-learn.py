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


#Common loss, this class will be improved in the future
class Loss():

	def calculate(self, output, valid):
		#Calculation of losses
		sample_losses = self.forward(output, valid)
		#Mean of the losses
		mean_loss = np.mean(sample_losses)
		
		return mean_loss	


# Cross_entropy
# This loss calculation is applied to a probability distribution
# We use the output data from our neurons and the desired output
class Cross_entropy(Loss):
	
	def forward(self, prediction, valid):
		
		#Number of samples
		samples = len(prediction)
		# We clip the data to solve the issue of -log(0) that is impossible
		#We are doing the same for the value 1 to avoid getting a negative result
		prediction_clipped = np.clip(prediction, 1e-10, 1 - 1e-10)
		
		if len(valid.shape) == 1:
			confidence = prediction_clipped[range(samples), valid]
			
		elif len(valid.shape) == 2:
			confidence = np.sum(predicion_clipped * valid, axis=1)
			
		#Losses
		neg_log_loss = -np.log(confidence)
		return neg_log_loss


# Accuracy of our model
# This is another process that we will be unsing to improve our model
def accuracy(ouputs, targets):
	# Calculate the index of the greatest value in each row
	preds = np.argmax(outputs, axis=1)
	
	if len(targets.shape) == 2:
		targets = np.argmax(targets, axis=1)
	
	acc = np.mean(preds == targets)
	return acc