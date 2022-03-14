# Libs
# We are going to use numpy for the calculations
# This framework is more optimised and commonly used

import numpy as np

# When we are working with neural network, we use large amount of data
# Using each neuron as a single entity is time consuming
# To solve this problem we use batches of data
# These batches represent the inputs, weights and biases associated to each neurons
# To use these batches we will use matrices and the properties associated


# Inputs is the batch of data coming in the neurons
inputs = [[0.3, 0.8, 1],
	  [1.5, 0.9, 1.2],
	  [-0.7, 1.2, 1.6],
	  [2.1, 1.3, -1.6],
	  [-0.2, 0.3, 0.7]]

# The batch of weights is linked to the number of neurons
weights = [[0.67, -0.09, 1.19],
	  [1.62, 1.47, 1.19],
	  [0.22, 0.43, -0.6],
	  [-0.23, -1.52, 1.55],
	  [-1.06, 0.85, -1.12]]

# Each biases is unique to a neuron
biases = [2, 1.5, -0.7, 0.5, -1.2]


# We now need to compute the dot product of the inputs and weights matrices
# To do so, we will use numpy but we have to be careful with the shapes of our matrices
# The current shapes of our matrices are layer: (5,3) and weights: (5,3)
# We cannot compute the dot product of two matrices with these shapes
# The second dimensions of the first matrix has to be equal to the first one of the second matrix

# Example : matrix_a: (x,y) matrix_b: (a,b)
# y has to be equal to a
# The output of the calculation will have the dimensions (x,b)

# Back to our problem
# We can use matrix transpose, this operation inverts the 2 dimensions of the layer
# matrix_a: (x,y) matrix_a.T: (y,x)

# Knowing that we can now compute the dot product of the weights and biases

output = (np.dot(inputs, np.array(weights).T)) + biases
print(output)
print("=Output shape", np.shape(output))
