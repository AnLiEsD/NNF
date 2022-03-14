# Inputs is the data coming in the neuron
# This data can come from sensors or other neurons
inputs = [0.2, 0.6, 0.3]

# We need to have a batch of weights for each neuron
# The number of weights is linked to the number of inputs
weights = [[0.3, -0.7, 1.0],
	   [0.8, 0.2, 0.4],
	   [-0.1, -0.6, 0.3]]

# The bias is unique to each neuron, we only have one for each
biases = [0.5, -0.2, 3]

# To calculate the output of our neron we need to apply the dot product
# The dot product is applied to the inputs and weights
# The bias is added at the end of the calculation

# Init of the output list
outputs = []
# Each neuron
for neuron_weights, neuron_bias in zip(weights, biases):
	# Init the value of the neuron
	neuron_out = 0
	# Each parameters linked to the neuron
	for input_n, weight in zip(inputs, neuron_weights):
		# Dot product of the neuron
		# Each time we calculate and add the input*weight to neuron_out
		neuron_out += input_n*weight
	# Bias at the end of the dot product
	neuron_out += neuron_bias
	#Output of the neuron in output batch
	outputs.append(neuron_out)

print(outputs)
