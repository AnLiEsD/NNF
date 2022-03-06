# Neural Network Library
In this repository, I will create my own library for neural networks related tasks.
We will shortly see how a neuron works and what we can do with it.


## Artificial Neural Nerworks
An artificial neural network is a computing system based on the <b>biological brain</b>.
A collection of <b>interconnected neurons</b> can transmit electrical currents to each other, artificial neurons do the same with data. The neuron then <b>processes the signal</b> and decides to have affects other neurons or not.
The <b>strength of the connexions</b> between the neurons is a strong factor in how our brain works. an artificial neuron has weights applied to each input of the neuron to mimic this mechanism. These <b>weights are trained</b> to perform the intended tasks.


## How does an artificial neuron work ?
As previously said, a neuron takes data as inputs and returns data on the output.
The input data is modified by the <b>weights applied to each input</b>. The bias is also a parameter that we can use to train our model, it is commonly used to <b>offset the output</b> value of the neuron.

There is how we could represent an artificial neuron: 

The neuron itself then computes this data, it first calculates the <b>dot product</b> of these numbers and then the output goes through an <b>activation function</b> to decide to fire or not the next neuron.

There is a common representation of a multilayer neural network:


## What are we going to see in this repository ?
We will see how to <b>compute a neuron and layers of neurons</b>, the different ways we calculate outputs <b>either to another neuron</b> (hidden layers) or to the <b>output of our neural network</b>.
We will then see how to <b>train</b> the variables of our neurons to get the desired output.
Along the way, I will explain each <b>concept</b> work and I will make the features as <b>fast</b> as possible.
