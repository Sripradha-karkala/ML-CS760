import scipy.io.arff as arff
import sys
import math
import random
import numpy as np

'''This class captures all the useful information of the dataset
'''
class DataSet:

	def __init__(self, data, metadata):

		self.data = data
		self.n_instance = len(data)
		self.no_of_features = len(data[0])-1 # Excluding the class
		self.attribute_names = metadata.names()
		self.attribute_values=[]
		self.attribute_types=[]
		
		for attribute in self.attribute_names:
			self.attribute_values.append(metadata[attribute][-1])
			self.attribute_types.append(metadata[attribute][0])

		# The last attribute is always the class
		self.output_class = self.attribute_values[-1] 
		self.output_class_count = {}

		for row in self.data:
			if self.output_class_count.has_key(row[-1]):
				self.output_class_count[row[-1]] = self.output_class_count.get(row[-1])+1
			else:
				self.output_class_count[row[-1]] = 1
'''
Function to calculate the accuracy of the neural network
Input:
======
Take the output of the neural network in the format 
fold_num predicted_class actual_class confidence_of_prediction

Outputs:
=======
The accuracy of the prediction
'''
def get_accuracy(output):
	correct = 0
	for row in output:
		if row[1] == row[2]:
			correct = correct +1
	total = len(output)
	return (correct/float(total)) * 100.0

'''
Generates stratified samples 
of num_folds from the given dataset

Input
========
Takes the input dataset and the num_folds required for cross validation

Output
=======
Dataset with a fold_instance attached to each instance.
'''
def generate_stratified_samples(dataset, num_folds):

	# Initialize a list containing the sample size for each class for n-folds
	total_samples = dataset.n_instance
	each_class_sample_size = {}

	for class_value in dataset.output_class:
		value = int(math.ceil(dataset.output_class_count.get(class_value) / float(num_folds)))
		last_value = dataset.output_class_count.get(class_value) - (value * (num_folds-1))
		each_class_sample_size[class_value] = [value]*(num_folds-1)
		each_class_sample_size[class_value].append(last_value)

	sampled_data = []

	for row in dataset.data:
		class_value = row[-1]
		fold_no = False

		while not fold_no:
			rand = random.randint(1, num_folds)
			if each_class_sample_size.get(class_value)[rand-1] > 0:
				temp = list(row)
				temp.append(rand)
				sampled_data.append(temp)

				each_class_sample_size[class_value][rand-1] = (each_class_sample_size.get(class_value)[rand-1]) - 1
				fold_no = True

	return sampled_data

'''
This function basically takes the dataset, separates out the training examples
and returns the training examples in a randomized fashion
'''
def get_randomized_data(sampled_data, test_set):
	#Get the entire dataset, except the test set
	temp = []
	for row in sampled_data:
		if row[-1] != test_set:
			temp.append(row)

	randomized_set = [None] * len(temp)
	count = 0
	#Randomize and add to the list, until the temp list is empty
	while count < len(temp):
		rand = random.randint(0,len(temp)-1)
		if randomized_set[rand] == None:
			randomized_set[rand] =temp[count]
			count = count +1
			
	return randomized_set		

'''
Given a stratified dataset, run it through the neural network
using num_folds cross validation
and accumulate results

Input
=======
Takes in the stratified data, num_folds, num_Epochs and the learning rate

Output
========
Return a list of all the instances with their actual output and predicted output
along with the confidence of prediction
'''
def cross_validation(sampled_data, dataset, num_folds, num_epochs, learning_rate):

	result = [None]* len(sampled_data)

	for num in range(num_folds):
		test_set = num+1
		weights_input, weights_hidden = train_neural_net(sampled_data, dataset, num_epochs, test_set, learning_rate)
		result = test_neural_net(sampled_data, dataset, weights_input, weights_hidden, test_set, result)
	return result

'''
Trains a given dataset for one hidden layer of neural network
using backpropagation and stochastic gradient descent for error correction

Input
========
Takes in the sampled data, num_epochs, learning rate and the indication of the sample set aside for test

Output
=========
List of corrected weights
'''
def train_neural_net(sampled_data, dataset, num_epochs, test_set, learning_rate):

	#Initialize random weights to features and one bias input unit - bias unit weight added in the beginning
	length = dataset.no_of_features+1 
	weights_input = []
	weights_hidden = []
	weight_hidden_bias = random.uniform(0, 0.01)
	
	for f in range(length-1):
		weights = []
		for r in range(length):
			weights.append(random.uniform(0,0.01))

		weights_input.append(weights)
		weights_hidden.append(random.uniform(0,0.01))

	weights_hidden.insert(0, weight_hidden_bias)

	randomize_data = get_randomized_data(sampled_data, test_set)

	# The weights are in the format 
	# --------------------------
	# 0th output - weights from all the inputs with 0th as bias unit
	# 1st output - weights from all the inputs with 0th as bias unit 
	# ...
	#---------------------------------------------

	while num_epochs > 0:
		for row in randomize_data:
			output_hidden = []

			target_class = 1
			# Find the target class
			if row[-2] == dataset.output_class[0]:
				target_class = 0
			#-------------------------------------------------------------------------------------
			#Initially calculate the output for all the units 
			#0th will be the bias unit
			output_hidden.append(1) # The weight for the bias unit will be 1
			for output in range(len(row)-2):
				sum_value = weights_input[output][0]
				for feature in range(len(row)-2):
					#Calculate the outputs for every unit in the hidden layer
					sum_value = sum_value + weights_input[output][feature+1] * row[feature]
				# Calculate the sigmoid function and append to the output for hidden layer
				sigmoid_value = 1/ (1 + math.exp(-sum_value))
				output_hidden.append(sigmoid_value)

			#Calculate the output for the output unit
			sum_value = 0
			for output in range(len(output_hidden)):
				sum_value = sum_value + output_hidden[output] * weights_hidden[output]
			output_node = 1/ (1 + math.exp(-sum_value))

			#-----------------------------------------------------------------------------------
			# Propagating the errors backwards through the networks 
			# Calculate the delta for output unit
			delta_output_node = output_node * ( 1- output_node)*(target_class - output_node)
			
			# Calculate the delta for each hidden unit
			delta_hidden_node = []
			for hidden in range(len(output_hidden)):
				value = output_hidden[hidden] *(1- output_hidden[hidden]) * (delta_output_node * weights_hidden[hidden])
				delta_hidden_node.append(value)

			#Adjust all the weights
			# First adjust the weight for the output unit 
			for output in range(len(output_hidden)):
				delta_weight = learning_rate * delta_output_node * output_hidden[output]
				weights_hidden[output] = weights_hidden[output] + delta_weight

			# Adjust the weight for the hidden layers 
			for hidden_output in range(length-1):
				for input_ in range(length):
					if input_ == 0:
						value = 1
					else:
						value = row[input_-1]
					delta_weight = learning_rate * delta_hidden_node[hidden_output+1] * value

					weights_input[hidden_output][input_] = weights_input[hidden_output][input_] + delta_weight

		num_epochs = num_epochs - 1
	return weights_input, weights_hidden

'''
Function takes the weights generated by the training 
through the neural network and outputs the following

fold_instance predicted_class actual_class confidence_of_prediction
'''
def test_neural_net(sampled_data, dataset, weights_input, weights_hidden, test_set, result):

	length = dataset.no_of_features

	#Calculate the output from each output unit in the hidden layer
	for row in sampled_data:
		if row[-1] == test_set:
			outputs_hidden = []
			outputs_hidden.append(1)
			for hidden_output in range(length):
				sum_value = weights_input[hidden_output][0]
				for i in range(length):
					sum_value = sum_value + weights_input[hidden_output][i+1] * row[i]
				sigmoid_value = 1/(1+ math.exp(-sum_value))
				outputs_hidden.append(sigmoid_value)
			
			#Calculate the final output using the hidden weights
			sum_value = 0
			
			for output in range(len(outputs_hidden)):
				sum_value = sum_value + weights_hidden[output] * outputs_hidden[output]
			sigmoid_value = 1/(1 + math.exp(-sum_value))

			if sigmoid_value > 0.5:
				predicted_class = dataset.output_class[1]
			else:
				predicted_class = dataset.output_class[0]

			# Append the result to the output in the same order as input
			result[sampled_data.index(row)] = [test_set, predicted_class, row[-2], sigmoid_value]
	return result
'''
Function to display output in the correct manner

Input
=====
Takes the list of output from the test_neural network

Output
=====
Prints the output
'''
def display_output(output):
	for o in output:
		print o[0] ,",", o[1],"," ,o[2], ",",o[3]	

''' This program trains and tests a dataset 
	through a neural network with one hidden layer
	and one output. 

	The program generates n-folds stratified samples of input
	and outputs the prediction for each fold

	The neural network is implemented using sigmoid function with backpropagation and 
	stochastic gradient descent for error correction.

	Usage
	===========
	python neuralnetwork trainfile num_folds learning_rate num_epochs

	trainfile: is of arff format, contains numeric values

	num_folds: number of folds for cross validation

	learning_rate: learning rate during error correction

	num_epochs: The number of times the entire training file is passed 
				through the neural network
'''
if __name__=='__main__':
	
	#Add the error checking for the command line arguments
	if not len(sys.argv) == 5:
		print "Usage: neuralnet trainfile num_folds learning_rate num_epochs"
		exit()

	trainfile = sys.argv[1]
	num_folds = int(sys.argv[2])
	learning_rate = float(sys.argv[3])
	num_epochs = int(sys.argv[4])

	#Load the arff file, getting data and meta data
	try:
		data, metadata = arff.loadarff(open(trainfile))
	except ParseArffError:
		print "Trying to load a file which is not in arff format"
		exit()

	dataset = DataSet(data, metadata)

	#Generate the training set and test set from the data based on n-folds
	sampled_data = generate_stratified_samples(dataset, num_folds)

	#Apply the neural network on the training set and obtain weights
	#Apply the weights on the test set and predict output
	output = cross_validation(sampled_data, dataset, num_folds, num_epochs, learning_rate)

	display_output(output)







