# Division by zero error
# Laplace estimate do we devide it by number of classes existing in the dataset or overall??
# Check if you need to build network for building the paretn -child graph

import scipy.io.arff as arff
import utility
import sys

def bayes_learning(trainFile, testFile):
	dataset = arff.loadarff(open(trainFile))
	testset, test_meta = arff.loadarff(open(testFile))

	data, meta = dataset
	
	# error check for train file and test file having same schema
	info = utility.metaData(meta)

	feature_prob_values, class_prob_values = utility.train_for_conditional_prob(data, info)
	output = test_dataset(testset, feature_prob_values, class_prob_values)

	display_output(output, info)

def test_dataset(testset, feature_prob_values, class_prob_values):

	output = []

	for data in testset:
		maximum = 0
		sum_of_probabilitites = 0
		for class_value in class_prob_values:
			#prediction_values = {}
			prediction = class_prob_values.get(class_value)
			for value in range(len(data)-1):
				feature = data[value]
				prediction = prediction * feature_prob_values.get(class_value)[value].get(feature)

			sum_of_probabilitites = sum_of_probabilitites + prediction
			if(prediction > maximum):
				maximum = prediction
				predicted_class = class_value
			
		# Create a dictionary for output and add the values
		probability = maximum / sum_of_probabilitites

		# Assuming the last column in the test set is class value, generating output for each example
		if(predicted_class == data[-1]):
			correct = 1
		else:
			correct = 0
		output_row = {'predicted_class': predicted_class, 'actual_class': data[-1], 'probability':probability, 'correct': correct}  
		output.append(output_row)

	return output

def display_output(output, meta_info):

	# First display all the attributes and their parent class
	for attribute in range(len(meta_info.attribute_names)-1):
		sys.stdout.write(meta_info.attribute_names[attribute] + " " + meta_info.attribute_names[-1]+' \n') # In NB the parent is always the label attribute

	sys.stdout.write('\n')

	correctly_classified = 0
	for row in output:
		sys.stdout.write(row.get('predicted_class') + " " + row.get('actual_class') + " "+ str(row.get('probability'))+"\n")
		if row.get('correct') == 1:
			correctly_classified = correctly_classified +1

	sys.stdout.write("\n"+str(correctly_classified)+"\n")

	
		
