import scipy.io.arff as arff
import math
import MST as mst
import utility
import sys

def tan_learning(trainFile, testFile):
	# Handle file not found exception

	# Loading arff files and instantiating metadata and data classes
	dataset = arff.loadarff(open(trainFile))
	testset, test_meta = arff.loadarff(open(testFile))
	data, meta = dataset
	info = utility.metaData(meta)
	data_obj_train = utility.dataSet(data)

	#Obtain the count for each class
	data_obj_train.class_values = utility.get_class_values(data)

	# Obtain the conditional probability for every attribute per class value
	feature_prob_values, data_obj_train.class_prob_values = utility.train_for_conditional_prob(data, info)

	# Obtain the counts for every feature vs every other feature per classvalue
	feature_counts = get_feature_counts_for_all(data, info)

	# Generating the mutual information gain
	mutual_info_gain = get_mutual_information(feature_counts, feature_prob_values, data_obj_train, info)

	#Obtain the feature counts for every feature per classvalue
	feature_count_for_class = utility.get_feature_count(data, info)

	# Obtain the MST given mutual information gain
	adj_matrix, edges = mst.maximum_spanning_tree(mutual_info_gain)

	# Function to get the CPT for the given tree structure
	conditional_probabilities = get_conditional_probabilities(data_obj_train, info, edges, feature_counts,feature_count_for_class)

	# Test the dataset
	output = test_dataset(testset, conditional_probabilities, data_obj_train, info, edges)

	#Display output as per requirement
	display_output_tan(output, info, edges)	

def get_feature_counts_for_all(data, info):

	# For each value of class attribute
	feature_counts = {}
	for class_value in info.attribute_discrete_values[-1]:
		feature_count_per_class = {}
		for row in data:
			for i in range(len(row)-2): #Excluding the class attribute
				j = i +1
				while j < len(row)-1:
					if row[-1] == class_value: # If its a match to the class value
						key = info.attribute_names[i]+'_'+str(row[i])+'_'+info.attribute_names[j]+'_'+str(row[j])
						if feature_count_per_class.has_key(key):
							feature_count_per_class[key] = feature_count_per_class.get(key)+1
						else:
							feature_count_per_class[key] = 1
					j = j +1
		feature_counts[class_value] = feature_count_per_class

	return feature_counts

def get_mutual_information(feature_counts, feature_prob_values, data_obj_train, info):

	# get ready for a lottt of for loops
	length = len(info.attribute_names)-1 
	mutual_info_gain = [[0 for x in range(length)] for y in range(length)]

	for i in range(length): # Excluding the class value
		for j in range(length):
			info_gain = 0
			# Calculate the information gain for each attribute 
			if i == j:
				mutual_info_gain[i][j] = -1.0
			elif i > j:
				mutual_info_gain[i][j] = mutual_info_gain[j][i]
			else:
				for class_value in data_obj_train.class_values.keys():
					#At this point we need to calcualate the information gain
					for x in info.attribute_discrete_values[i]:
						for y in info.attribute_discrete_values[j]:
							key = info.attribute_names[i]+'_'+x +'_'+info.attribute_names[j]+'_'+y
							if not feature_counts.get(class_value).has_key(key):
								count = 0
							else:
								count = feature_counts.get(class_value).get(key)

							log_numerator = float((count + 1.0)/ (data_obj_train.class_values.get(class_value) + (len(info.attribute_discrete_values[i]) * len(info.attribute_discrete_values[j]))))
							log_den = feature_prob_values.get(class_value)[i].get(x) * feature_prob_values.get(class_value)[j].get(y)

							log_value = math.log((log_numerator/log_den), 2)

							p_xi_xi_y = (count +1.0)/ (data_obj_train.data_count + (len(info.attribute_discrete_values[i]) * len(info.attribute_discrete_values[j]) * len(data_obj_train.class_values)))

							info_gain = info_gain + (p_xi_xi_y * log_value) 
				mutual_info_gain[i][j] = info_gain
	return mutual_info_gain

def get_parent(attribute,edges, info):

	if attribute == 0: # The root and the only parent is the class
		return len(info.attribute_names)-1
	for edge in edges:
		if edge[1] == attribute:
			return edge[0]

def get_conditional_probabilities(data_obj_train, info, edges, feature_counts, feature_count_for_class):

	conditional_probabilities = {}

	for class_value in data_obj_train.class_values:
		cond_prob_list = []
		for attribute in range(len(info.attribute_names)-1):
			parent = get_parent(attribute, edges, info)
			#print parent
			cond_prob = 0
			prob_for_parent = {}
			for p in info.attribute_discrete_values[parent]:
				prob_for_each_feature = {}
				for feature in info.attribute_discrete_values[attribute]:
						# Handle special case for root, ie. the only parent is class
					if parent == len(info.attribute_names)-1:
						numerator = feature_count_for_class.get(class_value)[attribute].get(feature) + 1.0
						denominator = data_obj_train.class_values.get(class_value) + len(info.attribute_discrete_values[attribute])

					else:
						if attribute < parent:
							key = info.attribute_names[attribute] +'_'+ feature+'_'+info.attribute_names[parent]+'_'+p
						else:
							key = info.attribute_names[parent]+'_'+p+'_'+info.attribute_names[attribute]+'_'+feature

						if feature_counts.get(class_value).has_key(key):
							count = feature_counts.get(class_value).get(key)
						else:
							count = 0

						numerator = count +1.0
						denominator = feature_count_for_class.get(class_value)[parent].get(p) + len(info.attribute_discrete_values[attribute])

					cond_prob = numerator/ denominator
					prob_for_each_feature[feature] = cond_prob
				prob_for_parent[p] = prob_for_each_feature
			cond_prob_list.append(prob_for_parent)
		conditional_probabilities[class_value] = cond_prob_list

	return conditional_probabilities

def test_dataset(testset, conditional_probabilities, data_obj_train, info, edges):

	output = []

	for data in testset:
		maximum = 0
		sum_of_probabilitites = 0
		for class_value in data_obj_train.class_prob_values:
			prediction = data_obj_train.class_prob_values.get(class_value)
			# For every attribute in the data
			for value in range(len(data)-1):
				feature = data[value]
				parent = get_parent(value, edges, info)
				# For every possible combination in its parent
				if parent == len(info.attribute_names)-1:
					# Then the root is just the class attribute
					prediction = prediction * conditional_probabilities.get(class_value)[value].get(class_value).get(feature)

				else:
					parent_value = data[parent]
					prediction = prediction * conditional_probabilities.get(class_value)[value].get(parent_value).get(feature)
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

def display_output_tan(output, meta_info, edges):
	# First display all the attributes and their parent class
	for attribute in range(len(meta_info.attribute_names)-1):
		parent = get_parent(attribute, edges, meta_info)

		#For root
		if parent == len(meta_info.attribute_names)-1:
			sys.stdout.write(meta_info.attribute_names[attribute] + ' ' + meta_info.attribute_names[parent]+' \n')
		else:
			sys.stdout.write(meta_info.attribute_names[attribute] + ' ' + meta_info.attribute_names[parent]+ ' ' +meta_info.attribute_names[-1]+' \n') # In NB the parent is always the label attribute

	sys.stdout.write('\n')

	correctly_classified = 0
	for row in output:
		sys.stdout.write(row.get('predicted_class') + ' '+ row.get('actual_class') + ' '+ str(row.get('probability'))+'\n')
		if row.get('correct') == 1:
			correctly_classified = correctly_classified +1

	sys.stdout.write('\n'+str(correctly_classified)+"\n")


	

	
