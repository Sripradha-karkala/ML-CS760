class metaData:

	def __init__(self, meta_data):
		#self.relation_name = meta_data.rel - figure out how to get this
		self.attribute_names = meta_data.names()
		self.attribute_discrete_values = []
		for attribute in self.attribute_names:
			self.attribute_discrete_values.append(meta_data[attribute][-1])

class dataSet:

	def __init__(self, data):
		self.data = data
		self.class_values = {}
		self.class_prob_values={}
		self.data_count = len(data)

def get_class_values(data):
	class_values={}
	for row in data:
		if class_values.has_key(row[-1]):
			class_values[row[-1]] = class_values[row[-1]] +1
		else:
			class_values[row[-1]] = 1
	return class_values

def get_class_probabilities(class_values, example_count):
	class_prob_values={}

	for key in class_values.keys():
		# Find probability using laplace estimates
		class_prob_values[key] = (class_values.get(key) + 1) / (float(example_count) + len(class_values))

	#print len(class_values)
	return class_prob_values

def get_feature_probabilities(class_values, info, feature_count_for_class):

	for class_value in feature_count_for_class.keys():
		class_count = class_values.get(class_value)
		#print class_count
		attribute_count = 0
		for attributes in feature_count_for_class.get(class_value):
			attribute_len = len(info.attribute_discrete_values[attribute_count])
			for attribute in attributes.keys():
				attributes[attribute] = (attributes.get(attribute)+1)/ (float(class_count) + attribute_len)
			attribute_count = attribute_count+1

	return feature_count_for_class
def get_feature_count(data, info):
	example_count = len(data)
	attribute_count = len(data[0])

	# Get the discrete class values

	class_values = get_class_values(data)
	#print class_values

	#Getting the count of every feature for every class attribute
	feature_count_for_class={}
	for class_value in class_values.keys():
		feature_count_list=[]
		for attribute in range(attribute_count):
			feature_count = {}
			for feature in range(example_count):
				value = data[feature][attribute]
				class_name = data[feature][-1]
				if class_name == class_value:
					if feature_count.has_key(value):
						feature_count[value] = feature_count.get(value) +1
					else:
						feature_count[value] = 1
			# For features for which there is no data
			for feature in info.attribute_discrete_values[attribute]:
				if not feature_count.has_key(feature):
					feature_count[feature] = 0
			feature_count_list.append(feature_count)
		#Delete the last entry in the list as it is for the class value
		feature_count_list.pop()
		feature_count_for_class[class_value] = feature_count_list
	#print feature_count_for_class
	return feature_count_for_class	


def train_for_conditional_prob(data, info):
	example_count = len(data)
	attribute_count = len(data[0])

	# Get the discrete class values

	class_values = get_class_values(data)

	feature_count_for_class = get_feature_count(data,info)

	# Probability of class values
	class_prob_values = get_class_probabilities(class_values, example_count)
	#print class_prob_values

	feature_prob_values = get_feature_probabilities(class_values, info, feature_count_for_class)
	#print feature_prob_values
	return feature_prob_values, class_prob_values