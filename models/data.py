import numpy as np

def read_data(file_name):
	''' reads the data and captures all the metadata 
		: In this case the metadata is the maximum sequence length'''

	with open(file_name, 'r') as data_file:
		data = np.load(data_file)
		max_len = max([len(sequence) for sequence in data])
	return max_len

def generate_dataset(file_name):
	''' reads and genertes the dataset from the file '''

	with open(file_name, 'r') as data_file:
		data = np.load(data_file)

	return data

