import numpy as np
from sklearn.model_selection import train_test_split

def read_data(file_name):
	''' reads the data and captures all the metadata 
		: In this case the metadata is the maximum sequence length'''

	with open(file_name, 'r') as data_file:
		data = np.load(data_file)
		max_len = max([len(sequence) for sequence in data])
	return max_len

def generate_dataset(file_name, val_split = 0.1, random_seed = 1):
	''' reads and genertes the dataset from the file '''

	with open(file_name, 'r') as data_file:
		data = np.load(data_file)

	train_data, valid_data =train_test_split(data, test_size = val_split, random_state = random_seed)
	return train_data, valid_data

