import numpy as np
import math
from sklearn.model_selection import train_test_split

def read_data(file_name):
	''' reads the data and captures all the metadata 
		: In this case the metadata is the maximum sequence length'''

	with open(file_name, 'r') as data_file:
		data = np.load(data_file)
		max_len = max([len(sequence) for sequence in data])
	return max_len

def generate_dataset_uncond(strokes_file_name, seq_len, val_split = 0.1, random_seed = 1):
	''' reads and genertes the dataset from the file '''

	with open(strokes_file_name, 'r') as strokes_data_file:
		strokes_data = np.load(strokes_data_file)
		chopped_strokes_data = []

		for idx in range(strokes_data.shape[0]):
			num_chunks = math.ceil( 1. * strokes_data[idx].shape[0] / seq_len)
			chopped_strokes_data.extend(np.array_split(strokes_data[idx], num_chunks))

	
	input_data, target_data = zip(*[ (x[:-1], x[1:]) for x in chopped_strokes_data])
	train_input_data, valid_input_data, train_target_data, valid_target_data = \
 		train_test_split(input_data, target_data, test_size = val_split, random_state = random_seed)

 	return train_input_data, valid_input_data, train_target_data, valid_target_data
