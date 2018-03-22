import numpy as np
from sklearn.model_selection import train_test_split

def read_data(file_name):
	''' reads the data and captures all the metadata 
		: In this case the metadata is the maximum sequence length'''

	with open(file_name, 'r') as data_file:
		data = np.load(data_file)
		max_len = max([len(sequence) for sequence in data])
	return max_len

def generate_dataset(strokes_file_name, sentence_file_name, val_split = 0.1, random_seed = 1):
	''' reads and genertes the dataset from the file '''

	with open(strokes_file_name, 'r') as strokes_data_file:
		strokes_data = np.load(strokes_data_file)

	with open(sentence_file_name, 'r') as sentence_data_file:
		sentence_data = sentence_data_file.readlines()

 	train_strokes_data, train_sentence_data, valid_strokes_data, valid_sentence_data = /
 		train_test_split(data, sentence_data, test_size = val_split, random_state = random_seed)
 		
	return train_data, valid_data

