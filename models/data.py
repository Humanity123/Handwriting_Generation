import numpy as np
import math
from sklearn.model_selection import train_test_split


def read_data(strokes_file_name, sentence_file_name):
	''' reads the data and captures all the metadata 
		: In this case the metadata is the maximum sequence length
	'''

	with open(strokes_file_name, 'r') as strokes_data_file:
		strokes_data = np.load(strokes_data_file)
		strokes_max_len = max([len(sequence) for sequence in strokes_data])

	with open(sentence_file_name, 'r') as sentences_data_file:
		sentence_data = [sentence.strip()
						 for sentence in sentences_data_file.readlines()]
		sentence_max_len = max([len(sentence) for sentence in sentence_data])

	return strokes_max_len, sentence_max_len


def get_data_stats(strokes_file_name):
	''' reads the data and returns the means and std for delta x, y,
		ignores the mean and std for eos
	'''

	with open(strokes_file_name, 'r') as strokes_data_file:
		strokes_data = np.load(strokes_data_file)
		all_points = []
		for stroke in strokes_data:
			all_points.extend(stroke)
	mean = np.mean(all_points, axis=0)
	std = np.std(all_points, axis=0)

	mean[0] = 0.
	std[0] = 1.

	return mean, std


def generate_dataset_uncond(strokes_file_name, seq_len, val_split=0.1, random_seed=1):
	''' reads and genertes the dataset from the file for unconditional model

		strokes greater than seq_len are chopped off.
		This is to reduce training time.

		returns the tuple - 
		(train_input_data, valid_input_data, train_target_data, valid_target_data)
	'''

	with open(strokes_file_name, 'r') as strokes_data_file:
		strokes_data = np.load(strokes_data_file)
		chopped_strokes_data = [stroke[:seq_len] for stroke in strokes_data]

	input_data, target_data = zip(*[(x[:-1], x[1:])
									for x in chopped_strokes_data])
	dataset = \
		train_test_split(input_data, target_data,
						 test_size=val_split, random_state=random_seed)

	return dataset


def generate_dataset_cond(strokes_file_name, sentences_file_name, seq_len, sen_len, val_split=0.1, random_seed=1):
	''' reads and generates the dataset from the file for conditional model

		strokes greater than seq_len are chopped off.
		sentences greater than sen_len are chopped off.
		This is to reduce training time.

		returns the tuple - 
		(train_input_data, valid_input_data, train_target_data, valid_target_data, 
		train_sentence_data, valid_sentence_data)
	'''

	with open(strokes_file_name, 'r') as strokes_data_file:
		strokes_data = np.load(strokes_data_file)
		chopped_strokes_data = [stroke[:seq_len] for stroke in strokes_data]

	with open(sentences_file_name, 'r') as sentences_data_file:
		sentence_data = [sentence.strip()
						 for sentence in sentences_data_file.readlines()]
		chopped_sentence_data = [sentence[:sen_len]
								 for sentence in sentence_data]

	input_data, target_data = zip(*[(x[:-1], x[1:])
									for x in chopped_strokes_data])
	dataset = \
		train_test_split(input_data, target_data, chopped_sentence_data,
						 test_size=val_split, random_state=random_seed)

	return dataset
