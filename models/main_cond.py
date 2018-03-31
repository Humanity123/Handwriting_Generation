import sys
sys.path.insert(0,  '../utils')

import tensorflow as tf
import numpy as np
import data 
import pprint

from __init__ import *
from model_cond import *


pp = pprint.PrettyPrinter()

flags = tf.app.flags

flags.DEFINE_string("strokes_train_data", "../data/strokes.npy", "train data set path")
flags.DEFINE_string("sentence_train_data", "../data/sentences.txt", "train data set path")
flags.DEFINE_string("saved_model_directory", "../", "path to directory containing saved models")
flags.DEFINE_integer("LSTM_layers", 3, "number of LSTM layers")
flags.DEFINE_integer("mixture_components", 20, "number of components in the mixture model")
flags.DEFINE_integer("window_mixture_components", 10, "number of components in the mixture model")
flags.DEFINE_integer("LSTM_outdim", 400, "output dimension of LSTM")
flags.DEFINE_integer("batch_size", 50, "batch_size")
flags.DEFINE_integer("epochs", 30, "number of epochs")
flags.DEFINE_integer("seq_len", 500, "sequence_length")
flags.DEFINE_integer("sen_len", 25, "sentence_length")
flags.DEFINE_integer("char_dim", 64, "char_dimension")
flags.DEFINE_float("lr", 1e-3, "learning_rate")
flags.DEFINE_float("decay", 0.95, "learning_rate")
flags.DEFINE_float("momentum", 0.9, "learning_rate")
flags.DEFINE_float("grad_clip", 10, "gradient_clipping")
flags.DEFINE_float("eps", 1e-20, "epsillon")
flags.DEFINE_float("bias", 0.5, "probability_bias")
flags.DEFINE_float("RNN_outkeep_prob", 0.8, "RNN_outkeep_prob")
flags.DEFINE_float("data_scale", 10, "scale down for data points")


FLAGS = flags.FLAGS


small  = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
caps   = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
digits = ['1','2','3','4','5','6','7','8','9','0']
symbols = [' ']
chars  = small + caps + digits + symbols
char_dict = dict(zip(chars, range(1, len(chars)+1)))

def get_1_hot_data(sentence, char_dict):
	def get_1_hot(char):
			''' converts the char into 1 hot encoding '''
			vector = np.zeros([len(char_dict)+1])
			if char in char_dict:
				vector[char_dict[char]] = 1
			return vector

	return np.array([get_1_hot(char) for char in sentence])

def train_and_save_model():
	''' function to train and save the model '''
	train_input_data, valid_input_data, train_target_data, valid_target_data, train_sentence_data, valid_sentence_data\
		= data.generate_dataset_cond(FLAGS.strokes_train_data, FLAGS.sentence_train_data, FLAGS.seq_len, FLAGS.sen_len)
	print "Data Loaded!"

	pp.pprint(flags.FLAGS.__flags)

	train_sentence_vector_data = [get_1_hot_data(sentence, char_dict) for sentence in train_sentence_data]
	valid_sentence_vector_data = [get_1_hot_data(sentence, char_dict) for sentence in valid_sentence_data]

	with tf.Session() as sess:
		model = SynNet(FLAGS, sess,  training = True)
		model.build_model()
		saver = tf.train.Saver()
		model.train(train_input_data, train_target_data, train_sentence_vector_data, saver) 


def sample(sentence):
	''' function to sample from the model '''
	
	sentence_vector = get_1_hot_data(sentence, char_dict)

	with tf.Session() as sess:
		FLAGS.sen_len = len(sentence)
		model = SynNet(FLAGS, sess,  training = False)
		model.build_model()
		saver = tf.train.Saver(tf.trainable_variables())
		ckpt = tf.train.get_checkpoint_state(FLAGS.saved_model_directory)
		print ckpt.model_checkpoint_path
		saver.restore(sess, ckpt.model_checkpoint_path)
		plot_stroke(model.generate(600, sentence_vector))

if __name__ == '__main__':
	train_and_save_model()
	# sample()