import sys
sys.path.insert(0,  '../utils')

import tensorflow as tf
from tensorflow.python import debug as tf_debug
import numpy as np
import data 
import pprint

from __init__ import *
from model import *


pp = pprint.PrettyPrinter()

flags = tf.app.flags

flags.DEFINE_string("strokes_train_data", "../data/strokes.npy", "train data set path")
flags.DEFINE_string("sentence_train_data", "../data/sentences.txt", "train data set path")
flags.DEFINE_string("saved_model_directory", "../", "path to directory containing saved models")
flags.DEFINE_integer("LSTM_layers", 3, "number of LSTM layers")
flags.DEFINE_integer("mixture_components", 20, "number of components in the mixture model")
flags.DEFINE_integer("LSTM_outdim", 400, "output dimension of LSTM")
flags.DEFINE_integer("batch_size", 50, "batch_size")
flags.DEFINE_integer("epochs", 30, "number of epochs")
flags.DEFINE_integer("seq_len", 400, "sequence_length")
flags.DEFINE_float("lr", 1e-2, "learning_rate")
flags.DEFINE_float("decay", 0.95, "learning_rate")
flags.DEFINE_float("momentum", 0.9, "learning_rate")
flags.DEFINE_float("grad_clip", 10, "gradient_clipping")
flags.DEFINE_float("data_scale", 10, "scale down for data points")
flags.DEFINE_float("eps", 1e-10, "epsillon")
flags.DEFINE_float("bias", 0.5, "probability_bias")
flags.DEFINE_float("RNN_outkeep_prob", 0.8, "RNN_outkeep_prob")


FLAGS = flags.FLAGS

def train_and_save_model():
	''' function to train and save the model '''
	mean, std = data.get_data_stats(FLAGS.strokes_train_data)
	train_input_data, valid_input_data, train_target_data, valid_target_data\
		= data.generate_dataset_uncond(FLAGS.strokes_train_data, FLAGS.seq_len)
	print "Data Loaded!"

	pp.pprint(flags.FLAGS.__flags)

	train_input_data, valid_input_data, train_target_data, valid_target_data = \
									([(stroke - mean) / std for stroke in data]  
									for data in (train_input_data, valid_input_data, train_target_data, valid_target_data))

	with tf.Session() as sess:
		debug_sess = sess
		model = MixtureDensityNetwork(FLAGS, debug_sess,  training = True)
    	model.build_model()
    	saver = tf.train.Saver()
    	return model.train(train_input_data, train_target_data, saver) 
    	

def sample():
	''' function to sample from the model '''
	mean, std = data.get_data_stats(FLAGS.strokes_train_data)

	with tf.Session() as sess:
		model = MixtureDensityNetwork(FLAGS, sess,  training = False)
    	model.build_model()
    	saver = tf.train.Saver(sess.trainable_variables())
    	ckpt = tf.train.get_checkpoint_state(FLAGS.saved_model_directory)
    	print ckpt.model_checkpoint_path
    	saver.restore(sess, ckpt.model_checkpoint_path)
    	strokes = (model.synthesize(600) * std) + mean

    	plot_stroke(strokes)

if __name__ == '__main__':
	train_and_save_model()
	# sample()
