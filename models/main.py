import sys
sys.path.insert(0,  '../utils')

import tensorflow as tf
import numpy as np
import data 
import pprint

from __init__ import *
from model import *


pp = pprint.PrettyPrinter()

flags = tf.app.flags

flags.DEFINE_string("strokes_train_data", "../data/strokes.npy", "train data set path")
flags.DEFINE_string("sentence_train_data", "../data/sentences.txt", "train data set path")
flags.DEFINE_string("saved_model_directory", "../saved_model", "path to directory containing saved models")
flags.DEFINE_integer("LSTM_layers", 3, "number of LSTM layers")
flags.DEFINE_integer("mixture_components", 20, "number of components in the mixture model")
flags.DEFINE_integer("LSTM_outdim", 122, "output dimension of LSTM")
flags.DEFINE_integer("batch_size", 200, "batch_size")
flags.DEFINE_integer("epochs", 10, "number of epochs")
flags.DEFINE_integer("seq_len", 400, "sequence_length")
flags.DEFINE_float("lr", 0.1, "learning_rate")
flags.DEFINE_float("grad_clip", 100, "gradient_clipping")
flags.DEFINE_float("eps", 1e-5, "epsillon")
flags.DEFINE_float("bias", 0.5, "probability_bias")


FLAGS = flags.FLAGS

def train_and_save_model():
	''' function to train and save the model '''
	train_input_data, valid_input_data, train_target_data, valid_target_data\
		= data.generate_dataset_uncond(FLAGS.strokes_train_data, FLAGS.seq_len)
	print "Data Loaded!"

	pp.pprint(flags.FLAGS.__flags)

	with tf.Session() as sess:
		model = MixtureDensityNetwork(FLAGS, sess,  training = True)
    	model.build_model()
    	saver = tf.train.Saver()
    	model.train(train_input_data, train_target_data, saver) 
    	

def sample():
	''' function to sample from the model '''
	
	with tf.Session() as sess:
		model = MixtureDensityNetwork(FLAGS, sess,  training = False)
    	model.build_model()
    	saver = tf.train.Saver()
    	ckpt = tf.train.get_checkpoint_state(FLAGS.saved_model_directory)
    	saver.restore(sess, ckpt.model_checkpoint_path)
    	plot_stroke(model.synthesize(200))

if __name__ == '__main__':
	train_and_save_model()
	sample()