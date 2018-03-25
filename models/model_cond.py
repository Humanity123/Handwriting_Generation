import tensorflow as tf
import numpy as np
import math 

class SynNet(object):
	''' class for Synthesis Network Architecture '''

	def __init__(self, config, sess, training = True):
		''' initialisation function for class '''

		self.sess = sess
		self.seq_len = config.seq_len
		self.LSTM_layers = config.LSTM_layers
		self.mixture_components = config.mixture_components
		self.windoe_mixture_components = config.windoe_mixture_components
		self.batch_size = config.batch_size
		self.LSTM_outdim = config.LSTM_outdim
		self.init_lr = config.lr
		self.grad_clip = config.grad_clip
		self.epochs = config.epochs
		self.eps = config.eps
		self.bias = config.bias
		self.saved_model_directory = config.saved_model_directory


		if not training:
			self.batch_size = 1
			self.seq_len = 1

		self.mixture_dim = 1 + self.mixture_components * 6
		self.window_dim  = self.windoe_mixture_components * 3


	def build_model(self):
		''' function to build the model '''

		self.global_step = tf.Variable(0, name='global_step', trainable=False)

		def normal_bivariate(x1, x2, mu1, mu2, sigma1, sigma2, rho):
			''' calculates the bivariate normal function from the parameters '''

			cons_pi = tf.constant(math.pi)
			norm_x1 = (x1 - mu1) / sigma1
			norm_x2 = (x2 - mu2) / sigma2
			Z = tf.square(norm_x1) + tf.square(norm_x2) - (2. * rho * (norm_x1 * norm_x2))
			C = 1.0 / (1.0 - tf.square(rho) + self.eps)

			normal_bivariate_prob = (1.0 / (2. * cons_pi * sigma1 * sigma2)) * tf.sqrt(C) * tf.exp((-1 * Z) * C / 2.)
			return normal_bivariate_prob

		def get_mixture_parameters(y, bias = 0.):
			''' computes the mixture parameters from the LSTM output '''

			mc = self.mixture_components

			pi     = tf.nn.softmax(y[:, :, :mc] * (1 + bias))
			mu1    = y[:, :, mc : 2 * mc]
			mu2    = y[:, :, 2* mc : 3 * mc]
			sigma1 = tf.exp(y[:, :, 3 * mc : 4 * mc] - bias) + self.eps
			sigma2 = tf.exp(y[:, :, 4 * mc : 5 * mc] - bias ) + self.eps
			rho    = tf.tanh(y[:, :, 5 * mc : 6 * mc])
			e      = 1. / (1 + tf.exp(y[:, :, 6 * mc :]))

			return pi, mu1, mu2, sigma1, sigma2, rho, e
		
		def get_window_parameters(window_parameters, prev_kappa):
			''' computes the character weights for the text string '''

			wmc = self.window_mixture_components

			alpha = tf.exp(window_parameters[:, :, :wmc])
			beta  = tf.exp(window_parameters[:, :, wmc: 2 * wmc])
			kappa = prev_kappa + tf.exp(window_parameters[:, :, 2 * wmc : 3 * wmc])

		def get_phi(alpha, beta, kappa, u):
			''' computes phi(character weight) for a character in the text string '''

			phi = tf.multiply(alpha, tf.exp(-1 * tf.multiply(beta, kappa - u)))
			reduced_phi = tf.reduce_sum(phi, axis = 2)

			return reduced_phi
			
		def get_loss(normal_bivariate_prob, pi, eos, e, mask):
			''' function to calculate the loss '''

			eos_prob = tf.multiply(eos, e) + tf.multiply((1. - eos), (1. - e))
			eos_prob_2dim = tf.reshape(eos_prob, [self.batch_size, self.seq_len])
			self.check1 = eos_prob_2dim

			pi_prob  = tf.multiply(pi, normal_bivariate_prob)
			self.check2 = pi_prob
			reduced_pi_prob = tf.reduce_sum(pi_prob, axis = 2)
			self.check3 = reduced_pi_prob

		self.input  = tf.placeholder(tf.float32, [None, self.seq_len, 3], name = 'input')
		self.target  = tf.placeholder(tf.float32, [None, self.seq_len, 3], name = 'target')
		self.mask   = tf.placeholder(tf.float32, [None, self.seq_len], name = 'mask')

		self.hidden1_init_state = hidden1_LSTM.zero_state(self.batch_size, tf.float32)
		hidden1_LSTM = tf.contrib.rnn.LSTMCell(self.hidden1_LSTM_outdim, state_is_tuple = True)

		hidden2_LSTMs = []
		for idx in range(self.LSTM_layers):
			hidden2_LSTMs.append(tf.contrib.rnn.LSTMCell(self.LSTM_outdim, state_is_tuple = True) )#batch * LSTM_outdim

		hidden2_ML_LSTM = tf.nn.rnn_cell.MultiRNNCell(LSTM_cells, state_is_tuple = True) #batch * layer * LSTM_outdim
		self.hidden2_init_state = hidden2_ML_LSTM.zero_state(self.batch_size, tf.float32)





