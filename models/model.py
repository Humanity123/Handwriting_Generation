import tensorflow as tf
import math 


class MixtureDensityNetwork(object):
	''' class for Mixture Density Network '''

	def __init__(self, config, sess):
		''' initialisation function for class '''
		self.sess = sess
		self.seq_len = config.seq_len
		self.LSTM_layers = config.LSTM_layers
		self.mixture_components = config.mixture_components
		self.batch_size = config.batch_size
		self.LSTM_outdim = config.LSTM_outdim

		self.Mixture_dim = 1 + self.mixture_components * 6

	def build_model(self):
		''' function to build the model '''
		def normal_bivariate(x1, x2, mu1, mu2, sigma1, sigma2, rho):
			''' calculates the bivariate normal function from the parameters '''
			cons_pi = tf.constant(m.pi)
			norm_x1 = (x1 - mu1) / sigma1
			norm_x2 = (x2 - mu2) / sigma2
			Z = tf.square(norm_x1) + tf.square(norm_x2) - (norm_x1 * norm_x2) 
			C = 1.0 / (1.0 - tf.square(rho)

			normal_bivariate_prob = (1.0 / (2 * cons_pi * sigma1 * sigma2)) * tf.sqrt(C) * tf.exp((-1 * Z) * C)
			return normal_bivariate_prob

		def get_mixture_parameters(y):
			''' computes the mixture parameters from the LSTM output '''
			pass

		self.input = tf.placeholder(tf.float32, [None, self.seq_len, 3], name = 'input')

		ML_LSTM_output_W = tf.get_variable('ML_LSTM_output_W', [self.LSTM_layers, self.LSTM_outdim, self.Mixture_dim])
		ML_LSTM_output_b = tf.get_variable('ML_LSTM_output_b', [self.Mixture_dim])

		LSTM_cell = tf.contrib.rnn.LSTMCell(self.LSTM_outdim, state_is_tuple = True) #batch * LSTM_outdim
		ML_LSTM_cell = tf.nn.rnn_cell.MultiRNNCell([LSTM_cell] * self.LSTM_layers, state_is_tuple=True) #batch * layer * LSTM_outdim
		ML_LSTM_output, layer_LSTM_state = tf.nn.dyanmic_rnn(ML_LSTM_cell, self.input) #batch * time * layer * LSTM_outdim

		ML_LSTM_output_5dim = tf.reshape(ML_LSTM_output, [self.batch_size, self.seq_len, self.LSTM_layers, 1, self.LSTM_outdim])

		output_W_5dim  = tf.reshape(ML_LSTM_output_W, [1, 1, self.LSTM_layers, self.LSTM_outdim, self.Mixture_dim])
		til_output_W   = tf.tile(output_W_5dim, [self.batch_size, self.seq_len, 1, 1, 1])
		
		output_b_3dim  = tf.reshape(ML_LSTM_output_b, [1, 1, self.Mixture_dim])
		til_output_b = tf.tile(output_b_3dim, [self.batch_size, self.seq_len, 1])

		W_multiplied_out = tf.matmul(ML_LSTM_output_5dim, til_output_W) # batch time layer 1 mix_dim
		W_multiplied_out_4dim = tf.reshape(W_multiplied_out, [self.batch_size, self.seq_len, self.LSTM_layers, self.Mixture_dim])
		reduced_out = tf.reduce_sum(W_multiplied_out_4dim, axis = 2) #batch time mix_dim

		y = tf.add(reduced_out, til_output_b_4dim)






	def train():
		''' function to train the model '''
		pass

