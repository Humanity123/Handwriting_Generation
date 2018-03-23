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

			mc = self.mixture_components

			pi     = tf.nn.softmax(y[:, :, :mc])
			mu1    = y[:, :, mc : 2 * mc]
			mu2    = y[:, :, 2* mc : 3 * mc]
			sigma1 = tf.exp(y[:, :, 3 * mc : 4 * mc])
			sigma2 = tf.exp(y[:, :, 4 * mc : 5 * mc])
			rho    = tf.tanh(y[:, :, 5 * mc : 6 * mc])
			e      = 1. / tf.exp(y[:, :, 6 * mc :])

			return pi, mu1, mu2, sigma1, sigma2, rho, e
			
		def get_loss(normal_bivariate_prob, pi, eos, e, mask):
			''' function to calculate the loss '''

			eos_prob = tf.multiply(eos, e) + tf.multiply((1. - eos), (1. - e))
			pi_prob  = tf.multiply(pi, normal_bivariate_prob)
			reduced_pi_prob = tf.reduce_sum(pi_prob, axis = 2)

			loss_per_timestep = tf.multiply(reduced_pi_prob, eos_prob)
			loss_per_timestep_2dim = tf.reshape(loss_per_timestep, [self.batch_size, self.seq_len])
			masked_loss_per_timestep = tf,multiply(tf.mask, loss_per_timestep_2dim)
			loss = tf.reduce_sum(masked_loss_per_timestep, axis = 1)

			log_loss = -1 * tf.reduced_sum(tf.log(loss), axis = 0)
			return log_loss


		self.input  = tf.placeholder(tf.float32, [None, self.seq_len, 3], name = 'input')
		self.mask   = tf.placeholder(tf.float32, [None, seq_len], name = 'mask')

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

		y = tf.add(reduced_out, til_output_b_4dim) $ batch time mix_dim
		y_except_last = t[:, :-1, :]
		pi, mu1, mu2, sigma1, sigma2, rho, e = get_mixture_parameters(y_except_last)

		x1, x2, eos = tf.split(axis = 1, num_or_size_splits = 3, value = self.input)

		til_x1_t_plus_1  = tf.tile(x1[:, 1:, :], [1, 1, self.mixture_components])
		til_x2_t_plus_1  = tf.tile(x2[:, 1:, :], [1, 1, self.mixture_components])
		eos_t_plus_1 = eos[:, 1:, :]

		normal_bivariate_prob = normal_bivariate(til_x1_t_plus_1, til_x2_t_plus_1, mu1, mu2, sigma1, sigma2, rho) #batch time matrix_com
		loss = get_loss(normal_bivariate_prob, pi, eos_t_plus_1, e, mask)












	def train():
		''' function to train the model '''
		pass

