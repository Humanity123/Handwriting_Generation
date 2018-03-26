import tensorflow as tf
import numpy as np
import math 

class SynNet(object):
	''' class for Synthesis Network Architecture '''

	def __init__(self, config, sess, training = True):
		''' initialisation function for class '''

		self.sess = sess
		self.seq_len = config.seq_len
		self.sen_len = config.sen_len
		self.char_dim = config.char_dim
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

			pi     = tf.nn.softmax(y[:, :mc] * (1 + bias))
			mu1    = y[:, mc : 2 * mc]
			mu2    = y[:, 2* mc : 3 * mc]
			sigma1 = tf.exp(y[:, 3 * mc : 4 * mc] - bias) + self.eps
			sigma2 = tf.exp(y[:, 4 * mc : 5 * mc] - bias ) + self.eps
			rho    = tf.tanh(y[:, 5 * mc : 6 * mc])
			e      = 1. / (1 + tf.exp(y[:, 6 * mc :]))

			return pi, mu1, mu2, sigma1, sigma2, rho, e
		
		def get_window_parameters(window_parameters, prev_kappa):
			''' computes the character weights for the text string '''

			wmc = self.window_mixture_components

			alpha = tf.exp(window_parameters[:, :wmc])
			beta  = tf.exp(window_parameters[:, wmc: 2 * wmc])
			kappa = prev_kappa + tf.exp(window_parameters[:, 2 * wmc :])

			return alpha, beta, kappa, prev_kappa

		def get_phi(alpha, beta, kappa, u):
			''' computes phi(character weight) for a character in the text string '''

			phi = tf.multiply(alpha, tf.exp(-1 * tf.multiply(beta, tf.square(kappa - u) ) ) )
			reduced_phi = tf.reduce_sum(phi, axis = 1)

			return reduced_phi
			
		def get_loss(normal_bivariate_prob, pi, eos, e, mask):
			''' function to calculate the loss '''

			eos_prob = tf.multiply(eos, e) + tf.multiply((1. - eos), (1. - e))
			eos_prob_1dim = tf.reshape(eos_prob, [self.batch_size])
			self.check1 = eos_prob_1dim

			pi_prob  = tf.multiply(pi, normal_bivariate_prob)
			self.check2 = pi_prob
			reduced_pi_prob = tf.reduce_sum(pi_prob, axis = 1)
			self.check3 = reduced_pi_prob

			loss = tf.multiply(reduced_pi_prob, eos_prob_1dim)
			log_loss = tf.log(tf.maximum(loss, 1e-20))
			masked_log_loss = tf.multiply(mask, log_loss)
			self.see = masked_log_loss
			reduced_log_loss = (-1 * tf.reduce_sum(masked_log_loss, axis = 0))
			return reduced_log_loss

		self.input    = tf.placeholder(tf.float32, [None, self.seq_len, 3], name = 'input')
		self.target   = tf.placeholder(tf.float32, [None, self.seq_len, 3], name = 'target')
		self.text   = tf.placeholder(tf.float32, [None, self.sen_len, self.char_dim], name = 'target')
		self.mask     = tf.placeholder(tf.float32, [None, self.seq_len], name = 'mask')
		self.sen_mask = tf.placeholder(tf.float32, [None, self.sen_len], name = 'sen_mask')

		char_indices= tf.constant(range(self.sen_len))

		hidden1_W   = tf.get_variable('hidden_W', [self.hidden1_LSTM_outdim, self.window_dim])
		hidden1_b   = tf.get_variable('hidden_b', [self.window_dim])
		hidden2_W   = tf.get_variable('hidden_W', [self.hidden2_LSTM_outdim, self.mixture_dim])
		hidden2_b   = tf.get_variable('hidden_b', [self.mixture_dim])

		hidden1_LSTM = tf.contrib.rnn.LSTMCell(self.hidden1_LSTM_outdim, state_is_tuple = True)
		self.hidden1_init_state = hidden1_LSTM.zero_state(self.batch_size, tf.float32)

		hidden2_LSTMs = []
		for idx in range(self.LSTM_layers):
			hidden2_LSTMs.append(tf.contrib.rnn.LSTMCell(self.hidden_LSTM_outdim, state_is_tuple = True) )#batch * LSTM_outdim

		hidden2_ML_LSTM = tf.nn.rnn_cell.MultiRNNCell(LSTM_cells, state_is_tuple = True) #batch * layer * LSTM_outdim
		self.hidden2_init_state = hidden2_ML_LSTM.zero_state(self.batch_size, tf.float32)

		hidden1_prev_state = hidden1_init_state
		hidden2_prev_state = hidden2_init_state
		prev_kappa = tf.zeros([self.batch_size, self.window_mixture_components]) 
		prev_w     = tf.zeros([self.batch_size, self.char_dim])

		self.total_loss = tf.zeros([])

		for time_step in range(self.seq_len):

			hidden1_LSTM_input = tf.concat([self.input[:, time_step, :], prev_w], axis = 1)
			hidden1_LSTM_out, hidden1_prev_state = hidden1_LSTM(hidden1_LSTM_input, hidden1_prev_state) #batch , win_dim

			raw_window_params = tf.matmul(hidden1_LSTM_out, hidden_W) + hidden_b
			alpha, beta, kappa, prev_kappa = get_window_parameters(raw_window_params, prev_kappa) #batch, win_comp

			get_phi_fn = lambda x: get_phi(alpha, beta, kappa, x)
			phi = tf.transpose(tf.map_fn(get_phi_fn, char_indices), perm = [1,0])
			phi_3dim = tf.reshape(phi, [self.batch_size, self.sen_len, 1])
			til_phi  = tf.tile(phi_3dim, [1, 1, self.char_dim]) 
			w = tf.multiply(til_phi, self.text)
			sen_mask_3dim = tf.reshape(self.sen_mask, [self.batch_size, self.sen_mask, 1])
			til_sen_mask = tf.tile = tf.tile(sen_mask_3dim, [1, 1, self.char_dim])
			masked_w = tf.multiply(til_sen_mask, w)
			reduced_w = tf.reduce_sum(masked_w, axis = 1) # batch char_dim
			prev_w = reduced_w

			hidden2_LSTM_input = tf.concat([self.input[:, time_step, :], reduced_w, hidden1_LSTM_out], axis = 1)
			hidden2_LSTM_out, hidden2_prev_state = hidden2_ML_LSTM(hidden2_LSTM_input, hidden2_prev_state) #batch hidden2_outdim

			hidden2_b_2dim = tf.reshape(hidden2_b, [1, self.mixture_dim])
			til_hidden2_b = tf.tile(hidden2_b_2dim, [self.batch_size, 1])
			
			y = tf.matmul(hidden2_LSTM_out, hidden2_W) + til_hidden2_b
			self.pi, self.mu1, self.mu2, self.sigma1, self.sigma2, self.rho, self.e = get_mixture_parameters(y, self.bias)

			eos, x1, x2 = tf.split(axis = 1, num_or_size_splits = 3, value = self.target[:, time_step, :])

			til_x1  = tf.tile(x1, [1, self.mixture_components])
			til_x2  = tf.tile(x2, [1, self.mixture_components])
			eos     = eos

			normal_bivariate_prob = normal_bivariate(til_x1, til_x2, self.mu1, self.mu2, self.sigma1, self.sigma2, self.rho) #batch time matrix_com
			self.check4 = normal_bivariate_prob
			self.loss = get_loss(normal_bivariate_prob, self.pi, eos, self.e, self.mask[:, time_step]) 
			self.total_loss = self.total_loss + self.loss

		self.lr = tf.Variable(self.init_lr, trainable=False)
		self.opt = tf.train.RMSPropOptimizer(self.lr)

		grads_and_vars = self.opt.compute_gradients(self.loss)
		clipped_grads_and_vars = [(tf.clip_by_value(gv[0], -1 * self.grad_clip, self.grad_clip), gv[1]) \
                                for gv in grads_and_vars]

		inc = self.global_step.assign_add(1)
		with tf.control_dependencies([inc]):
			self.optim = self.opt.apply_gradients(clipped_grads_and_vars)

  		self.sess.run(tf.initialize_all_variables())







			











