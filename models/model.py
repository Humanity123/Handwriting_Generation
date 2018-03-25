import tensorflow as tf
import numpy as np
import math 


class MixtureDensityNetwork(object):
	''' class for Mixture Density Network '''

	def __init__(self, config, sess, training = True):
		''' initialisation function for class '''

		self.sess = sess
		self.seq_len = config.seq_len
		self.LSTM_layers = config.LSTM_layers
		self.mixture_components = config.mixture_components
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

		self.Mixture_dim = 1 + self.mixture_components * 6

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
			
		def get_loss(normal_bivariate_prob, pi, eos, e, mask):
			''' function to calculate the loss '''

			eos_prob = tf.multiply(eos, e) + tf.multiply((1. - eos), (1. - e))
			eos_prob_2dim = tf.reshape(eos_prob, [self.batch_size, self.seq_len])
			self.check1 = eos_prob_2dim

			pi_prob  = tf.multiply(pi, normal_bivariate_prob)
			self.check2 = pi_prob
			reduced_pi_prob = tf.reduce_sum(pi_prob, axis = 2)
			self.check3 = reduced_pi_prob

			loss_per_timestep = tf.multiply(reduced_pi_prob, eos_prob_2dim)
			loss_per_timestep_2dim = tf.reshape(loss_per_timestep, [self.batch_size, self.seq_len])
			masked_loss_per_timestep = tf.multiply(mask, loss_per_timestep_2dim)
			loss = tf.reduce_sum(masked_loss_per_timestep, axis = 1)
			self.see = loss
			log_loss = (-1 * tf.reduce_sum(tf.log(loss), axis = 0))
			return log_loss


		self.input  = tf.placeholder(tf.float32, [None, self.seq_len, 3], name = 'input')
		self.target  = tf.placeholder(tf.float32, [None, self.seq_len, 3], name = 'target')
		self.mask   = tf.placeholder(tf.float32, [None, self.seq_len], name = 'mask')

		ML_LSTM_output_W = tf.get_variable('ML_LSTM_output_W', [self.LSTM_outdim, self.Mixture_dim])
		ML_LSTM_output_b = tf.get_variable('ML_LSTM_output_b', [self.Mixture_dim])

		LSTM_cells = []
		for idx in range(self.LSTM_layers):
			LSTM_cells.append(tf.contrib.rnn.LSTMCell(self.LSTM_outdim, state_is_tuple = True) )#batch * LSTM_outdim

		ML_LSTM_cell = tf.nn.rnn_cell.MultiRNNCell(LSTM_cells, state_is_tuple = True) #batch * layer * LSTM_outdim
		self.init_state = ML_LSTM_cell.zero_state(self.batch_size, tf.float32)

		l = tf.unstack(self.init_state, axis=0)
		rnn_tuple_state = tuple([tf.nn.rnn_cell.LSTMStateTuple(l[idx][0],l[idx][1]) for idx in range(self.LSTM_layers)])

		ML_LSTM_output, self.layer_LSTM_state = tf.nn.dynamic_rnn(ML_LSTM_cell, self.input, initial_state = rnn_tuple_state) #batch * time * layer * LSTM_outdim
		ML_LSTM_output_4dim = tf.reshape(ML_LSTM_output, [self.batch_size, self.seq_len, 1, self.LSTM_outdim])

		output_W_4dim  = tf.reshape(ML_LSTM_output_W, [1, 1, self.LSTM_outdim, self.Mixture_dim])
		til_output_W   = tf.tile(output_W_4dim, [self.batch_size, self.seq_len, 1, 1])

		output_b_3dim  = tf.reshape(ML_LSTM_output_b, [1, 1, self.Mixture_dim])
		til_output_b = tf.tile(output_b_3dim, [self.batch_size, self.seq_len, 1])

		W_multiplied_out = tf.matmul(ML_LSTM_output_4dim, til_output_W) # batch time layer 1 mix_dim
		W_multiplied_out_4dim = tf.reshape(W_multiplied_out, [self.batch_size, self.seq_len, self.Mixture_dim])
		reduced_out = W_multiplied_out_4dim #batch time mix_dim

		y = tf.add(reduced_out, til_output_b) # batch time mix_dim
		self.pi, self.mu1, self.mu2, self.sigma1, self.sigma2, self.rho, self.e = get_mixture_parameters(y, self.bias)

		eos, x1, x2 = tf.split(axis = 2, num_or_size_splits = 3, value = self.target)

		til_x1  = tf.tile(x1, [1, 1, self.mixture_components])
		til_x2  = tf.tile(x2, [1, 1, self.mixture_components])
		eos     = eos

		normal_bivariate_prob = normal_bivariate(til_x1, til_x2, self.mu1, self.mu2, self.sigma1, self.sigma2, self.rho) #batch time matrix_com
		self.check4 = normal_bivariate_prob
		self.loss = get_loss(normal_bivariate_prob, self.pi, eos, self.e, self.mask) / self.batch_size

		self.lr = tf.Variable(self.init_lr, trainable=False)
		self.opt = tf.train.AdagradOptimizer(self.lr)

		grads_and_vars = self.opt.compute_gradients(self.loss)
		clipped_grads_and_vars = [(tf.clip_by_norm(gv[0], self.grad_clip), gv[1]) \
                                for gv in grads_and_vars]

		inc = self.global_step.assign_add(1)
		with tf.control_dependencies([inc]):
			self.optim = self.opt.apply_gradients(clipped_grads_and_vars)

  		self.sess.run(tf.initialize_all_variables())

	def train(self, strokes_train_data, strokes_target_data, saver):
		''' function to train the model '''

		train_input  = np.ndarray([self.batch_size, self.seq_len, 3], np.float32)
		train_target = np.ndarray([self.batch_size, self.seq_len, 3], np.float32)
		train_mask   = np.ndarray([self.batch_size, self.seq_len], np.float32)

		best_loss = np.inf
		for epoch in range(self.epochs):
			num_batches = int(math.ceil(len(strokes_train_data) / self.batch_size))
			rand_idx, cur = np.random.permutation(len(strokes_train_data)), 0

			total_loss_per_epoch = 0.
			for batch_idx in range(num_batches):
				train_input.fill(0.)
				train_mask.fill(0.)

				for idx in range(self.batch_size):
					stroke_idx = rand_idx[cur]

					stroke_input_data = strokes_train_data[stroke_idx]
					train_input[idx, :stroke_input_data.shape[0], :] = stroke_input_data


					stroke_target_data = strokes_target_data[stroke_idx]
					train_target[idx, :stroke_target_data.shape[0], :] = stroke_target_data

					train_mask[idx, :stroke_input_data.shape[0]].fill(1.)
					cur += 1

				c1,c2,c3, c4, s, loss, _ = self.sess.run( [self.check1, self.check2, self.check3, self.check4, self.see, self.loss, self.optim], feed_dict = {
																	self.input : train_input,
																	self.target: train_target,
																	self.mask  : train_mask})
				total_loss_per_epoch += loss
				print loss
				print np.any(np.isnan(c4))
				print np.any(np.isnan(c3))
				print np.any(np.isnan(c2))
				print np.any(np.isnan(c1))
				

				
			print('epoch=%d train-loss=%.2f;' % (epoch, loss))
			if total_loss_per_epoch < best_loss:
				best_loss = total_loss_per_epoch
				saver.save(self.sess, self.saved_model_directory, global_step = epoch)


	def synthesize(self, length):
		''' function to generate unconditional text '''

		def generate_sample(parameters):
			''' generates a sample of the mixture model of bivariate gaussian and bernoulli '''
			
			pi, mu1, mu2, sigma1, sigma2, rho, e = parameters
			random_number = np.random.random()

			cumsum_pi = np.cumsum(pi)
			try:
				mixture_idx = next(x[0] for x in enumerate(cumsum_pi) if x[1] >= random_number)
			except:
				mixture_idx = self.mixture_components - 1 
				print mixture_idx

			eos = 1 if e >= random_number else 0

			mix_pi, mix_mu1, mix_mu2, mix_sigma1, mix_sigma2, mix_rho = [param[0][0][mixture_idx] for param in parameters[:-1]]
			mean = [mix_mu1, mix_mu2]
			cov = [[mix_sigma1 * mix_sigma1, mix_rho * mix_sigma1 * mix_sigma2], [mix_rho * mix_sigma1 * mix_sigma2, mix_sigma2 * mix_sigma2]]
			x1, x2 = np.random.multivariate_normal(mean, cov)

			return eos, x1, x2

		stroke_data = np.zeros([1, length, 3])
		mask_data = np.ones([1, length])

		stroke_data[0][0][0] = 1

		init_state = self.sess.run(self.init_state)

		for stroke_idx in range(length-1):
			init_state, pi, mu1, mu2, sigma1, sigma2, rho, e = self.sess.run([self.layer_LSTM_state, self.pi, self.mu1, self.mu2, self.sigma1, 
																	self.sigma2, self.rho, self.e], feed_dict = {
																	self.input : stroke_data[:, stroke_idx: stroke_idx+1, :],
																	self.mask : mask_data[:, stroke_idx: stroke_idx+1],
																	self.init_state : init_state})
			stroke_data[0, stroke_idx, :] = generate_sample((pi, mu1, mu2, sigma1, sigma2, rho, e))


		return stroke_data[0]




