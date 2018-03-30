import tensorflow as tf
from tensorflow.python import debug as tf_debug
import numpy as np
import math 

class RNNModel(tf.nn.rnn_cell.RNNCell):
	''' class for the RNN cell, inherits from RNN cell '''

	def __init__(self, LSTM_layers, num_units, batch_size):
		
		super(RNNModel, self).__init__()
		self.LSTM_layers = LSTM_layers
		self.num_units = num_units

		with tf.variable_scope('rnn', reuse=None):
			self.lstms = [tf.nn.rnn_cell.LSTMCell(num_units) for _ in range(self.LSTM_layers)]
			self.states = [tf.Variable(tf.zeros([batch_size, s]), trainable=False) for s in self.state_size]
			self.zero_states = tf.group(*[sp.assign(sc) for sp, sc in zip(self.states, self.zero_state(batch_size, dtype=tf.float32))])
														

	@property
	def state_size(self):
		return [self.num_units] * self.LSTM_layers * 2 

	@property
	def output_size(self):
		return [self.num_units]

	def call(self, inputs, state, **kwargs):
		output_state = []
		prev_output = []
		for layer in range(self.LSTM_layers):
			if layer == 0:
				x = inputs
			else:
				x = tf.concat([inputs, prev_output], axis = 1)
			
			with tf.variable_scope('lstm_{}'.format(layer)):
				output, s = self.lstms[layer](x, tf.nn.rnn_cell.LSTMStateTuple(state[2 * layer], state[2 * layer + 1]))
				prev_output = output
		
			output_state += [s[0]] + [s[1]]
		
		return output, output_state



variables = dict()
def add_to_dict(variable, name):
	variables[name] = variable

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
		self.decay = config.decay
		self.momentum = config.momentum
		self.grad_clip = config.grad_clip
		self.epochs = config.epochs
		self.eps = config.eps
		self.bias = config.bias
		self.data_scale = config.data_scale
		self.saved_model_directory = config.saved_model_directory

		if training:
			self.bias = 0.
			self.RNN_outkeep_prob = config.RNN_outkeep_prob

		if not training:
			self.batch_size = 1
			self.seq_len = 1
			self.RNN_outkeep_prob = 1.0

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

			# print "norm_x1 - ", norm_x1
			# print "norm_x2 - ", norm_x2
			# print "Z - ", Z
			# print "C - ", C

			normal_bivariate_prob = (1.0 / (2. * cons_pi * sigma1 * sigma2)) * tf.sqrt(C) * tf.exp((-1 * Z) * C / 2.)

			# print "normal_bivariate_prob - ", normal_bivariate_prob
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
			# print "e - ", e
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

			bernoulli_log_loss = tf.log(tf.maximum(eos_prob_2dim, 1e-20))
			gaussian_log_loss  =  tf.log(tf.maximum(reduced_pi_prob, 1e-20))
			masked_bernoulli_log_loss = tf.multiply(mask, bernoulli_log_loss)
			masked_gaussian_log_loss  = tf.multiply(mask, gaussian_log_loss)
			total_loss_per_timestep = masked_gaussian_log_loss + masked_bernoulli_log_loss
			total_loss = tf.reduce_sum(total_loss_per_timestep, axis = 1)
			self.see = total_loss
			reduced_total_loss = -1 * tf.reduce_sum(total_loss, axis = 0)

			return reduced_total_loss


		self.input  = tf.placeholder(tf.float32, [None, self.seq_len, 3], name = 'input')
		self.target  = tf.placeholder(tf.float32, [None, self.seq_len, 3], name = 'target')
		self.mask   = tf.placeholder(tf.float32, [None, self.seq_len], name = 'mask')

		add_to_dict(self.input, "input")
		add_to_dict(self.target, "target")
		add_to_dict(self.mask, "mask")

		self.graves_initializer = tf.truncated_normal_initializer(mean=0., stddev=.075, seed=None, dtype=tf.float32)

		ML_LSTM_output_W = tf.get_variable('ML_LSTM_output_W', [self.LSTM_outdim, self.Mixture_dim], initializer=self.graves_initializer)
		ML_LSTM_output_b = tf.get_variable('ML_LSTM_output_b', [self.Mixture_dim], initializer=self.graves_initializer)

		add_to_dict(ML_LSTM_output_W, "ML_LSTM_output_W")
		add_to_dict(ML_LSTM_output_b, "ML_LSTM_output_b")

		ML_LSTM_cell = RNNModel(self.LSTM_layers, self.LSTM_outdim, self.batch_size)
		self.init_state = tf.identity(ML_LSTM_cell.zero_state(self.batch_size, tf.float32))

		l = tf.unstack(self.init_state, axis=0)
		rnn_tuple_state = l

		dropped_out_LSTM_cell = tf.nn.rnn_cell.DropoutWrapper(ML_LSTM_cell, output_keep_prob = RNN_outkeep_prob)
		ML_LSTM_output, self.layer_LSTM_state = tf.nn.dynamic_rnn(dropped_out_LSTM_cell, self.input, initial_state = rnn_tuple_state) #batch * time  * LSTM_outdim

		add_to_dict(ML_LSTM_output, "ML_LSTM_output")
		add_to_dict(self.layer_LSTM_state, "self.layer_LSTM_state")

		ML_LSTM_output_4dim = tf.reshape(ML_LSTM_output, [self.batch_size, self.seq_len, 1, self.LSTM_outdim])
		add_to_dict(ML_LSTM_output_4dim, 'ML_LSTM_output_4dim')

		output_W_4dim  = tf.reshape(ML_LSTM_output_W, [1, 1, self.LSTM_outdim, self.Mixture_dim])
		til_output_W   = tf.tile(output_W_4dim, [self.batch_size, self.seq_len, 1, 1])

		output_b_3dim  = tf.reshape(ML_LSTM_output_b, [1, 1, self.Mixture_dim]) 
		til_output_b = tf.tile(output_b_3dim, [self.batch_size, self.seq_len, 1])

		W_multiplied_out = tf.matmul(ML_LSTM_output_4dim, til_output_W) # batch time layer 1 mix_dim
		W_multiplied_out_4dim = tf.reshape(W_multiplied_out, [self.batch_size, self.seq_len, self.Mixture_dim])
		reduced_out = W_multiplied_out_4dim #batch time mix_dim

		y = reduced_out + til_output_b # batch time mix_dim
		add_to_dict(y, "y")

		self.pi, self.mu1, self.mu2, self.sigma1, self.sigma2, self.rho, self.e = get_mixture_parameters(y, self.bias)

		eos, x1, x2 = tf.split(axis = 2, num_or_size_splits = 3, value = self.target)

		til_x1  = tf.tile(x1, [1, 1, self.mixture_components])
		til_x2  = tf.tile(x2, [1, 1, self.mixture_components])
		self.eos     = eos

		normal_bivariate_prob = normal_bivariate(til_x1, til_x2, self.mu1, self.mu2, self.sigma1, self.sigma2, self.rho) #batch time matrix_com
		self.check4 = normal_bivariate_prob
		add_to_dict(normal_bivariate_prob, "normal_bivariate_prob")
		self.loss = get_loss(normal_bivariate_prob, self.pi, eos, self.e, self.mask) / (self.batch_size * self.seq_len)

		self.lr = tf.Variable(self.init_lr, trainable=False)
		learning_rate = tf.train.exponential_decay(self.lr, self.global_step, staircase=True, decay_steps=10000, decay_rate=0.5)
		self.opt = tf.train.AdamOptimizer(learning_rate=learning_rate)

		tvars = tf.trainable_variables()
		grads_and_vars = self.opt.compute_gradients(self.loss, tvars)
		self.g = grads_and_vars
		clipped_grads_and_vars = [(tf.clip_by_value(gv[0], -1 * self.grad_clip, self.grad_clip), gv[1]) \
                                for gv in grads_and_vars]

		inc = self.global_step.assign_add(1)
		with tf.control_dependencies([inc]):
			self.optim = self.opt.apply_gradients(clipped_grads_and_vars, global_step=self.global_step)

  		self.sess.run(tf.initialize_all_variables())

	def train(self, strokes_train_data, strokes_target_data, saver):
		''' function to train the model '''

		train_input  = np.ndarray([self.batch_size, self.seq_len, 3], np.float32)
		train_target = np.ndarray([self.batch_size, self.seq_len, 3], np.float32)
		train_mask   = np.ndarray([self.batch_size, self.seq_len], np.float32)

		best_loss = np.inf
		loss_list = []
		for epoch in range(self.epochs):
			num_batches = int(math.ceil(len(strokes_train_data) / self.batch_size))
			rand_idx, cur = np.random.permutation(len(strokes_train_data)), 0

			total_loss_per_epoch = 0.
			debug_dicts = []
			for batch_idx in range(num_batches):
				if batch_idx == 2:
					return debug_dicts

				train_input.fill(0.)
				train_mask.fill(0.)
				train_target.fill(0.)
				for idx in range(self.batch_size):
					stroke_idx = rand_idx[cur]

					stroke_input_data = strokes_train_data[stroke_idx]
					train_input[idx, :stroke_input_data.shape[0], :] = stroke_input_data


					stroke_target_data = strokes_target_data[stroke_idx]
					train_target[idx, :stroke_target_data.shape[0], :] = stroke_target_data

					train_mask[idx, :stroke_input_data.shape[0]].fill(1.)
					cur += 1

					#return train_input[idx], train_target[idx], train_mask[idx] 
					# print train_input[idx]
					# print train_target[idx]
					# print train_mask[idx]
					# exit()

				debug_params_name = variables.keys()
				
				# debug_params_add = self.sess.run(variables.values()+[self.loss, self.optim], feed_dict = {
				# 													self.input : train_input,
				# 													self.target: train_target,
				# 													self.mask  : train_mask})
				eos, pi, mu1, mu2, sigma1, sigma2, rho, e,g, c1,c2,c3, c4, s, loss, o = self.sess.run( [ self.eos, self.pi, self.mu1, self.mu2, self.sigma1, self.sigma2, self.rho, self.e, self.g, self.check1, self.check2, self.check3, self.check4, self.see, self.loss, self.optim], feed_dict = {
																	self.input : train_input,
																	self.target: train_target,
																# self.mask  : train_mask})
				# loss = debug_params_add[-2]
				# debug_params = debug_params_add[:-2]
				# debug_dict = dict(zip(debug_params_name, debug_params))
				# print debug_dict['target'][0, :40, :]
				# print debug_dict['x1'][0, :40, :]
				# debug_dicts.append(debug_dict)

				total_loss_per_epoch += loss
				print loss
				loss_list.append(loss)
				if np.isnan(loss):
					return loss_list		
				
			print('epoch=%d train-loss=%.2f;' % (epoch, total_loss_per_epoch))
			if total_loss_per_epoch < best_loss:
				best_loss = total_loss_per_epoch
				saver.save(self.sess, self.saved_model_directory+"/uncond", global_step = epoch)


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
			stroke_data[0, stroke_idx+1, :] = generate_sample((pi, mu1, mu2, sigma1, sigma2, rho, e))

		return stroke_data[0]




