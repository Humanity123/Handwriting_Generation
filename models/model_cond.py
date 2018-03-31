import tensorflow as tf
import numpy as np
import math 

class RNNModel(tf.nn.rnn_cell.RNNCell):
	''' class for the RNN cell, inherits from RNN cell '''

	def __init__(self, LSTM_layers, num_units, batch_size, window_fn, window_output_size):
		
		super(RNNModel, self).__init__()
		self.LSTM_layers = LSTM_layers
		self.num_units = num_units
		self.window_fn = window_fn
		self.window_output_size = window_output_size

		with tf.variable_scope('rnn', reuse=None):
			self.lstms = [tf.nn.rnn_cell.LSTMCell(num_units) for _ in range(self.LSTM_layers)]														

	@property
	def state_size(self):
		return [self.num_units] * self.LSTM_layers * 2 + self.window_output_size

	@property
	def output_size(self):
		# return [self.num_units*self.LSTM_layers]
		return [self.num_units]

	def call(self, inputs, state, **kwargs):
		window, kappa, finish = state[-3:]
		output_state = []
		final_output = []
		prev_output = 0

		for layer in range(self.LSTM_layers):
			if layer == 0:
				x = tf.concat([inputs, window], axis = 1)
			else:
				x = tf.concat([inputs, window, prev_output], axis = 1)
			
			with tf.variable_scope('lstm_{}'.format(layer)):
				output, next_state = self.lstms[layer](x, tf.nn.rnn_cell.LSTMStateTuple(state[2 * layer], state[2 * layer + 1]))
				prev_output = output
		
			output_state += [next_state[0]] + [next_state[1]]
			final_output.append(output)

			if layer == 0:
				window, kappa, finish = self.window_fn(output, kappa)
			print "ye dekh - ", output
		# return tf.concat(final_output, axis = 1) , output_state + [window, kappa, finish]
		return output , output_state + [window, kappa, finish]


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
		self.window_mixture_components = config.window_mixture_components
		self.batch_size = config.batch_size
		self.LSTM_outdim = config.LSTM_outdim
		self.init_lr = config.lr
		self.decay = config.decay
		self.momentum = config.momentum
		self.grad_clip = config.grad_clip
		self.epochs = config.epochs
		self.eps = config.eps
		self.bias = 0.
		self.data_scale = config.data_scale
		self.RNN_outkeep_prob = config.RNN_outkeep_prob
		self.saved_model_directory = config.saved_model_directory	

		if not training:
			self.bias = config.bias
			self.RNN_outkeep_prob = 1.
			self.batch_size = 1
			self.seq_len = 1

		self.mixture_dim = 1 + self.mixture_components * 6
		self.window_dim  = self.window_mixture_components * 3
		# self.RNN_outdim = self.LSTM_outdim * self.LSTM_layers
		self.RNN_outdim = self.LSTM_outdim


	def build_model(self):
		''' function to build the model '''

		self.global_step = tf.Variable(0, name='global_step', trainable=False)

		def normal_bivariate(x1, x2, mu1, mu2, sigma1, sigma2, rho):
			''' calculates the bivariate normal function from the parameters '''

			cons_pi = tf.constant(math.pi)
			norm_x1 = (x1 - mu1) / sigma1
			norm_x2 = (x2 - mu2) / sigma2
			Z = tf.square(norm_x1) + tf.square(norm_x2) - (2. * rho * (norm_x1 * norm_x2))
			C = 1.0 / tf.maximum((1.0 - tf.square(rho)),  self.eps)

			normal_bivariate_prob = (1.0 / (2. * cons_pi * sigma1 * sigma2)) * tf.sqrt(C) * tf.exp((-1 * Z) * C / 2.)
			return normal_bivariate_prob

		def get_mixture_parameters(y, bias = 0.):
			''' computes the mixture parameters from the LSTM output '''

			mc = self.mixture_components

			pi	 = tf.nn.softmax(y[:, :, :mc] * (1 + bias))
			mu1	= y[:, :, mc : 2 * mc]
			mu2	= y[:, :, 2* mc : 3 * mc]
			sigma1 = tf.maximum(tf.exp(y[:, :, 3 * mc : 4 * mc] - bias), self.eps)
			sigma2 = tf.maximum(tf.exp(y[:, :, 4 * mc : 5 * mc] - bias ), self.eps)
			rho	= tf.tanh(y[:, :, 5 * mc : 6 * mc])
			e	  = 1. / (1 + tf.exp(y[:, :, 6 * mc :]))

			return pi, mu1, mu2, sigma1, sigma2, rho, e
		
		def get_window_parameters(window_parameters, prev_kappa):
			''' computes the character weights for the text string '''

			wmc = self.window_mixture_components

			alpha = tf.exp(window_parameters[:, :wmc])
			beta  = tf.exp(window_parameters[:, wmc: 2 * wmc])
			kappa = prev_kappa + tf.exp(window_parameters[:, 2 * wmc:])

			return alpha, beta, kappa

		def get_phi(alpha, beta, kappa, u):
			''' computes phi(character weight) for a character in the text string '''

			print "u - ", u
			phi = tf.multiply(alpha, tf.exp(-1 * tf.multiply(beta, tf.square(kappa - u) ) ) )
			reduced_phi = tf.reduce_sum(phi, axis = 1)
			print "re phi  - ", reduced_phi
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

			bernoulli_log_loss = tf.log(tf.maximum(eos_prob_2dim, self.eps))
			gaussian_log_loss  =  tf.log(tf.maximum(reduced_pi_prob, self.eps))
			masked_bernoulli_log_loss = tf.multiply(mask, bernoulli_log_loss)
			masked_gaussian_log_loss  = tf.multiply(mask, gaussian_log_loss)
			total_loss_per_timestep = masked_gaussian_log_loss + masked_bernoulli_log_loss
			total_loss = tf.reduce_sum(total_loss_per_timestep, axis = 1)
			self.see = total_loss
			reduced_total_loss = -1 * tf.reduce_sum(total_loss, axis = 0)

			return reduced_total_loss

		def get_window(input, prev_kappa, text, hidden_W, hidden_b, sen_mask):
			''' function to get the window '''

			hidden_b_2dim = tf.reshape(hidden_b, [1, self.window_dim])
			til_hidden_b  = tf.tile(hidden_b_2dim, [self.batch_size, 1])
			print input
			raw_window_params = tf.matmul(input, hidden_W) + til_hidden_b

			alpha, beta, kappa = get_window_parameters(raw_window_params, prev_kappa) #batch, win_comp

			char_indices= tf.constant(range(self.sen_len), dtype = tf.float32)
			phi_fn = lambda x: get_phi(alpha, beta, kappa, x)
			phi = tf.transpose(tf.map_fn(phi_fn, char_indices), perm = [1,0]) # batch sen_len
			is_finish = tf.cast(phi_fn(self.sen_len) > tf.reduce_max(phi[:, :], axis=1), tf.float32)
			is_finish_2_dim = tf.reshape(is_finish, [self.batch_size, 1])

			phi_3dim = tf.reshape(phi, [self.batch_size, self.sen_len, 1])
			til_phi  = tf.tile(phi_3dim, [1, 1, self.char_dim]) 
			w = tf.multiply(til_phi, text)
			sen_mask_3dim = tf.reshape(sen_mask, [self.batch_size, self.sen_len, 1])
			til_sen_mask =  tf.tile(sen_mask_3dim, [1, 1, self.char_dim])
			masked_w = tf.multiply(til_sen_mask, w)
			reduced_w = tf.reduce_sum(masked_w, axis = 1) # batch char_dim

			return reduced_w, kappa, is_finish_2_dim

		self.input	= tf.placeholder(tf.float32, [self.batch_size, self.seq_len, 3], name = 'input')
		self.target   = tf.placeholder(tf.float32, [self.batch_size, self.seq_len, 3], name = 'target')
		self.text   = tf.placeholder(tf.float32, [self.batch_size, self.sen_len, self.char_dim], name = 'target')
		self.mask	 = tf.placeholder(tf.float32, [self.batch_size, self.seq_len], name = 'mask')
		self.sen_mask = tf.placeholder(tf.float32, [self.batch_size, self.sen_len], name = 'sen_mask')

		hidden1_W   = tf.get_variable('hidden1_W', [self.LSTM_outdim, self.window_dim])
		hidden1_b   = tf.get_variable('hidden1_b', [self.window_dim])
		hidden2_W   = tf.get_variable('hidden2_W', [self.RNN_outdim, self.mixture_dim])
		hidden2_b   = tf.get_variable('hidden2_b', [self.mixture_dim])

		window_fn = lambda x, y: get_window(x, y, self.text, hidden1_W, hidden1_b, self.sen_mask)
		window_fn_outdim = [self.char_dim, self.window_mixture_components, 1]

		ML_LSTM_cell = RNNModel(self.LSTM_layers, self.LSTM_outdim, self.batch_size, window_fn, window_fn_outdim)
		ML_LSTM_zero_states = ML_LSTM_cell.zero_state(self.batch_size, tf.float32)
		self.init_state = [tf.Variable(state, trainable = False) for state in ML_LSTM_zero_states]

		rnn_tuple_state = self.init_state

		dropped_out_LSTM_cell = tf.nn.rnn_cell.DropoutWrapper(ML_LSTM_cell, output_keep_prob = self.RNN_outkeep_prob)
		ML_LSTM_output, self.layer_LSTM_state = tf.nn.dynamic_rnn(dropped_out_LSTM_cell, self.input, initial_state = rnn_tuple_state) #batch * time  * LSTM_outdim
		print ML_LSTM_output

		self.assign_state_op  = tf.group(*[tf.assign(prev_state_tensor, next_state_tensor) for prev_state_tensor, next_state_tensor in zip(self.init_state, self.layer_LSTM_state)])
		self.reset_state_op  = tf.group(*[tf.assign(state_tensor, zero_state_tensor) for state_tensor, zero_state_tensor in zip(self.init_state, ML_LSTM_zero_states)])
		
		ML_LSTM_output_4dim = tf.reshape(ML_LSTM_output, [self.batch_size, self.seq_len, 1, self.RNN_outdim])

		output_W_4dim  = tf.reshape(hidden2_W, [1, 1, self.RNN_outdim, self.mixture_dim])
		til_output_W   = tf.tile(output_W_4dim, [self.batch_size, self.seq_len, 1, 1])

		output_b_3dim  = tf.reshape(hidden2_b, [1, 1, self.mixture_dim])
		til_output_b = tf.tile(output_b_3dim, [self.batch_size, self.seq_len, 1])

		W_multiplied_out = tf.squeeze(tf.matmul(ML_LSTM_output_4dim, til_output_W)) # batch time layer 1 mix_dim
		# W_multiplied_out_3dim = tf.reshape(W_multiplied_out, [self.batch_size, self.seq_len, self.mixture_dim])
		# reduced_out = W_multiplied_out_3dim #batch time mix_dim

		y = W_multiplied_out + til_output_b # batch time mix_dim
		self.pi, self.mu1, self.mu2, self.sigma1, self.sigma2, self.rho, self.e = get_mixture_parameters(y, self.bias)

		eos, x1, x2 = tf.split(axis = 2, num_or_size_splits = 3, value = self.target)

		til_x1    = tf.tile(x1, [1, 1, self.mixture_components])
		til_x2    = tf.tile(x2, [1, 1, self.mixture_components])
		self.eos  = eos

		normal_bivariate_prob = normal_bivariate(til_x1, til_x2, self.mu1, self.mu2, self.sigma1, self.sigma2, self.rho) #batch time matrix_com
		self.check4 = normal_bivariate_prob
		self.loss = get_loss(normal_bivariate_prob, self.pi, eos, self.e, self.mask) / (self.batch_size * self.seq_len)
		# self.loss = get_loss(normal_bivariate_prob, self.pi, eos, self.e, self.mask) / (self.batch_size )

		self.lr = tf.Variable(self.init_lr, trainable=False)
		learning_rate = tf.train.exponential_decay(self.lr, self.global_step, staircase=True, decay_steps=1000, decay_rate=self.decay)
		self.opt = tf.train.AdamOptimizer(learning_rate=learning_rate)

		tvars = tf.trainable_variables()
		grads_and_vars = self.opt.compute_gradients(self.loss, tvars)
		self.g = grads_and_vars
		# clipped_grads_and_vars = [(tf.clip_by_global_norm(gv[0], self.grad_clip), gv[1]) for gv in grads_and_vars]
		clipped_grads_and_vars = [(tf.clip_by_value(gv[0], -1 * self.grad_clip, self.grad_clip), gv[1]) \
                                for gv in grads_and_vars]


		inc = self.global_step.assign_add(1)
		with tf.control_dependencies([inc]):
			self.optim = self.opt.apply_gradients(clipped_grads_and_vars, global_step=self.global_step)

		self.sess.run(tf.initialize_all_variables())
		
	def train(self, strokes_train_data, strokes_target_data, sentence_train_data, saver):
		''' function to train the model '''

		train_input		    = np.ndarray([self.batch_size, self.seq_len, 3], np.float32)
		train_target		= np.ndarray([self.batch_size, self.seq_len, 3], np.float32)
		train_mask		    = np.ndarray([self.batch_size, self.seq_len], np.float32)
		train_sentence	    = np.ndarray([self.batch_size, self.sen_len, self.char_dim], np.float32)
		train_sentence_mask = np.ndarray([self.batch_size, self.sen_len], np.float32)

		best_loss = np.inf

		for epoch in range(self.epochs):
			num_batches = int(math.ceil(len(strokes_train_data) / self.batch_size))
			rand_idx, cur = np.random.permutation(len(strokes_train_data)), 0

			total_loss_per_epoch = 0.
			for batch_idx in range(num_batches):
				train_input.fill(0.)
				train_target.fill(0.)
				train_mask.fill(0.)
				train_sentence.fill(0.)
				train_sentence_mask.fill(0.)

				for idx in range(self.batch_size):
					stroke_idx = rand_idx[cur]

					stroke_input_data = strokes_train_data[stroke_idx]
					train_input[idx, :stroke_input_data.shape[0], :] = stroke_input_data

					stroke_target_data = strokes_target_data[stroke_idx]
					train_target[idx, :stroke_target_data.shape[0], :] = stroke_target_data

					train_mask[idx, :stroke_input_data.shape[0]].fill(1.)

					sentence_data = sentence_train_data[stroke_idx]
					train_sentence[idx, :sentence_data.shape[0], :] = sentence_data

					train_sentence_mask[idx, :sentence_data.shape[0]].fill(1.)

					cur += 1

				loss, train_step = self.sess.run([self.loss, self.optim], feed_dict = {
															self.input:    train_input,
															self.target:   train_target,
															self.text:	   train_sentence,	
															self.mask:	   train_mask,
															self.sen_mask: train_sentence_mask})
				self.sess.run([self.reset_state_op])
				
				print('batch_idx=%d train-loss=%.2f;' % (batch_idx, loss))
				total_loss_per_epoch += loss

			print('epoch=%d train-loss=%.2f;' % (epoch, total_loss_per_epoch))
			if total_loss_per_epoch < best_loss:
				best_loss = total_loss_per_epoch
				saver.save(self.sess, self.saved_model_directory+"/cond", global_step = epoch)

 

	def generate(self, length, sentence_data):
		''' function to generate conditional output for a given text_sequence '''

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
		text_data = np.array([sentence_data])
		sentence_mask = np.ones([1, sentence_data.shape[0]])

		stroke_data[0][0][0] = 1

		init_state = [self.sess.run(state) for state in self.init_state]

		for stroke_idx in range(length-1):
			init_state, state_trans_op, pi, mu1, mu2, sigma1, sigma2, rho, e = self.sess.run([self.layer_LSTM_state, self.assign_state_op, self.pi, self.mu1, self.mu2, self.sigma1, self.sigma2,
																		 self.rho, self.e],
																	 	feed_dict = {
																		self.input:   stroke_data[:, stroke_idx: stroke_idx+1, :],
																		self.text:	text_data,	
																		self.mask:	mask_data[:, stroke_idx: stroke_idx+1],
																		self.sen_mask:sentence_mask})

			is_finished = init_state[-1][0][0]
			stroke_data[0, stroke_idx+1, :] = generate_sample((pi, mu1, mu2, sigma1, sigma2, rho, e))

			if is_finished:
				stroke_data = stroke_data[:, :stroke_idx+2, :]
				break
																		
		return stroke_data[0]

