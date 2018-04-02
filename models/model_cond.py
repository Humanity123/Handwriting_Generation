import tensorflow as tf
import numpy as np
import math


class RNNModel(tf.nn.rnn_cell.RNNCell):
	''' class for custom RNN cell, inherits from RNN cell 

		Attributes:
				LSTM_layers (int): number of stacked LSTM layers in the RNN.
				num_units (int): number of LSTM units in a LSTM cell. Also equal to the output dimension for a timestep 
						a sequnce in the batch.
				window_fn: function which generates the window for the input text. It depends on the input text and previous kappa.
						Refer to https://arxiv.org/pdf/1308.0850.pdf) by Alex Graves for the definition of window.
				window_output_size (int): output dimension of the window_fn. It is equal to the sum of size of kappa and window.
						size of window is window_dim (3 * window_mixture_components), size of kappa is (window_mixture_components).
	'''

	def __init__(self, LSTM_layers, num_units, window_fn, window_output_size):
		''' initialisation function for class '''

		super(RNNModel, self).__init__()
		self.LSTM_layers = LSTM_layers
		self.num_units = num_units
		self.window_fn = window_fn
		self.window_output_size = window_output_size

		with tf.variable_scope('rnn', reuse=None):
			self.lstms = [tf.nn.rnn_cell.LSTMCell(
				num_units) for _ in range(self.LSTM_layers)]

	@property
	def state_size(self):
		return [self.num_units] * self.LSTM_layers * 2 + self.window_output_size

	@property
	def output_size(self):
		return [self.num_units]

	def call(self, inputs, state, **kwargs):
		''' function to execute call method of RNN cell.
			It is abstract function in the class RNN cell.

			Takes in the initial state and the input and passes it through stacked LSTM layers. The output of a LSTM layer is 
			passed to the next/upper LSTM layer. The input is fed to every LSTM layer through skip connections. It returns the 
			output of the final LSTM layer and a tuple containing the output state of all the LSTM layers combined with the current]
			window and kappa.
			The output of the first LSTM layer is used to compute the current window and kappa and is_finished using the window_fn.
			The is_finished value is ignored during training.
			The is_finished value is used during sampling of strokes. It tells if all the strokes have been generated for the input text
			or not and if it is time to stop the sampling or not.

			Args:
					input: float32 `Tensor` vector of size [batch_size, 3] which is the stroke data of a timestep.
					state: tuple of size (2 * LSTM_layers + 1) containing 2 * LSTM_layers float32 'Tensor' vector 
							of size [batch_size, num_units] corresponding to the initial state for the RNN cell
							and 1 float32 'Tensor' vector of size [batch_size, window_output_size] corresponding to the initial
							window and kappa and is_finished.

			Returns:
					output: float32 `Tensor` vector of size [batch_size, num_units]. The final of the RNN cell (stacked LSTM cells).
					output_state: tuple of size (LSTM_layers * 2 containing + 1). It contains LSTM_layers * 2 float32 `Tensor` vector
					of size [batch_size, num_units] corresponding to the output state of the RNN cell after processing the input and 1
					float32 `Tensor` vector of size [batch_size, window_output_size] corresponding to the current window, kappa and is_finished

		'''

		window, kappa, is_finished = state[-3:]
		output_state = []
		prev_output = 0

		for layer in range(self.LSTM_layers):
			if layer == 0:
				x = tf.concat([inputs, window], axis=1)
			else:
				x = tf.concat([inputs, window, prev_output], axis=1)

			with tf.variable_scope('lstm_{}'.format(layer)):
				output, next_state = self.lstms[layer](
					x, tf.nn.rnn_cell.LSTMStateTuple(state[2 * layer], state[2 * layer + 1]))
				prev_output = output

			output_state += [next_state[0]] + [next_state[1]]

			if layer == 0:
				window, kappa, is_finished = self.window_fn(output, kappa)

		return output, output_state + [window, kappa, is_finished]


class SynNet(object):
	''' class for Conditional Handwriting Synthesis Network

		Attributes:
				sess : tensorflow session for the current instantiation of model.
				seq_len (int): length of the input sequence i.e length of the stroke sequence.
				LSTM_layers (int): number of stacked LSTM layers in network.
				mixture_components (int): number of components in the mixture density network.
				window_mixture_components (int): number of components in the window mixture density network.
				batch_size (int): number of training points in the batch.
				LSTM_outdim (int): number of LSTM units in the LSTM cell.
				init_lr (float):  initial learning rate.
				decay (float):  decay rate for exponential decay of learning rate.
				grad_clip (float):  value of the gradient clip. All the gradients are clipped between (- grad_clip, grad_clip).
				epochs (int):  number of epochs during training.
				eps (float):  epsillon. Etremely small values are clipped at eps.
				bias (float):  bias used in prediction/generation of stroke samples. Value between [0, 1].
						Increasing the value increases readability but reduces variation in handwriting style.
				RNN_outkeep_prob (float): dropout probability for output of RNN.
				saved_model_directory (string): path to directory to save to trained model.
				mixture_dim (int): number corresponding to total number of parameters of mixture density network.
						mixture_dim = 6 * mixture_components + 1
	 '''

	def __init__(self, config, sess, training=True):
		''' initialisation function for class 

			Args:
					config   : tf.app.flags.FLAGS object containing the model arguments and parameters
					sess	 : tensorflow session for the current instantiation of model
					training (Boolean): Boolean variable. True if the model is training. Defaults to True
		'''

		self.sess = sess
		self.seq_len = config.cond_seq_len
		self.sen_len = config.cond_sen_len
		self.char_dim = config.cond_char_dim
		self.LSTM_layers = config.cond_LSTM_layers
		self.mixture_components = config.cond_mixture_components
		self.window_mixture_components = config.cond_window_mixture_components
		self.batch_size = config.cond_batch_size
		self.LSTM_outdim = config.cond_LSTM_outdim
		self.init_lr = config.cond_lr
		self.decay = config.cond_decay
		self.grad_clip = config.cond_grad_clip
		self.epochs = config.cond_epochs
		self.eps = config.cond_eps
		self.bias = 0.
		self.RNN_outkeep_prob = config.cond_RNN_outkeep_prob
		self.saved_model_directory = config.cond_saved_model_directory

		if not training:
			self.bias = config.cond_bias
			self.RNN_outkeep_prob = 1.
			self.batch_size = 1
			self.seq_len = 1

		self.mixture_dim = 1 + self.mixture_components * 6
		self.window_dim = self.window_mixture_components * 3
		self.RNN_outdim = self.LSTM_outdim

	def build_model(self):
		''' function to build the model '''

		self.global_step = tf.Variable(0, name='global_step', trainable=False)

		def normal_bivariate(x1, x2, mu1, mu2, sigma1, sigma2, rho):
			''' calculates the bivariate normal function from the distribution parameters 

			Z = (x1 - mu1)^2 / sigma1^2 + (x2 - mu2)^2 / sigma2^2 - 2 * rho * (x1 - mu1) * (x2 - mu2) / (sigma1 * sigma2)
			C = 1 / (1 - rho^2)
			N(x | mu, sigma, rho) = (1 / (2 * pi * sigma1 * sigma2)) * sqrt(C) * exp(-1 * Z * C / 2)

			Args:
					x1:  float32 `Tensor` vector of size [batch_size, seq_len, mixture_components].
					x2:  float32 `Tensor` vector of size [batch_size, seq_len, mixture_components].
					mu1: float32 `Tensor` vector of size [batch_size, seq_len, mixture_components]. 
							Corresponds to mean1 in normal bivariate distribution.
					mu2: float32 `Tensor` vector of size [batch_size, seq_len, mixture_components]. 
							Corresponds to mean2 in normal bivariate distribution.
					sigma1: float32 `Tensor` vector of size [batch_size, seq_len, mixture_components]. 
							Corresponds to sigma1 in normal bivariate distribution.
					sigma2: float32 `Tensor` vector of size [batch_size, seq_len, mixture_components]. 
							Corresponds to sigma2 in normal bivariate distribution.
					rho: float32 `Tensor` vector of size [batch_size, seq_len, mixture_components]. 
							Corresponds to rho in normal bivariate distribution.

			Returns:
					float32 `Tensor` vector of size [batch_size, seq_len, mixture_components]

			'''

			cons_pi = tf.constant(math.pi)
			norm_x1 = (x1 - mu1) / sigma1
			norm_x2 = (x2 - mu2) / sigma2
			Z = tf.square(norm_x1) + tf.square(norm_x2) - \
				(2. * rho * (norm_x1 * norm_x2))
			C = 1.0 / tf.maximum((1.0 - tf.square(rho)),  self.eps)

			normal_bivariate_prob = (
				1.0 / (2. * cons_pi * sigma1 * sigma2)) * tf.sqrt(C) * tf.exp((-1 * Z) * C / 2.)
			return normal_bivariate_prob

		def get_mixture_parameters(y, bias=0.):
			''' computes the parameters of the mixture density network from the RNN output

				The parameters are mu1, mu2, sigma1, sigma2, rho, e.

				(mu1, mu2, sigma1, sigma2, rho) are the parameters of the normal bivariate distribution.
				e is the parameter for bernoulli distribution


				Args:
						y: float32 `Tensor` vector of the size [batch_size, seq_len, mixture_dim]. 
								Corresponds to output of RNN containing all the parameters of mixture density network.
						bias (float): bias used in prediction/generation of stroke samples. 
								Value between [0, 1]. Defaults to 0.

				Returns:
						A tuple of (pi, mu1, mu2, sigma1, sigma2, rho, e). All the elements of the tuple are 
						float32 `Tensor` of shape [batch_size, seq_len, mixture_components]
			 '''

			mc = self.mixture_components

			pi = tf.nn.softmax(y[:, :, :mc] * (1 + bias))
			mu1 = y[:, :, mc: 2 * mc]
			mu2 = y[:, :, 2 * mc: 3 * mc]
			sigma1 = tf.maximum(
				tf.exp(y[:, :, 3 * mc: 4 * mc] - bias), self.eps)
			sigma2 = tf.maximum(
				tf.exp(y[:, :, 4 * mc: 5 * mc] - bias), self.eps)
			rho = tf.tanh(y[:, :, 5 * mc: 6 * mc])
			e = 1. / (1 + tf.exp(y[:, :, 6 * mc:]))

			return pi, mu1, mu2, sigma1, sigma2, rho, e

		def get_window_parameters(window_parameters, prev_kappa):
			''' computes the window parameters for the text string 

				The parameters are alpha, beta, kappa

				Args:
						window_parameters: float32 `Tensor` vector of size [batch_size, window_dim]
								Corresponds to the output of the first layer of RNN. contains concatenated window
								parameters.
						prev_kappa: float32 `Tensor` vector of size [batch_size, window_mixture_components]
								Corresponds to the kappa computed in previous time step. As the parameters output
								by RNN corresponds to the offset in kappa, we need previous kappa to get the actual 
								value of kappa.

				Returns:
						A tuple of size 3 with float32 `Tensor` of size [batch_size, window_mixture_components].
						which corresponds to the paramers alpha, beta, kappa respectively.

			'''

			wmc = self.window_mixture_components

			alpha = tf.exp(window_parameters[:, :wmc])
			beta = tf.exp(window_parameters[:, wmc: 2 * wmc])
			kappa = prev_kappa + tf.exp(window_parameters[:, 2 * wmc:])

			return alpha, beta, kappa

		def get_phi(alpha, beta, kappa, u):
			''' computes phi(character weight) for a character in the text string at index u

				phi[t, u] = sum(alpha[t, k] * exp(- beta[t, k] * (kappa[t, k] - u)^2) for k = 1 to window_mixture_components )

				Args:
						alpha: a float32 `Tensor` of size [batch_size, window_mixture_components], controls the importance of window
								within the mixture.
						beta: a float32 `Tensor` of size [batch_size, window_mixture_components], controls the width of window.
						kappa: a float32 `Tensor` of size [batch_size, window_mixture_components], controls the location of window.
						u: a scalar value which corresponds to an index in the sentence array. Value between 0 and sen_len - 1.

				Returns:
						phi: a float32 `Tensor` of size [batch_size] which corresponds to phi[t, u] which is basically the weight of 
								the window of the character at index u.

			'''

			phi = tf.multiply(
				alpha, tf.exp(-1 * tf.multiply(beta, tf.square(kappa - u))))
			reduced_phi = tf.reduce_sum(phi, axis=1)
			return reduced_phi

		def get_loss(normal_bivariate_prob, pi, eos, e, mask):
			''' function to calculate the loss 

				Loss = sum( -log( Pr(x[t+1] | y[t]) ) for t = 1 to seq_len )

				Refer to https://arxiv.org/pdf/1308.0850.pdf) by Alex Graves for definition of Pr(x[t+1] | y[t]).

				batch may contain sequences of different lengh, therefore a mask is passed which contains 1 for valid elements
				in a sequence and 0. for padded zeros.

				Args:
						normal_bivariate_prob: float32 `Tensor` vector of shape [batch_size, seq_len, mixture_components]
								corresponding to the normal bivaiate probability.
						pi: float32 `Tensor` vector of shape [batch_size, seq_len, mixture_components]
										corresponding to the pi parameter in mixture density network.
						eos: float32 `Tensor` vector of shape [batch_size, seq_len, mixture_components]
										corresponding to wheather pen was lifted at timestep t+1 or not.
						e: float32 `Tensor` vector of shape [batch_size, seq_len, mixture_components]
										corresponding to the e parameter in mixture density network.
						mask: float32 `Tensor` vector of shape [batch_size, seq_len]. Corresponds to 
										wheather if an element in a stroke sequence is valid or is a padded 0.
										If it is a padded 0 its contribution to the loss is neglected.

				Returns:
						float32 scalar value corresponding to the total log loss of the batch.
			'''

			eos_prob = tf.multiply(eos, e) + tf.multiply((1. - eos), (1. - e))
			eos_prob_2dim = tf.reshape(
				eos_prob, [self.batch_size, self.seq_len])
			self.check1 = eos_prob_2dim

			pi_prob = tf.multiply(pi, normal_bivariate_prob)
			self.check2 = pi_prob
			reduced_pi_prob = tf.reduce_sum(pi_prob, axis=2)
			self.check3 = reduced_pi_prob

			bernoulli_log_loss = tf.log(tf.maximum(eos_prob_2dim, self.eps))
			gaussian_log_loss = tf.log(tf.maximum(reduced_pi_prob, self.eps))
			masked_bernoulli_log_loss = tf.multiply(mask, bernoulli_log_loss)
			masked_gaussian_log_loss = tf.multiply(mask, gaussian_log_loss)
			total_loss_per_timestep = masked_gaussian_log_loss + masked_bernoulli_log_loss
			total_loss = tf.reduce_sum(total_loss_per_timestep, axis=1)
			self.see = total_loss
			reduced_total_loss = -1 * tf.reduce_sum(total_loss, axis=0)

			return reduced_total_loss

		def get_window(input, prev_kappa, text, hidden_W, hidden_b, sen_mask):
			''' function to get the window 

				Refer to https://arxiv.org/pdf/1308.0850.pdf) by Alex Graves for definition of window.

				Args:
						input: float32 `Tensor` vector of size [batch_size, LSTM_outdim] corresponding to the 
								output of the first layer of RNN for a timestep.
						prev_kappa: float32 `Tensor` vector of size [batch_size, window_mixture_components] corresponding
								to the kappa of previous timestep.
						text: float32 `Tensor` vector of size [batch_size, sen_len, char_dim] corresponding to the sentence.
						hidden_W: float32 `Tensor` vector of size [LSTM_outdim, window_dim] which converts the output of the LSTM_layer
								to window parameter space.
						hidden_b: float32 `Tensor` vector of size [window_dim].
						sen_mask: float32 `Tensor` vector of size [batch_size, sen_len]. Corresponds to 
										wheather if an element in a sentence is valid or is a padded 0.
										If it is a padded 0 its contribution to the window is neglected.

				Returns:
						a tuple of size 3 consisting of 
						window: float32 `Tensor` vector of size [batch_size, window_dim]
						kappa: float32 `Tensor` vector of size [batch_size, window_mixture_components]
						is_finish_2_dim: int `Tensor` vector of size [batch_size, 1] which signals wheather to stop generation of 
								strokes or not. Used during synthesis of handwriting for a given text.
			'''

			hidden_b_2dim = tf.reshape(hidden_b, [1, self.window_dim])
			til_hidden_b = tf.tile(hidden_b_2dim, [self.batch_size, 1])
			raw_window_params = tf.matmul(input, hidden_W) + til_hidden_b

			alpha, beta, kappa = get_window_parameters(
				raw_window_params, prev_kappa)

			char_indices = tf.constant(range(self.sen_len), dtype=tf.float32)
			phi_fn = lambda x: get_phi(alpha, beta, kappa, x)
			phi = tf.transpose(tf.map_fn(phi_fn, char_indices), perm=[1, 0])

			is_finish = tf.cast(phi_fn(self.sen_len) > tf.reduce_max(
				phi[:, :], axis=1), tf.float32)
			is_finish_2_dim = tf.reshape(is_finish, [self.batch_size, 1])

			phi_3dim = tf.reshape(phi, [self.batch_size, self.sen_len, 1])
			til_phi = tf.tile(phi_3dim, [1, 1, self.char_dim])

			w = tf.multiply(til_phi, text)

			sen_mask_3dim = tf.reshape(
				sen_mask, [self.batch_size, self.sen_len, 1])
			til_sen_mask = tf.tile(sen_mask_3dim, [1, 1, self.char_dim])

			masked_w = tf.multiply(til_sen_mask, w)
			reduced_w = tf.reduce_sum(masked_w, axis=1)

			return reduced_w, kappa, is_finish_2_dim

		self.input = tf.placeholder(
			tf.float32, [self.batch_size, self.seq_len, 3], name='input')
		self.target = tf.placeholder(
			tf.float32, [self.batch_size, self.seq_len, 3], name='target')
		self.text = tf.placeholder(
			tf.float32, [self.batch_size, self.sen_len, self.char_dim], name='target')
		self.mask = tf.placeholder(
			tf.float32, [self.batch_size, self.seq_len], name='mask')
		self.sen_mask = tf.placeholder(
			tf.float32, [self.batch_size, self.sen_len], name='sen_mask')

		self.graves_initializer = tf.truncated_normal_initializer(
			mean=0., stddev=.075, seed=None, dtype=tf.float32)

		hidden1_W = tf.get_variable('hidden1_W', [
									self.LSTM_outdim, self.window_dim], initializer=self.graves_initializer)
		hidden1_b = tf.get_variable(
			'hidden1_b', [self.window_dim], initializer=self.graves_initializer)
		hidden2_W = tf.get_variable('hidden2_W', [
									self.RNN_outdim, self.mixture_dim], initializer=self.graves_initializer)
		hidden2_b = tf.get_variable(
			'hidden2_b', [self.mixture_dim], initializer=self.graves_initializer)

		window_fn = lambda x, y: get_window(
			x, y, self.text, hidden1_W, hidden1_b, self.sen_mask)
		window_fn_outdim = [self.char_dim, self.window_mixture_components, 1]

		ML_LSTM_cell = RNNModel(
			self.LSTM_layers, self.LSTM_outdim, window_fn, window_fn_outdim)

		# Zero state for the RNN
		ML_LSTM_zero_states = ML_LSTM_cell.zero_state(
			self.batch_size, tf.float32)
		# Variable which corresponds to the initial state of the RNN
		self.init_state = [tf.Variable(state, trainable=False)
						   for state in ML_LSTM_zero_states]

		rnn_tuple_state = self.init_state

		dropped_out_LSTM_cell = tf.nn.rnn_cell.DropoutWrapper(
			ML_LSTM_cell, output_keep_prob=self.RNN_outkeep_prob)

		# ML_LSTM_output is of shape [batch_size, seq_len, LSTM_outdim]
		ML_LSTM_output, self.layer_LSTM_state = tf.nn.dynamic_rnn(
			dropped_out_LSTM_cell, self.input, initial_state=rnn_tuple_state)

		# Tensor op node to assign next state to previous state of the RNN.
		# used during generation of handwriting
		self.assign_state_op = tf.group(*[tf.assign(prev_state_tensor, next_state_tensor)
										  for prev_state_tensor, next_state_tensor in zip(self.init_state, self.layer_LSTM_state)])

		# Tensor op node to reset initial state of RNN to zero state.
		self.reset_state_op = tf.group(*[tf.assign(state_tensor, zero_state_tensor)
										 for state_tensor, zero_state_tensor in zip(self.init_state, ML_LSTM_zero_states)])

		# Converting RNN output to 4 dimenstions [batch_size, seq_len, 1,
		# LSTM_outdim]
		ML_LSTM_output_4dim = tf.reshape(
			ML_LSTM_output, [self.batch_size, self.seq_len, 1, self.RNN_outdim])

		# Converting ML_LSTM_output_W to shape[batch_size, seq_len, LSTM_outdim, mixture_dim] b reshaping and
		# tiling
		output_W_4dim = tf.reshape(
			hidden2_W, [1, 1, self.RNN_outdim, self.mixture_dim])
		til_output_W = tf.tile(
			output_W_4dim, [self.batch_size, self.seq_len, 1, 1])

		# Converting ML_LSTM_output_b to shape[batch_size, seq_len, mixture_dim] b reshaping and
		# tiling
		output_b_3dim = tf.reshape(hidden2_b, [1, 1, self.mixture_dim])
		til_output_b = tf.tile(
			output_b_3dim, [self.batch_size, self.seq_len, 1])

		W_multiplied_out = tf.squeeze(
			tf.matmul(ML_LSTM_output_4dim, til_output_W))

		y = W_multiplied_out + til_output_b
		self.pi, self.mu1, self.mu2, self.sigma1, self.sigma2, self.rho, self.e = get_mixture_parameters(
			y, self.bias)

		eos, x1, x2 = tf.split(axis=2, num_or_size_splits=3, value=self.target)

		til_x1 = tf.tile(x1, [1, 1, self.mixture_components])
		til_x2 = tf.tile(x2, [1, 1, self.mixture_components])
		self.eos = eos

		normal_bivariate_prob = normal_bivariate(
			til_x1, til_x2, self.mu1, self.mu2, self.sigma1, self.sigma2, self.rho)  # batch time matrix_com
		self.check4 = normal_bivariate_prob
		self.loss = get_loss(normal_bivariate_prob, self.pi, eos,
							 self.e, self.mask) / (self.batch_size * self.seq_len)

		self.lr = tf.Variable(self.init_lr, trainable=False)
		learning_rate = tf.train.exponential_decay(
			self.lr, self.global_step, staircase=True, decay_steps=1000, decay_rate=self.decay)
		self.opt = tf.train.AdamOptimizer(learning_rate=learning_rate)

		tvars = tf.trainable_variables()
		grads_and_vars = self.opt.compute_gradients(self.loss, tvars)
		self.g = grads_and_vars

		clipped_grads_and_vars = [(tf.clip_by_value(gv[0], -1 * self.grad_clip, self.grad_clip), gv[1])
								  for gv in grads_and_vars]

		inc = self.global_step.assign_add(1)
		with tf.control_dependencies([inc]):
			self.optim = self.opt.apply_gradients(
				clipped_grads_and_vars, global_step=self.global_step)

		self.sess.run(tf.initialize_all_variables())

	def train(self, strokes_train_data, strokes_target_data, sentence_train_data, saver):
		''' function to train the model 
			Takes input and target stroke data and sentence data for training.

			1)   x[t] -> RNN_layers[0] ->  out[t]
			2)	 (out[t], sentence[t]) -> window_fn -> window
			2)   (x[t]; window) -> RNN_layers[1:] -> y -> P(x[t+1] | y) 

			x[t] comes from strokes_train_data
			x[t+1] comes from strokes_target_data
			sentence[t] comes from sentence_train_data

			input strokes data is passed through the RNN layer to compute y.
			target train data is used along with y to calculate P(x | y) and compute loss.
			strokes are padded so that all the strokes are of length self.seq_len. A mask is
			passed to ignore contribution of loss due to padded zeros.
			sentences are padded so that all the sentences are of length self.sen_len. A sen_mask
			is passed to ignore contribution of padded zeros in the window.
			Model is saved after every epoch if the loss during that epoch is better than best loss till now.

			Args:
					strokes_train_data  : list of input stroke data arrays for training
					strokes_target_data : list of target strokes data array for training
					saver				: tensorflow saver object to store the model after every epoch

		'''

		train_input = np.ndarray(
			[self.batch_size, self.seq_len, 3], np.float32)
		train_target = np.ndarray(
			[self.batch_size, self.seq_len, 3], np.float32)
		train_mask = np.ndarray([self.batch_size, self.seq_len], np.float32)
		train_sentence = np.ndarray(
			[self.batch_size, self.sen_len, self.char_dim], np.float32)
		train_sentence_mask = np.ndarray(
			[self.batch_size, self.sen_len], np.float32)

		best_loss = np.inf

		for epoch in range(self.epochs):
			num_batches = int(
				math.ceil(len(strokes_train_data) / self.batch_size))
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
					train_input[idx, :stroke_input_data.shape[
						0], :] = stroke_input_data

					stroke_target_data = strokes_target_data[stroke_idx]
					train_target[idx, :stroke_target_data.shape[
						0], :] = stroke_target_data

					train_mask[idx, :stroke_input_data.shape[0]].fill(1.)

					sentence_data = sentence_train_data[stroke_idx]
					train_sentence[idx, :sentence_data.shape[
						0], :] = sentence_data

					train_sentence_mask[idx, :sentence_data.shape[0]].fill(1.)

					cur += 1

				loss, train_step = self.sess.run([self.loss, self.optim], feed_dict={
																	self.input:	train_input,
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
				saver.save(self.sess, self.saved_model_directory +
						   "/cond", global_step=epoch)

	def generate(self, length, sentence_data):
		''' function to generate strokes conditioned on the input text data.

			strokes are generated by the model one time step at a time until either is_finished
			becomes one or number of timesteps become greater than length.

			Args:
					length (int): length of the sample generated by the model.
					sentence_data (string): text data which is used to condition the handwriting
							generated by the model.

			Returns:
					Strokes sequence generated by the model of size [len, 3]. Suppose the timestep where is_finished 
					becomes 1 is equal to idx then len = min(idx, length).
		'''

		def generate_sample(parameters):
			''' generates a sample from the mixture model of bivariate gaussian and bernoulli

				parameters of the denisty mixture model are 
				(pi, mu1, mu2, sigma1, sigma2, rho): parameters of the normal bivariate distribution.
				e: parameter for bernoulli distribution

				Sampling for Bernoulli - 
						Select a random number, if e is bigger than that random number sample 1 else sample 0
				Sampling for Gaussian Bivariate - 
						There are mixture_components number of Gaussian Bivariate distributions in the mixture model.
						Select a random number, select the mixture component, mixture_idx, for which cumsum(pi)[idx] > random_number.
						Sample x1, x2 using the parameters of the Gaussian Bivariate distribution at the index mixture_idx. 


				Args:
						parameters: tuple containing the parameters that define the mixture model of mixture_components 
								number of bivariate gaussian and bernoulli distributions.

				Returns:
						Returns a tuple of size 3 which is the sample of the mixture density model.

			 '''

			pi, mu1, mu2, sigma1, sigma2, rho, e = parameters
			random_number = np.random.random()

			cumsum_pi = np.cumsum(pi)
			try:
				mixture_idx = next(x[0] for x in enumerate(
					cumsum_pi) if x[1] >= random_number)
			except:
				mixture_idx = self.mixture_components - 1
				print mixture_idx

			eos = 1 if e >= random_number else 0

			mix_pi, mix_mu1, mix_mu2, mix_sigma1, mix_sigma2, mix_rho = [
				param[0][0][mixture_idx] for param in parameters[:-1]]
			mean = [mix_mu1, mix_mu2]
			cov = [[mix_sigma1 * mix_sigma1, mix_rho * mix_sigma1 * mix_sigma2],
				   [mix_rho * mix_sigma1 * mix_sigma2, mix_sigma2 * mix_sigma2]]
			x1, x2 = np.random.multivariate_normal(mean, cov)

			return eos, x1, x2

		stroke_data = np.zeros([1, length, 3])
		mask_data = np.ones([1, length])
		text_data = np.array([sentence_data])
		sentence_mask = np.ones([1, sentence_data.shape[0]])

		stroke_data[0][0][0] = 0

		init_state = [self.sess.run(state) for state in self.init_state]

		for stroke_idx in range(length - 1):
			init_state, state_trans_op, pi, mu1, mu2, sigma1, sigma2, rho, e = self.sess.run([self.layer_LSTM_state, 
															self.assign_state_op, self.pi, self.mu1, self.mu2, self.sigma1, self.sigma2,
															self.rho, self.e], feed_dict={
																				self.input:   stroke_data[:, stroke_idx: stroke_idx + 1, :],
																				self.text:	text_data,
																				self.mask:	mask_data[:, stroke_idx: stroke_idx + 1],
																				self.sen_mask: sentence_mask})

			is_finished = init_state[-1][0][0]
			stroke_data[
				0, stroke_idx + 1, :] = generate_sample((pi, mu1, mu2, sigma1, sigma2, rho, e))

			if is_finished:
				stroke_data = stroke_data[:, :stroke_idx + 2, :]
				break

		return stroke_data[0]
