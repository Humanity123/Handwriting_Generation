import tensorflow as tf
from tensorflow.python import debug as tf_debug
import numpy as np
import math


class RNNModel(tf.nn.rnn_cell.RNNCell):
	''' class for custom RNN cell, inherits from RNN cell 

		Attributes:
			LSTM_layers (int): number of stacked LSTM layers in the RNN
			num_units (int): number of LSTM units in a LSTM cell. Also equal to the output dimension for a timestep 
				a sequnce in the batch
	'''

	def __init__(self, LSTM_layers, num_units):
		''' initialisation function for class '''

		super(RNNModel, self).__init__()
		self.LSTM_layers = LSTM_layers
		self.num_units = num_units

		with tf.variable_scope('rnn', reuse=None):
			self.lstms = [tf.nn.rnn_cell.LSTMCell(
				num_units) for _ in range(self.LSTM_layers)]

	@property
	def state_size(self):
		return [self.num_units] * self.LSTM_layers * 2

	@property
	def output_size(self):
		return [self.num_units]

	def call(self, input, state, **kwargs):
		''' function to execute call method of RNN cell.
			It is abstract function in the class RNN cell.

			Takes in the initial state and the input and passes it through stacked LSTM layers. The output of a LSTM layer is 
			passed to the next/upper LSTM layer. The input is fed to every LSTM layer through skip connections. It returns the 
			output of the final LSTM layer and the output state of all the LSTM layers combined in a tuple.

			Args:
				input: float32 `Tensor` vector of size [batch_size, 3] which is the stroke data of a timestep.
				state: tuple of size 2 * LSTM_layers containing float32 'Tensor' vector of size [batch_size, num_units].
					Corresponds to the initial state for the RNN cell.

			Returns:
				output: float32 `Tensor` vector of size [batch_size, num_units]. The final of the RNN cell (stacked LSTM cells).
				output_state: tuple of size LSTM_layers * 2 containing float32 `Tensor` vector of size [batch_size, num_units].
					Corresponds to the output state of the RNN cell after processing the input.

		'''

		output_state = []
		prev_output = []
		for layer in range(self.LSTM_layers):
			if layer == 0:
				x = input
			else:
				x = tf.concat([input, prev_output], axis=1)

			with tf.variable_scope('lstm_{}'.format(layer)):
				output, s = self.lstms[layer](x, tf.nn.rnn_cell.LSTMStateTuple(
					state[2 * layer], state[2 * layer + 1]))
				prev_output = output

			output_state += [s[0]] + [s[1]]

		return output, output_state


class PredNet(object):
	''' class for Unconditional Handwriting Prediction Network

		Attributes:
			sess : tensorflow session for the current instantiation of model.
			seq_len (int): length of the input sequence i.e length of the stroke sequence.
			LSTM_layers (int): number of stacked LSTM layers in network.
			mixture_components (int): number of components in the mixture density network.
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
		self.seq_len = config.uncond_seq_len
		self.LSTM_layers = config.uncond_LSTM_layers
		self.mixture_components = config.uncond_mixture_components
		self.batch_size = config.uncond_batch_size
		self.LSTM_outdim = config.uncond_LSTM_outdim
		self.init_lr = config.uncond_lr
		self.decay = config.uncond_decay
		self.grad_clip = config.uncond_grad_clip
		self.epochs = config.uncond_epochs
		self.eps = config.uncond_eps
		self.bias = 0.
		self.RNN_outkeep_prob = config.uncond_RNN_outkeep_prob
		self.saved_model_directory = config.uncond_saved_model_directory

		if not training:
			self.bias = config.uncond_bias
			self.batch_size = 1
			self.seq_len = 1
			self.RNN_outkeep_prob = 1.0

		self.mixture_dim = 1 + self.mixture_components * 6

	def build_model(self):
		''' function to generate the tensorflow model '''

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
			C = 1.0 / (1.0 - tf.square(rho) + self.eps)

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
			sigma1 = tf.exp(y[:, :, 3 * mc: 4 * mc] - bias) + self.eps
			sigma2 = tf.exp(y[:, :, 4 * mc: 5 * mc] - bias) + self.eps
			rho = tf.tanh(y[:, :, 5 * mc: 6 * mc])
			e = 1. / (1 + tf.exp(y[:, :, 6 * mc:]))
		
			return pi, mu1, mu2, sigma1, sigma2, rho, e

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

			pi_prob = tf.multiply(pi, normal_bivariate_prob)
			reduced_pi_prob = tf.reduce_sum(pi_prob, axis=2)

			bernoulli_log_loss = tf.log(tf.maximum(eos_prob_2dim, self.eps))
			gaussian_log_loss = tf.log(tf.maximum(reduced_pi_prob, self.eps))

			masked_bernoulli_log_loss = tf.multiply(mask, bernoulli_log_loss)
			masked_gaussian_log_loss = tf.multiply(mask, gaussian_log_loss)

			total_loss_per_timestep = masked_gaussian_log_loss + masked_bernoulli_log_loss
			total_loss_per_sequence = tf.reduce_sum(total_loss_per_timestep, axis=1)
			total_loss = -1 * tf.reduce_sum(total_loss_per_sequence, axis=0)

			return total_loss

		self.input = tf.placeholder(
			tf.float32, [None, self.seq_len, 3], name='input')
		self.target = tf.placeholder(
			tf.float32, [None, self.seq_len, 3], name='target')
		self.mask = tf.placeholder(
			tf.float32, [None, self.seq_len], name='mask')

		self.graves_initializer = tf.truncated_normal_initializer(
			mean=0., stddev=.075, seed=None, dtype=tf.float32)

		ML_LSTM_output_W = tf.get_variable('ML_LSTM_output_W', [
			self.LSTM_outdim, self.mixture_dim], initializer=self.graves_initializer)
		ML_LSTM_output_b = tf.get_variable(
			'ML_LSTM_output_b', [self.mixture_dim], initializer=self.graves_initializer)

		ML_LSTM_cell = RNNModel(
			self.LSTM_layers, self.LSTM_outdim)
		self.init_state = tf.identity(
			ML_LSTM_cell.zero_state(self.batch_size, tf.float32))

		rnn_tuple_state = tf.unstack(self.init_state, axis=0)

		dropped_out_LSTM_cell = tf.nn.rnn_cell.DropoutWrapper(
			ML_LSTM_cell, output_keep_prob=self.RNN_outkeep_prob)

		# ML_LSTM_output is of shape [batch_size, seq_len, LSTM_outdim]
		ML_LSTM_output, self.layer_LSTM_state = tf.nn.dynamic_rnn(
			dropped_out_LSTM_cell, self.input, initial_state=rnn_tuple_state) 

		# Converting RNN output to 4 dimenstions [batch_size, seq_len, 1, LSTM_outdim]
		ML_LSTM_output_4dim = tf.reshape(
			ML_LSTM_output, [self.batch_size, self.seq_len, 1, self.LSTM_outdim])

		# Converting ML_LSTM_output_W to shape[batch_size, seq_len, LSTM_outdim, mixture_dim] b reshaping and 
		# tiling
		output_W_4dim = tf.reshape(
			ML_LSTM_output_W, [1, 1, self.LSTM_outdim, self.mixture_dim])
		til_output_W = tf.tile(
			output_W_4dim, [self.batch_size, self.seq_len, 1, 1])

		# Converting ML_LSTM_output_b to shape[batch_size, seq_len, mixture_dim] b reshaping and 
		# tiling
		output_b_3dim = tf.reshape(ML_LSTM_output_b, [1, 1, self.mixture_dim])
		til_output_b = tf.tile(
			output_b_3dim, [self.batch_size, self.seq_len, 1])

		# W_multiplied_out_3dim is (ML_LSTM_output_4dim * til_output_w) reshaped to [batch_size, seq_len, mixture_dim]
		W_multiplied_out = tf.matmul(ML_LSTM_output_4dim, til_output_W)
		W_multiplied_out_3dim = tf.reshape(
			W_multiplied_out, [self.batch_size, self.seq_len, self.mixture_dim])

		y = W_multiplied_out_3dim + til_output_b

		self.pi, self.mu1, self.mu2, self.sigma1, self.sigma2, self.rho, self.e = get_mixture_parameters(
			y, self.bias)

		eos, x1, x2 = tf.split(axis=2, num_or_size_splits=3, value=self.target)

		til_x1 = tf.tile(x1, [1, 1, self.mixture_components])
		til_x2 = tf.tile(x2, [1, 1, self.mixture_components])

		normal_bivariate_prob = normal_bivariate(
			til_x1, til_x2, self.mu1, self.mu2, self.sigma1, self.sigma2, self.rho)

		self.loss = get_loss(normal_bivariate_prob, self.pi, eos,
							 self.e, self.mask) / (self.batch_size * self.seq_len)

		self.lr = tf.Variable(self.init_lr, trainable=False)
		learning_rate = tf.train.exponential_decay(
			self.lr, self.global_step, staircase=True, decay_steps=10000, decay_rate=0.5)
		self.opt = tf.train.AdamOptimizer(learning_rate=learning_rate)

		tvars = tf.trainable_variables()
		grads_and_vars = self.opt.compute_gradients(self.loss, tvars)
		clipped_grads_and_vars = [(tf.clip_by_value(gv[0], -1 * self.grad_clip, self.grad_clip), gv[1])
								  for gv in grads_and_vars]

		inc = self.global_step.assign_add(1)
		with tf.control_dependencies([inc]):
			self.optim = self.opt.apply_gradients(
				clipped_grads_and_vars, global_step=self.global_step)

		self.sess.run(tf.initialize_all_variables())

	def train(self, strokes_train_data, strokes_target_data, saver):
		''' function to train the model 
			Takes input and target stroke data for training.

			x[t] -> RNN -> y -> P(x[t+1] | y) 

			x[t] comes from strokes_train_data
			x[t+1] comes from strokes_target_data

			input strokes data is passed through the RNN layer to compute y.
			target train data is used along with y to calculate P(x | y) and compute loss.
			strokes are padded so that all the strokes are of length self.seq_len. A mask is
			passed to ignore contribution of loss due to padded zeros.
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
		train_mask = np.ndarray(
			[self.batch_size, self.seq_len], np.float32)

		best_loss = np.inf
		loss_list = []

		for epoch in range(self.epochs):
			num_batches = int(
				math.ceil(len(strokes_train_data) / self.batch_size))
			rand_idx, cur = np.random.permutation(len(strokes_train_data)), 0

			total_loss_per_epoch = 0.

			for batch_idx in range(num_batches):
				train_input.fill(0.)
				train_mask.fill(0.)
				train_target.fill(0.)

				for idx in range(self.batch_size):
					stroke_idx = rand_idx[cur]

					stroke_input_data = strokes_train_data[stroke_idx]
					train_input[idx, :stroke_input_data.shape[
						0], :] = stroke_input_data

					stroke_target_data = strokes_target_data[stroke_idx]
					train_target[idx, :stroke_target_data.shape[
						0], :] = stroke_target_data

					train_mask[idx, :stroke_input_data.shape[0]].fill(1.)
					cur += 1

					pi, mu1, mu2, sigma1, sigma2, rho, e, loss, o = self.sess.run([self.pi, self.mu1, self.mu2,
																					self.sigma1, self.sigma2, self.rho, self.e,
																					self.loss, self.optim], feed_dict={
					self.input: train_input,
					self.target: train_target,
					self.mask: train_mask})

				total_loss_per_epoch += loss
				print('batch_idx=%d train-loss=%.2f;' % (batch_idx, loss))

			print('epoch=%d train-loss=%.2f;' % (epoch, total_loss_per_epoch))
			if total_loss_per_epoch < best_loss:
				best_loss = total_loss_per_epoch
				saver.save(self.sess, self.saved_model_directory +
						   "/uncond", global_step=epoch)

	def synthesize(self, length):
		''' function to generate unconditional strokes 

			Args:
				length (int): length of the sample generated by the model.

			Returns:
				Strokes sequence of shape [length, 3] generated by the model.
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

		stroke_data[0][0][0] = 0

		init_state = self.sess.run(self.init_state)

		for stroke_idx in range(length - 1):
			init_state, pi, mu1, mu2, sigma1, sigma2, rho, e = self.sess.run([self.layer_LSTM_state, self.pi, self.mu1, self.mu2, self.sigma1,
																			self.sigma2, self.rho, self.e], feed_dict={
																			self.input: stroke_data[:, stroke_idx: stroke_idx + 1, :],
																			self.mask: mask_data[:, stroke_idx: stroke_idx + 1],
																			self.init_state: init_state})
			stroke_data[
				0, stroke_idx + 1, :] = generate_sample((pi, mu1, mu2, sigma1, sigma2, rho, e))

		return stroke_data[0]
