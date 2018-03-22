import tensorflow as tf
import math 


class MixtureDensityNetwork(object):
	''' class for Mixture Density Network '''

	def __init__(self, config, sess):
		''' initialisation function for class '''
		self.sess = sess
		self.seq_len = config.seq_len

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

		def loss_function


	def train():
		''' function to train the model '''
		pass

