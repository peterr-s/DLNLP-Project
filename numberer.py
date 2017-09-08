import numpy as np
import tensorflow as tf

class Numberer:
	def __init__(self, model, config):
		self.v2n = dict()
		self.n2v = list()
		self.model = model
		self.config = config

	def number(self, value, add_if_absent=True):
		n = np.zeros(self.config.embedding_sz)

		if value in self.model.wv.vocab.items() :
			n = self.model.wv[value]
		
		return tf.stack(n)

	def value(self, number):
		return model.wv.similar_by_vector(number, topn = 1)[0][0]

	def max_number(self):
		return len(self.model.wv.vocab) + 1
