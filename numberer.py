class Numberer:
	def __init__(self, model):
		self.v2n = dict()
		self.n2v = list()
		self.model = model

	def number(self, value, add_if_absent=True):
		n = self.model.wv[value]

		if n is None:
			n = numpy.zeros(DefaultConfig.embedding_sz)

		return tf.pack(n)

	def value(self, number):
		return model.wv.similar_by_vector(number, topn = 1)[0][0]

	def max_number(self):
		return len(model.wv.vocab) + 1
