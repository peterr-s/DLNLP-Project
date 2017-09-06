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
		self.n2v[number]

	def max_number(self):
		return len(self.n2v) + 1
