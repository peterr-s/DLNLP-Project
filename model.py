from enum import Enum

import tensorflow as tf
import numpy
from tensorflow.contrib import rnn

class Phase(Enum):
	Train = 0
	Validation = 1
	Predict = 2

class Model:
	def __init__(self, config, batch, lens_batch, label_batch, embedding_model, n_chars, numberer, phase = Phase.Predict):
		batch_size = batch.shape[1]
		input_size = batch.shape[2]
		label_size = label_batch.shape[2]
		
		# The integer-encoded words. input_size is the (maximum) number of
		# time steps.
		self._x = tf.placeholder(tf.int32, shape=[batch_size, input_size])

		# This tensor provides the actual number of time steps for each
		# instance.
		self._lens = tf.placeholder(tf.int32, shape=[batch_size])

		# The label distribution.
		if phase != Phase.Predict:
			self._y = tf.placeholder(
				tf.float32, shape=[batch_size, label_size])

		# convert to embeddings
		self._embedding_model = embedding_model
		self._numberer = numberer
		input_layer = tf.map_fn(self.sent_embed, self._x, dtype = tf.int32)
		# input_layer = self._x

		# make a bunch of LSTM cells and link them
		# use rnn.DropoutWrapper instead of tf.nn.dropout because the layers are anonymous
		stacked_LSTM = rnn.MultiRNNCell([rnn.DropoutWrapper(rnn.BasicLSTMCell(config.LSTM_sz), output_keep_prob = config.dropout_ratio) for _ in range(config.LSTM_ct)])
		
		# import pdb; pdb.set_trace()
		
		# run the whole thing
		_, hidden = tf.nn.dynamic_rnn(stacked_LSTM, input_layer, sequence_length = self._lens, dtype = tf.int32)
		w = tf.get_variable("W", shape=[hidden[-1].h.shape[1], label_size]) # if I understood the structure of MultiRNNCell correctly, hidden[-1] should be the final state
		b = tf.get_variable("b", shape=[1])
		logits = tf.matmul(hidden[-1].h, w) + b

		if phase == Phase.Train or Phase.Validation:
			losses = tf.nn.softmax_cross_entropy_with_logits(
				labels=self._y, logits=logits)
			self._loss = loss = tf.reduce_sum(losses)

		if phase == Phase.Train:
			start_lr = 0.01
			self._train_op = tf.train.AdamOptimizer(start_lr) \
				.minimize(losses)
			self._probs = probs = tf.nn.softmax(logits)

		if phase == Phase.Validation:
			# Highest probability labels of the gold data.
			hp_labels = tf.argmax(self.y, axis=1)

			# Predicted labels
			labels = tf.argmax(logits, axis=1)

			correct = tf.equal(hp_labels, labels)
			correct = tf.cast(correct, tf.float32)
			self._accuracy = tf.reduce_mean(correct)

	def sent_embed(self, sent_arr) :
		return tf.map_fn(self.get_embedding, sent_arr, dtype = tf.int32)
	
	def get_embedding(self, word_int) :
		word = self._numberer.value(word_int)
		if word in self._embedding_model.wv :
			print(word, self._embedding_model.wv[word])
			return (self._embedding_model.wv[word] * 100).astype(numpy.int32)
		else :
			return numpy.zeros(self._embedding_model.vector_size).astype(numpy.int32)
	
	@property
	def accuracy(self):
		return self._accuracy

	@property
	def lens(self):
		return self._lens

	@property
	def loss(self):
		return self._loss

	@property
	def probs(self):
		return self._probs

	@property
	def train_op(self):
		return self._train_op

	@property
	def x(self):
		return self._x

	@property
	def y(self):
		return self._y
