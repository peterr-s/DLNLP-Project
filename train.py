#!/usr/bin/python3

# Authors:	Peter Schoener, 4013996
#			Luana Vaduva, 3974913
# Honor Code: We pledge that this program represents our own work.

from enum import Enum
import os
import sys
import re

import numpy as np
import tensorflow as tf
import gensim

from bs4 import BeautifulSoup


from config import DefaultConfig
from model import Model, Phase
from numberer import Numberer

def is_english(text):
	eng_stopwords = [u'all', u'just', u'being', u'over', u'both', u'through', u'yourselves', u'its', u'before', u'hadn', u'herself', u'll', u'had', u'should', u'to', u'only', u'won', u'under', u'ours', u'has', u'do', u'them', u'his', u'very', u'they', u'not', u'during', u'now', u'him', u'nor', u'd', u'did', u'didn', u'this', u'she', u'each', u'further', u'where', u'few', u'because', u'doing', u'some', u'hasn', u'are', u'our', u'ourselves', u'out', u'what', u'for', u'while', u're', u'does', u'above', u'between', u'mustn', u't', u'be', u'we', u'who', u'were', u'here', u'shouldn', u'hers', u'by', u'on', u'about', u'couldn', u'of', u'against', u's', u'isn', u'or', u'own', u'into', u'yourself', u'down', u'mightn', u'wasn', u'your', u'from', u'her', u'their', u'aren', u'there', u'been', u'whom', u'too', u'wouldn', u'themselves', u'weren', u'was', u'until', u'more', u'himself', u'that', u'but', u'don', u'with', u'than', u'those', u'he', u'me', u'myself', u'ma', u'these', u'up', u'will', u'below', u'ain', u'can', u'theirs', u'my', u'and', u've', u'then', u'is', u'am', u'it', u'doesn', u'an', u'as', u'itself', u'at', u'have', u'in', u'any', u'if', u'again', u'no', u'when', u'same', u'how', u'other', u'which', u'you', u'shan', u'needn', u'haven', u'after', u'most', u'such', u'why', u'a', u'off', u'i', u'm', u'yours', u'so', u'y', u'the', u'having', u'once']
	words = text.replace("'", " ").split(" ")
	if len(set(words) & set(eng_stopwords)) > 5:
		return True

	
def preprocess(text):
    #convert to lower case
    text = text.lower()
    #convert www.* or https?://* to URL
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',text)
    #convert @username to AT_USER
    text = re.sub('@[^\s]+','AT_USER',text)
    #remove additional white spaces
    text = re.sub('[\s]+', ' ', text)
    #replace #word with word
    text = re.sub(r'#([^\s]+)', r'\1', text)
    #trim
    text = text.strip('\'"')
    #get rid of HTML markup
    soup = BeautifulSoup(text, "html5lib")
        
	return soup.get_text()


def read_lexicon(filename):
	with open(filename, "r") as f:
		lex = {}
		
		for line in f:
			fields = line.split("\t")
			lex[preprocess(fields[1])] = {fields[-1].strip()[3:]:1.0}
		return lex


def recode_lexicon(lexicon, words, labels, train=False):
	int_lex = []

	for (sentence, tags) in lexicon.items():
		int_sentence = []
		for word in sentence.split():
			int_sentence.append(words.number(word, train))

		int_tags = {}
		for (tag, p) in tags.items():
			int_tags[labels.number(tag, train)] = p

		int_lex.append((int_sentence, int_tags))

	return int_lex


def generate_instances(
		data,
		max_label,
		max_timesteps,
		batch_size=128):
	n_batches = len(data) // batch_size

	# We are discarding the last batch for now, for simplicity.
	labels = np.zeros(
		shape=(
			n_batches,
			batch_size,
			max_label.max_number()),
		dtype=np.float32)
	lengths = np.zeros(
		shape=(
			n_batches,
			batch_size),
		dtype=np.int32)
	sentences = np.zeros(
		shape=(
			n_batches,
			batch_size,
			max_timesteps),
		dtype=np.int32)

	for batch in range(n_batches):
		for idx in range(batch_size):
			(sentence, l) = data[(batch * batch_size) + idx]

			# Add label distribution
			for label, prob in l.items():
				labels[batch, idx, label] = prob

			# Sequence
			timesteps = min(max_timesteps, len(sentence))

			# Sequence length (time steps)
			lengths[batch, idx] = timesteps

			# Word characters
			sentences[batch, idx, :timesteps] = sentence[:timesteps]

	return (sentences, lengths, labels)


def train_model(config, train_batches, validation_batches, embedding_model, numberer):
	train_batches, train_lens, train_labels = train_batches
	validation_batches, validation_lens, validation_labels = validation_batches

	n_chars = max(np.amax(validation_batches), np.amax(train_batches)) + 1

	with tf.Session() as sess:
		with tf.variable_scope("model", reuse=False):
			train_model = Model(
				config,
				train_batches,
				train_lens,
				train_labels,
				embedding_model,
				n_chars,
				numberer,
				phase=Phase.Train)

		with tf.variable_scope("model", reuse=True):
			validation_model = Model(
				config,
				validation_batches,
				validation_lens,
				validation_labels,
				embedding_model,
				n_chars,
				numberer,
				phase=Phase.Validation)

		sess.run(tf.global_variables_initializer())

		for epoch in range(config.n_epochs):
			train_loss = 0.0
			validation_loss = 0.0
			accuracy = 0.0

			# Train on all batches.
			for batch in range(train_batches.shape[0]):
				loss, _ = sess.run([train_model.loss, train_model.train_op], {
					train_model.x: train_batches[batch], train_model.lens: train_lens[batch], train_model.y: train_labels[batch]})
				train_loss += loss

			# validation on all batches.
			for batch in range(validation_batches.shape[0]):
				loss, acc = sess.run([validation_model.loss, validation_model.accuracy], {
					validation_model.x: validation_batches[batch], validation_model.lens: validation_lens[batch], validation_model.y: validation_labels[batch]})
				validation_loss += loss
				accuracy += acc

			train_loss /= train_batches.shape[0]
			validation_loss /= validation_batches.shape[0]
			accuracy /= validation_batches.shape[0]

			print(
				"epoch %d - train loss: %.2f, validation loss: %.2f, validation acc: %.2f" %
				(epoch, train_loss, validation_loss, accuracy * 100))


if __name__ == "__main__":
	if len(sys.argv) != 4:
		sys.stderr.write("Usage: %s TRAIN_SET DEV_SET EMBEDDINGS\n" % sys.argv[0])
		sys.exit(1)

	config = DefaultConfig()

	# Read training, validation, and embedding data.
	train_lexicon = read_lexicon(sys.argv[1])
	validation_lexicon = read_lexicon(sys.argv[2])
	embedding_model = gensim.models.Word2Vec.load(sys.argv[3])

	# Convert word characters and part-of-speech labels to numeral
	# representation.
	words = Numberer()
	labels = Numberer()
	train_lexicon = recode_lexicon(train_lexicon, words, labels, train=True)
	validation_lexicon = recode_lexicon(validation_lexicon, words, labels)

	# Generate batches
	train_batches = generate_instances(
		train_lexicon,
		labels,#.max_number(),
		config.max_timesteps,
		batch_size=config.batch_size)
	validation_batches = generate_instances(
		validation_lexicon,
		labels,#.max_number(),
		config.max_timesteps,
		batch_size=config.batch_size)

	# Train the model
	train_model(config, train_batches, validation_batches, embedding_model, words)
